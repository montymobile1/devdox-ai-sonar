from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, TypedDict, Tuple, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

project_key = "Project key"


class BlockType(str, Enum):
    """Valid code block types."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"


class ChangeType(str, Enum):
    FULL_CODE = "FULL_CODE"  # Complete function (small functions)
    DIFF = "DIFF"  # Line-level changes
    SEARCH_REPLACE = "SEARCH_REPLACE"  # Pattern replacement


class PlacementType(str, Enum):
    """Valid placement options for NEW_HELPER_CODE."""

    SIBLING = "SIBLING"
    GLOBAL_TOP = "GLOBAL_TOP"
    GLOBAL_BOTTOM = "GLOBAL_BOTTOM"
    INSIDE_CLASS = "INSIDE_CLASS"


class ChangeAction(str, Enum):
    REPLACE = "replace"
    INSERT = "insert"
    DELETE = "delete"


class LineChange(BaseModel):
    line: int
    action: ChangeAction  # "replace", "insert", "delete"
    old: Optional[str] = None
    new: Optional[str] = None


class SearchReplace(BaseModel):
    search: str
    replace: str
    is_regex: bool = False
    count: Optional[int] = None  # None = replace all


class CodeBlock(BaseModel):
    block_name: str
    start_line: int
    end_line: int
    has_changes: bool
    change_type: ChangeType
    block_type: BlockType

    # For FULL_CODE
    context: Optional[str] = None

    # For DIFF
    changes: Optional[List[LineChange]] = None

    # For SEARCH_REPLACE
    replacements: Optional[List[SearchReplace]] = None
    file_path: Optional[str] = Field(None, description="Path to the file being fixed")


class SonarFixResponse(BaseModel):
    """
    Complete response format for SonarQube fix from LLM.

    This model ensures the LLM returns properly structured JSON.
    """

    IMPORT_BLOCK: str = Field(
        default="",
        description=(
            "Updated import section as a string with newlines (\\n), "
            "or empty string if no changes needed"
        ),
    )

    FIXED_CODE_BLOCKS: List[CodeBlock] = Field(
        ...,
        description=(
            "Array of code blocks. Each block represents a complete function, class, "
            "or module-level code section. Include all blocks provided in input, "
            "marking has_changes=true for modified blocks."
        ),
        min_length=1,
    )

    NEW_HELPER_CODE: str = Field(
        default="",
        description=(
            "Any new helper functions or constants to add, "
            "or empty string if none needed"
        ),
    )

    PLACEMENT: PlacementType = Field(
        default=PlacementType.GLOBAL_TOP,
        description=(
            "Where to place NEW_HELPER_CODE: "
            "SIBLING (after first modified function), "
            "GLOBAL_TOP (after imports), "
            "GLOBAL_BOTTOM (end of file), "
            "INSIDE_CLASS (within a class)"
        ),
    )

    EXPLANATION: Optional[str] = Field(
        default="",
        description=(
            "Detailed explanation following structured format: "
            "1. Issue Analysis, 2. Fix Strategy, 3. Implementation Details, 4. Validation"
        ),
    )

    CONFIDENCE: float = Field(
        ...,
        description=(
            "Confidence score between 0.0 and 1.0. "
            "0.9-1.0 = high confidence, "
            "0.7-0.89 = moderate confidence, "
            "0.5-0.69 = low confidence, "
            "<0.5 = very low confidence"
        ),
        ge=0.0,
        le=1.0,
    )

    @field_validator("FIXED_CODE_BLOCKS")
    @classmethod
    def validate_has_changes(cls, v: List[CodeBlock]) -> List[CodeBlock]:
        """Ensure at least one block has changes."""
        if not any(block.has_changes for block in v):
            raise ValueError("At least one code block must have has_changes=True")
        return v

    @model_validator(mode="after")
    def validate_helper_code_placement(self) -> "SonarFixResponse":
        """If NEW_HELPER_CODE exists, ensure PLACEMENT is set appropriately."""
        if self.NEW_HELPER_CODE and self.PLACEMENT == PlacementType.SIBLING:
            # Verify at least one function exists for SIBLING placement
            if not any(
                block.block_type in [BlockType.FUNCTION, BlockType.METHOD]
                and block.has_changes
                for block in self.FIXED_CODE_BLOCKS
            ):
                raise ValueError(
                    "SIBLING placement requires at least one modified function/method"
                )
        return self

    @field_validator("NEW_HELPER_CODE", "IMPORT_BLOCK", mode="before")
    @classmethod
    def clean_triple_quotes(cls, v: Any) -> Any:
        """Remove triple quotes from code strings."""
        if not isinstance(v, str):
            return v

        # Replace triple quotes with single quotes
        v = v.replace('"""', '"')
        v = v.replace("'''", "'")

        return v

    @field_validator("FIXED_CODE_BLOCKS", mode="before")
    @classmethod
    def clean_code_blocks(cls, v: Any) -> Any:
        """Clean triple quotes from code blocks."""
        if not isinstance(v, list):
            return v

        cleaned_blocks = []
        for block in v:
            if isinstance(block, dict) and isinstance(block.get("context"), str):
                block = {
                    **block,
                    "context": block["context"].replace('"""', '"').replace("'''", "'"),
                }
            cleaned_blocks.append(block)

        return cleaned_blocks

    class Config:
        json_schema_extra = {
            "example": {
                "IMPORT_BLOCK": "from typing import List, Optional\nimport re",
                "FIXED_CODE_BLOCKS": [
                    {
                        "context": "def validate_email(email: str) -> bool:\n    pattern = EMAIL_REGEX\n    return bool(re.match(pattern, email))",
                        "line_number": 45,
                        "start_line": 45,
                        "end_line": 47,
                        "is_complete_function": True,
                        "block_type": "function",
                        "block_name": "validate_email",
                        "has_changes": True,
                    },
                    {
                        "context": "def validate_phone(phone: str) -> bool:\n    pattern = PHONE_REGEX\n    return bool(re.match(pattern, phone))",
                        "line_number": 50,
                        "start_line": 50,
                        "end_line": 52,
                        "is_complete_function": True,
                        "block_type": "function",
                        "block_name": "validate_phone",
                        "has_changes": True,
                    },
                    {
                        "context": "class UserValidator:\n    def __init__(self):\n        self.email_validator = validate_email\n        self.phone_validator = validate_phone",
                        "line_number": 55,
                        "start_line": 55,
                        "end_line": 58,
                        "is_complete_function": True,
                        "block_type": "class",
                        "block_name": "UserValidator",
                        "has_changes": False,
                    },
                ],
                "NEW_HELPER_CODE": "EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\nPHONE_REGEX = r'^\\+?1?\\d{9,15}$'",
                "PLACEMENT": "GLOBAL_TOP",
                "EXPLANATION": "### 1. Issue Analysis\n- Sonar Rule: python:S1192\n- Root Cause: Duplicated string literals\n\n### 2. Fix Strategy\n- Extract to constants\n\n### 3. Implementation Details\n- Created EMAIL_REGEX and PHONE_REGEX constants\n\n### 4. Validation\n- Syntax confirmed\n- Logic preserved",
                "CONFIDENCE": 0.92,
            }
        }


class FixGraphState(TypedDict):
    # ── inputs (set once, never mutated) ────────────────────────────────────
    provider: str
    model: str
    api_key: str
    prompt_system: str
    original_prompt: str  # never modified — used for retry enrichment
    max_tokens: int
    temperature: float
    max_attempts: int

    # ── working state (mutated per iteration) ────────────────────────────────
    prompt: str  # may be enriched on retry
    fix_result: Optional[Any]  # SonarFixResponse | None
    validation_error: Optional[str]
    attempt: int


class SonarType(str, Enum):
    BRANCH = "branch"
    PR = "pr"


class Severity(str, Enum):
    """SonarCloud issue severity levels."""

    BLOCKER = "BLOCKER"
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"


class IssueType(str, Enum):
    """SonarCloud issue types."""

    BUG = "BUG"
    VULNERABILITY = "VULNERABILITY"
    CODE_SMELL = "CODE_SMELL"
    SECURITY_HOTSPOT = "SECURITY_HOTSPOT"


class Impact(str, Enum):
    """SonarCloud issue impact levels."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SonarCloudConfig(BaseModel):
    token: str = Field(..., description="SonarCloud authentication token")
    organization: str = Field(..., description="SonarCloud organization key")
    project: str = Field(..., description="SonarCloud project key")
    project_path: str = Field(..., description="Path to local project directory")
    git_url: str = Field(..., description="Git URL for the project")

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate SonarCloud configuration."""
        if not self.token or not self.token.strip():
            return False, "Token cannot be empty"

        if not self.organization or not self.organization.strip():
            return False, "Organization key cannot be empty"

        if not self.project or not self.project.strip():
            return False, "Project key cannot be empty"

        if not self.project_path:
            return False, "Project path cannot be empty"

        if not self.git_url or not self.git_url.strip():
            return False, "Git URL cannot be empty"
        if not self.git_url.endswith(".git"):
            return False, "Git URL must end with .git"

        return True, None


class SonarIssue(BaseModel):
    """Represents a SonarCloud issue."""

    key: str = Field(..., description="Unique issue key")
    rule: str = Field(..., description="Rule identifier")
    severity: Severity = Field(..., description="Issue severity")
    component: str = Field(..., description="Component path")
    project: str = Field(..., description=project_key)
    first_line: Optional[int] = Field(None, description="First Line number")
    last_line: Optional[int] = Field(None, description="Last Line number")
    problem_lines: Optional[List[int]] = Field(None, description="Problem lines")
    message: str = Field(..., description="Issue description")
    type: IssueType = Field(..., description="Issue type")
    impact: Optional[Impact] = Field(None, description="Issue impact")
    file: Optional[str] = Field(None, description="File path")
    branch: Optional[str] = Field(None, description="Branch name")
    status: str = Field(default="OPEN", description="Issue status")
    creation_date: Optional[str] = Field(None, description="Creation date")
    update_date: Optional[str] = Field(None, description="Last update date")
    tags: List[str] = Field(default_factory=list, description="Issue tags")
    effort: Optional[str] = Field(None, description="Effort to fix")
    debt: Optional[str] = Field(None, description="Technical debt")
    model_config = ConfigDict(populate_by_name=True)

    @property
    def file_path(self) -> Optional[Path]:
        """Get the file path as a Path object."""
        if self.file:
            return Path(self.file)
        return None

    @property
    def is_fixable(self) -> bool:
        """Check if the issue is potentially fixable by LLM."""
        fixable_types = {IssueType.BUG, IssueType.CODE_SMELL}
        return (
            self.type in fixable_types
            and self.first_line is not None
            and self.last_line is not None
        )


class SonarSecurityIssue(BaseModel):
    key: str = Field(..., description="Unique issue key")
    component: str = Field(..., description="Component path")
    rule: str = Field(..., description="Rule identifier")
    project: str = Field(..., description=project_key)
    security_category: str = Field(..., description="Security category")
    vulnerability_probability: str = Field(..., description="Vulnerability probability")
    status: str = Field(default="OPEN", description="Issue status")
    first_line: Optional[int] = Field(None, description="First Line number")
    last_line: Optional[int] = Field(None, description="Last Line number")
    problem_lines: Optional[List[int]] = Field(None, description="Problem lines")
    message: str = Field(..., description="Issue description")
    file: Optional[str] = Field(None, description="File path")
    creation_date: Optional[str] = Field(None, description="Creation date")
    update_date: Optional[str] = Field(None, description="Last update date")
    model_config = ConfigDict(populate_by_name=True)

    @property
    def file_path(self) -> Optional[Path]:
        """Get the file path as a Path object."""
        if self.file:
            return Path(self.file)
        return None


class ProjectMetrics(BaseModel):
    """SonarCloud project metrics."""

    project_key: str = Field(..., description=project_key)
    lines_of_code: Optional[int] = None
    coverage: Optional[float] = None
    duplicated_lines_density: Optional[float] = None
    maintainability_rating: Optional[str] = None
    reliability_rating: Optional[str] = None
    security_rating: Optional[str] = None
    bugs: Optional[int] = None
    vulnerabilities: Optional[int] = None
    code_smells: Optional[int] = None
    technical_debt: Optional[str] = None


class SecurityAnalysisResult(BaseModel):
    project_key: str = Field(..., description=project_key)
    organization: str = Field(..., description="Organization key")
    branch: str = Field(default="main", description="Branch analyzed")
    total_issues: int = Field(..., description="Total number of issues")
    issues: List[SonarSecurityIssue] = Field(..., description="List of issues")
    fixable_issues_by_file: Dict[str, List[Union[SonarIssue, SonarSecurityIssue]]] = (
        Field(
            default_factory=dict,
            description="Security issues that can be fixed by LLM",
        )
    )
    analysis_timestamp: Optional[str] = Field(None, description="Analysis timestamp")

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to set fixable issues."""
        for issue in self.issues:
            if issue.file:
                file_key = str(issue.file_path)
                self.fixable_issues_by_file.setdefault(file_key, []).append(issue)


class AnalysisResult(BaseModel):
    """Results from SonarCloud analysis."""

    project_key: str = Field(..., description=project_key)
    organization: str = Field(..., description="Organization key")
    branch: str = Field(default="main", description="Branch analyzed")
    total_issues: int = Field(..., description="Total number of issues")
    issues: List[SonarIssue] = Field(..., description="List of issues")
    metrics: Optional[ProjectMetrics] = Field(None, description="Project metrics")
    fixable_issues: List[SonarIssue] = Field(
        default_factory=list, description="Issues that can be fixed by LLM"
    )
    fixable_issues_by_file: Dict[str, List[SonarIssue]] = Field(
        default_factory=dict,
        description="Issues grouped by file that can be fixed by LLM",
    )

    analysis_timestamp: Optional[str] = Field(None, description="Analysis timestamp")

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to set fixable issues."""
        self.fixable_issues = [issue for issue in self.issues if issue.is_fixable]
        for issue in self.issues:
            if issue.file:
                file_key = str(issue.file_path)
                self.fixable_issues_by_file.setdefault(file_key, []).append(issue)

    @property
    def issues_by_severity(self) -> Dict[Severity, List[SonarIssue]]:
        """Group issues by severity."""
        result: Dict[Severity, List[SonarIssue]] = {
            severity: [] for severity in Severity
        }
        for issue in self.issues:
            result[issue.severity].append(issue)
        return result

    @property
    def issues_by_type(self) -> Dict[IssueType, List[SonarIssue]]:
        """Group issues by type."""
        result: Dict[IssueType, List[SonarIssue]] = {
            issue_type: [] for issue_type in IssueType
        }
        for issue in self.issues:
            result[issue.type].append(issue)
        return result


class ProcessedRules(TypedDict):
    rules: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Hotspot /api/hotspots/show response types
# ---------------------------------------------------------------------------


class HotspotComponent(TypedDict, total=False):
    organization: str
    key: str
    qualifier: str
    name: str
    longName: str
    path: str


class HotspotProject(TypedDict, total=False):
    organization: str
    key: str
    qualifier: str
    name: str
    longName: str


class HotspotRule(TypedDict, total=False):
    key: str
    name: str
    securityCategory: str
    vulnerabilityProbability: str


class HotspotChangelogDiff(TypedDict, total=False):
    key: str
    newValue: str
    oldValue: str


class HotspotChangelog(TypedDict, total=False):
    user: str
    userName: str
    creationDate: str
    diffs: List["HotspotChangelogDiff"]
    avatar: str
    isUserActive: bool


class HotspotComment(TypedDict, total=False):
    key: str
    login: str
    htmlText: str
    markdown: str
    createdAt: str


class HotspotUser(TypedDict, total=False):
    login: str
    name: str
    active: bool


class HotspotDetail(TypedDict, total=False):
    """Typed response from ``GET /api/hotspots/show``."""

    key: str
    component: HotspotComponent
    project: HotspotProject
    rule: HotspotRule
    status: str
    line: int
    message: str
    assignee: str
    author: str
    creationDate: str
    updateDate: str
    changelog: List[HotspotChangelog]
    comment: List[HotspotComment]
    users: List[HotspotUser]
    canChangeStatus: bool


class FixSuggestion(BaseModel):
    """Represents an LLM-generated fix suggestion."""

    issue_key: str = Field(..., description="Related issue key")
    original_code: str = Field(..., description="Original problematic code")
    fixed_code: str = Field(..., description="Suggested fix")
    fixed_code_blocks: List[CodeBlock] = Field(
        ...,
        description=(
            "Array of code blocks representing the fix. "
            "Each block is a complete function/class/module section. "
            "Blocks with has_changes=True contain the actual fixes."
        ),
        min_length=1,
    )
    helper_code: Optional[str] = Field("", description="Additional helper code")
    import_block_code: Optional[str] = Field("", description="Import block code")
    end_import_block_code: Optional[int] = Field(
        None, description="End import block code"
    )
    placement_helper: Optional[str] = Field(
        "", description="Additional helper code for placing the fix"
    )
    explanation: str = Field(..., description="Explanation of the fix")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    llm_model: str = Field(..., description="LLM model used for fixing")
    rule_description: Optional[str] = Field(
        None, description="SonarCloud rule description"
    )
    file_path: Optional[str] = Field(None, description="Path to the file being fixed")
    sonar_line_number: Optional[int] = Field(
        None, description="SonarCloud line number of the issue"
    )
    line_number: Optional[int] = Field(None, description="Line number of the issue")
    last_line_number: Optional[int] = Field(
        None, description="Last line number of the issue"
    )

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence fix."""
        return self.confidence >= 0.8


class FixResult(BaseModel):
    """Results from applying fixes to a project."""

    project_path: Path = Field(..., description="Path to the project")
    total_fixes_attempted: int = Field(..., description="Total fixes attempted")
    successful_fixes: List[FixSuggestion] = Field(
        default_factory=list, description="Successfully applied fixes"
    )
    failed_fixes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Failed fixes with error info"
    )
    skipped_issues: List[SonarIssue] = Field(
        default_factory=list, description="Issues that were skipped"
    )
    backup_created: bool = Field(
        default=False, description="Whether backup was created"
    )
    backup_path: Optional[Path] = Field(None, description="Path to backup")

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of fixes."""
        if self.total_fixes_attempted == 0:
            return 0.0
        return len(self.successful_fixes) / self.total_fixes_attempted


@dataclass
class ValidationResult:
    """Result of issue group validation."""

    is_valid: bool
    file_path: Optional[Path] = None
    file_path_tmp: Optional[Path] = None
    line_range: Optional[Dict[str, Any]] = None
    language: Optional[str] = None
    error: Optional[str] = None
