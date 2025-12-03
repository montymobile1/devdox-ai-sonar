"""Data models for SonarCloud analysis and fixes."""

from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


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


class SonarIssue(BaseModel):
    """Represents a SonarCloud issue."""
    model_config = ConfigDict(use_enum_values=True)

    key: str = Field(..., description="Unique issue key")
    rule: str = Field(..., description="Rule identifier")
    severity: Severity = Field(..., description="Issue severity")
    component: str = Field(..., description="Component path")
    project: str = Field(..., description="Project key")
    first_line: Optional[int] = Field(None, description="First Line number")
    last_line: Optional[int] = Field(None, description="Last Line number")
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
        return self.type in fixable_types and self.first_line is not None and self.last_line is not None

class SonarSecurityIssue(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    key: str = Field(..., description="Unique issue key")
    component: str = Field(..., description="Component path")
    rule: str = Field(..., description="Rule identifier")
    project: str = Field(..., description="Project key")
    security_category: str = Field(..., description="Security category")
    vulnerability_probability: str = Field(..., description="Vulnerability probability")
    status: str = Field(default="OPEN", description="Issue status")
    first_line: Optional[int] = Field(None, description="First Line number")
    last_line: Optional[int] = Field(None, description="Last Line number")
    message: str = Field(..., description="Issue description")
    file: Optional[str] = Field(None, description="File path")
    creation_date: Optional[str] = Field(None, description="Creation date")
    update_date: Optional[str] = Field(None, description="Last update date")


    @property
    def file_path(self) -> Optional[Path]:
        """Get the file path as a Path object."""
        if self.file:
            return Path(self.file)
        return None



class FixSuggestion(BaseModel):
    """Represents an LLM-generated fix suggestion."""

    issue_key: str = Field(..., description="Related issue key")
    original_code: str = Field(..., description="Original problematic code")
    fixed_code: str = Field(..., description="Suggested fix")
    helper_code: Optional[str] = Field("", description="Additional helper code")
    placement_helper: Optional[str] = Field("", description="Additional helper code for placing the fix")
    explanation: str = Field(..., description="Explanation of the fix")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    llm_model: str = Field(..., description="LLM model used for fixing")
    rule_description: Optional[str] = Field(None, description="SonarCloud rule description")
    file_path: Optional[str] = Field(None, description="Path to the file being fixed")
    sonar_line_number: Optional[int] = Field(None, description="SonarCloud line number of the issue")
    line_number: Optional[int] = Field(None, description="Line number of the issue")
    last_line_number: Optional[int] = Field(None, description="Last line number of the issue")

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence fix."""
        return self.confidence >= 0.8


class ProjectMetrics(BaseModel):
    """SonarCloud project metrics."""

    project_key: str = Field(..., description="Project key")
    lines_of_code: Optional[int] = Field(None, description="Total lines of code")
    coverage: Optional[float] = Field(None, description="Test coverage percentage")
    duplicated_lines_density: Optional[float] = Field(None, description="Code duplication percentage")
    maintainability_rating: Optional[str] = Field(None, description="Maintainability rating")
    reliability_rating: Optional[str] = Field(None, description="Reliability rating")
    security_rating: Optional[str] = Field(None, description="Security rating")
    bugs: Optional[int] = Field(None, description="Number of bugs")
    vulnerabilities: Optional[int] = Field(None, description="Number of vulnerabilities")
    code_smells: Optional[int] = Field(None, description="Number of code smells")
    technical_debt: Optional[str] = Field(None, description="Technical debt time")

class SecurityAnalysisResult(BaseModel):

    project_key: str = Field(..., description="Project key")
    organization: str = Field(..., description="Organization key")
    branch: str = Field(default="main", description="Branch analyzed")
    total_issues: int = Field(..., description="Total number of issues")
    issues: List[SonarSecurityIssue] = Field(..., description="List of issues")
    analysis_timestamp: Optional[str] = Field(None, description="Analysis timestamp")

class AnalysisResult(BaseModel):
    """Results from SonarCloud analysis."""

    project_key: str = Field(..., description="Project key")
    organization: str = Field(..., description="Organization key")
    branch: str = Field(default="main", description="Branch analyzed")
    total_issues: int = Field(..., description="Total number of issues")
    issues: List[SonarIssue] = Field(..., description="List of issues")
    metrics: Optional[ProjectMetrics] = Field(None, description="Project metrics")
    fixable_issues: List[SonarIssue] = Field(default_factory=list, description="Issues that can be fixed by LLM")
    analysis_timestamp: Optional[str] = Field(None, description="Analysis timestamp")

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to set fixable issues."""
        self.fixable_issues = [issue for issue in self.issues if issue.is_fixable]

    @property
    def issues_by_severity(self) -> Dict[Severity, List[SonarIssue]]:
        """Group issues by severity."""
        result: Dict[Severity, List[SonarIssue]] = {severity: [] for severity in Severity}
        for issue in self.issues:
            result[issue.severity].append(issue)
        return result

    @property
    def issues_by_type(self) -> Dict[IssueType, List[SonarIssue]]:
        """Group issues by type."""
        result: Dict[IssueType, List[SonarIssue]] = {issue_type: [] for issue_type in IssueType}
        for issue in self.issues:
            result[issue.type].append(issue)
        return result


class FixResult(BaseModel):
    """Results from applying fixes to a project."""

    project_path: Path = Field(..., description="Path to the project")
    total_fixes_attempted: int = Field(..., description="Total fixes attempted")
    successful_fixes: List[FixSuggestion] = Field(default_factory=list, description="Successfully applied fixes")
    failed_fixes: List[Dict[str, Any]] = Field(default_factory=list, description="Failed fixes with error info")
    skipped_issues: List[SonarIssue] = Field(default_factory=list, description="Issues that were skipped")
    backup_created: bool = Field(default=False, description="Whether backup was created")
    backup_path: Optional[Path] = Field(None, description="Path to backup")

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of fixes."""
        if self.total_fixes_attempted == 0:
            return 0.0
        return len(self.successful_fixes) / self.total_fixes_attempted