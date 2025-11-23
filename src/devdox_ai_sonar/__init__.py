"""
DevDox AI Sonar - AI-powered Sonar fix issues
"""



__version__ = "0.0.1"
__author__ = "Hayat Bourgi"
__email__ = "hayat.bourgi@montyholding.com"

from .sonar_analyzer import SonarCloudAnalyzer
from .models import SonarIssue, AnalysisResult, FixSuggestion
from .fix_validator import FixValidator, ValidationStatus
from .llm_fixer import LLMFixer

__all__ = [
    "SonarCloudAnalyzer",
    "SonarIssue",
    "AnalysisResult",
    "FixSuggestion",
    "LLMFixer",
    "FixValidator",
    "ValidationStatus",
    "__version__",
]

