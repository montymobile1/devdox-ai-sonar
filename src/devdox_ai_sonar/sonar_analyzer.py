"""SonarCloud analyzer using direct REST API (production-ready)."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import SonarIssue, AnalysisResult, ProjectMetrics, Severity, IssueType, Impact
from .logging_config import setup_logging, get_logger

setup_logging(level='DEBUG',log_file='demo.log')
logger = get_logger(__name__)



class SonarCloudAnalyzer:
    """
    Production-ready SonarCloud analyzer using direct REST API.

    Features:
    - Connection pooling for better performance
    - Automatic retries with exponential backoff
    - Proper timeout handling
    - Secure token management (not exposed in process list)
    - Comprehensive error handling
    """

    def __init__(
        self,
        token: str,
        organization: str,
        base_url: str = "https://sonarcloud.io",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the SonarCloud analyzer.

        Args:
            token: SonarCloud authentication token
            organization: Organization key
            base_url: SonarCloud base URL (default: https://sonarcloud.io)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retries for failed requests (default: 3)
        """
        self.token = token
        self.organization = organization
        self.base_url = base_url
        self.timeout = timeout

        # Create session with connection pooling
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1  # Wait 1s, 2s, 4s between retries
        )

        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Number of connection pools
            pool_maxsize=20       # Connections per pool
        )

        self.session.mount("https://", adapter)

        # Set authentication header (secure - not in process list)
        self.session.headers.update({
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json'
        })


    def get_project_issues(
            self,
            project_key: str,
            branch: str = "main",
            statuses: List[str] = None,
            severities: List[str] = None,
            types: List[str] = None
    ) -> Optional[AnalysisResult]:
        """
        Fetch SonarCloud issues for a project using REST API.

        API Endpoint: GET /api/issues/search
        Documentation: https://sonarcloud.io/web_api/api/issues/search

        Args:
            project_key: SonarCloud project key (componentKeys)
            branch: Branch to analyze (default: main)
            statuses: Issue statuses to fetch (default: OPEN,ACCEPTED)
            severities: Issue severities to filter by
            types: Issue types to filter by

        Returns:
            AnalysisResult with issues and metadata, or None if error
        """
        if statuses is None:
            statuses = ["OPEN", "ACCEPTED"]

        # Build API URL
        url = urljoin(self.base_url, "/api/issues/search")

        # Build query parameters
        params = {
            'componentKeys': project_key,
            'organization': self.organization,
            'branch': branch,
            'issueStatuses': ','.join(statuses),  # Note: SonarCloud uses 'issueStatuses' not 'statuses'
            'ps': 500  # Page size (max 500)
        }

        # Add optional filters
        if severities:
            params['severities'] = ','.join(severities)
        if types:
            params['types'] = ','.join(types)

        try:


            # Fetch all issues with pagination
            all_issues = []
            page = 1

            while True:
                params['p'] = page

                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )

                # Raise exception for HTTP errors
                response.raise_for_status()

                data = response.json()

                # Extract issues from response
                issues = data.get('issues', [])
                all_issues.extend(issues)

                # Check if we've fetched all issues
                paging = data.get('paging', {})
                total = paging.get('total', 0)
                page_size = paging.get('pageSize', 500)


                if len(all_issues) >= total:
                    break

                page += 1



            # Convert raw data to SonarIssue objects
            parsed_issues = self._parse_issues(all_issues)

            # Get project metrics
            metrics = self.get_project_metrics(project_key)

            return AnalysisResult(
                project_key=project_key,
                organization=self.organization,
                branch=branch,
                total_issues=len(parsed_issues),
                issues=parsed_issues,
                metrics=metrics,
                analysis_timestamp=datetime.now().isoformat()
            )

        except requests.Timeout:
            logger.error(f"Request timeout while fetching issues for {project_key}", exc_info=True)
            return None

        except requests.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text

            logger.error(
                f"HTTP error {status_code} fetching issues for {project_key}: {error_text}",
                exc_info=True
            )

            # Provide helpful error messages
            if status_code == 401:
                logger.error("Authentication failed. Check your SonarCloud token.")
            elif status_code == 403:
                logger.error("Access forbidden. Check organization and project permissions.")
            elif status_code == 404:
                logger.error(f"Project '{project_key}' not found in organization '{self.organization}'.")

            return None

        except requests.RequestException as e:
            logger.error(f"Network error fetching issues for {project_key}: {e}", exc_info=True)
            return None

        except Exception as e:
            logger.error(f"Unexpected error fetching issues for {project_key}: {e}", exc_info=True)
            return None

    def get_project_metrics(self, project_key: str) -> Optional[ProjectMetrics]:
        """
        Fetch project metrics from SonarCloud.

        API Endpoint: GET /api/measures/component
        Documentation: https://sonarcloud.io/web_api/api/measures/component

        Args:
            project_key: SonarCloud project key

        Returns:
            ProjectMetrics object or None if error
        """
        # Build API URL
        url = urljoin(self.base_url, "/api/measures/component")

        # Metric keys to fetch
        metric_keys = [
            'ncloc',                        # Lines of code
            'coverage',                     # Test coverage
            'duplicated_lines_density',     # Duplication percentage
            'sqale_rating',                 # Maintainability rating
            'reliability_rating',           # Reliability rating
            'security_rating',              # Security rating
            'bugs',                         # Number of bugs
            'vulnerabilities',              # Number of vulnerabilities
            'code_smells',                  # Number of code smells
            'sqale_index',                  # Technical debt (minutes)
            'security_hotspots_reviewed'    # Security hotspots reviewed
        ]

        params = {
            'component': project_key,
            'metricKeys': ','.join(metric_keys)
        }

        try:

            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )

            response.raise_for_status()

            data = response.json()

            # Parse metrics data
            metrics_dict = {}
            for measure in data.get('component', {}).get('measures', []):
                metric_key = measure.get('metric')
                metric_value = measure.get('value')

                if metric_value is not None:
                    # Convert numeric values
                    if metric_key in ["ncloc", "bugs", "vulnerabilities", "code_smells"]:
                        try:
                            metrics_dict[metric_key] = int(metric_value)
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert {metric_key}={metric_value} to int")
                            metrics_dict[metric_key] = None
                    elif metric_key in ["coverage", "duplicated_lines_density", "security_hotspots_reviewed"]:
                        try:
                            metrics_dict[metric_key] = float(metric_value)
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert {metric_key}={metric_value} to float")
                            metrics_dict[metric_key] = None
                    else:
                        metrics_dict[metric_key] = metric_value



            return ProjectMetrics(
                project_key=project_key,
                lines_of_code=metrics_dict.get("ncloc"),
                coverage=metrics_dict.get("coverage"),
                duplicated_lines_density=metrics_dict.get("duplicated_lines_density"),
                maintainability_rating=metrics_dict.get("sqale_rating"),
                reliability_rating=metrics_dict.get("reliability_rating"),
                security_rating=metrics_dict.get("security_rating"),
                bugs=metrics_dict.get("bugs"),
                vulnerabilities=metrics_dict.get("vulnerabilities"),
                code_smells=metrics_dict.get("code_smells"),
                technical_debt=str(metrics_dict.get("sqale_index")) if metrics_dict.get("sqale_index") else None
            )

        except requests.Timeout:
            logger.error(f"Request timeout while fetching metrics for {project_key}", exc_info=True)
            return None

        except requests.HTTPError as e:
            logger.error(
                f"HTTP error {e.response.status_code} fetching metrics for {project_key}: {e.response.text}",
                exc_info=True
            )
            return None

        except requests.RequestException as e:
            logger.error(f"Network error fetching metrics for {project_key}: {e}", exc_info=True)
            return None

        except Exception as e:
            logger.error(f"Unexpected error fetching metrics for {project_key}: {e}", exc_info=True)
            return None

    def _parse_issues(self, issues_data: List[Dict[str, Any]]) -> List[SonarIssue]:
        """
        Parse raw issue data into SonarIssue objects.

        Args:
            issues_data: Raw issue data from SonarCloud API

        Returns:
            List of SonarIssue objects
        """
        issues = []

        for issue_data in issues_data:

            try:
                # Map severity enum
                severity_str = issue_data.get("severity", "").upper()
                severity = Severity(severity_str) if severity_str in Severity._value2member_map_ else Severity.INFO

                # Map issue type enum
                type_str = issue_data.get("type", "").upper()
                issue_type = IssueType(type_str) if type_str in IssueType._value2member_map_ else IssueType.CODE_SMELL

                # Map impact enum (from impacts object)
                impacts = issue_data.get("impacts", [])
                impact = None
                if impacts and len(impacts) > 0:
                    # Get first impact's severity
                    impact_severity = impacts[0].get("severity", "").upper()
                    if impact_severity in Impact._value2member_map_:
                        impact = Impact(impact_severity)

                # Extract file path from component
                component = issue_data.get("component", "")
                file_path = self._extract_file_path(component)

                first_line = issue_data.get("line")
                if first_line:
                    first_line = int(first_line)

                flows = issue_data.get("flows", [])
                last_line = first_line  # default

                for flow in flows:
                    for location in flow.get("locations", []):
                        text_range = location.get("textRange", {})
                        end_line = text_range.get("endLine")

                        if end_line:
                            end_line = int(end_line)
                            if end_line > last_line:
                                last_line = end_line

                issue = SonarIssue(
                    key=issue_data.get("key", ""),
                    rule=issue_data.get("rule", ""),
                    severity=severity.value,
                    component=component,
                    project=issue_data.get("project", ""),
                    first_line=first_line,
                    last_line=last_line,
                    message=issue_data.get("message", ""),
                    type=issue_type,
                    impact=impact,
                    file=file_path,
                    branch=issue_data.get("branch"),
                    status=issue_data.get("status", "OPEN"),
                    creation_date=issue_data.get("creationDate"),
                    update_date=issue_data.get("updateDate"),
                    tags=issue_data.get("tags", []),
                    effort=issue_data.get("effort"),
                    debt=issue_data.get("debt")
                )

                issues.append(issue)

            except Exception as e:
                logger.error(
                    f"Error parsing issue {issue_data.get('key', 'unknown')}: {e}",
                    exc_info=True
                )
                continue

        logger.info(f"Successfully parsed {len(issues)} issues")
        return issues

    def _extract_file_path(self, component: str) -> Optional[str]:
        """
        Extract file path from SonarCloud component string.

        Args:
            component: SonarCloud component identifier

        Returns:
            File path string or None
        """
        if not component:
            return None

        # Component format is usually "project_key:file_path"
        if ":" in component:
            return component.split(":", 1)[1]

        return component

    def get_fixable_issues(
            self,
            project_key: str,
            branch: str = "main",
            max_issues: Optional[int] = None
    ) -> List[SonarIssue]:
        """
        Get issues that are potentially fixable by LLM.

        Args:
            project_key: SonarCloud project key
            branch: Branch to analyze
            max_issues: Maximum number of issues to return

        Returns:
            List of fixable SonarIssue objects
        """
        analysis = self.get_project_issues(project_key, branch)
        if not analysis:
            return []

        fixable = analysis.fixable_issues

        # Sort by severity (most critical first)
        severity_order = {
            Severity.BLOCKER: 0,
            Severity.CRITICAL: 1,
            Severity.MAJOR: 2,
            Severity.MINOR: 3,
            Severity.INFO: 4
        }
        fixable.sort(key=lambda x: severity_order.get(x.severity, 999))

        if max_issues:
            fixable = fixable[:max_issues]

        return fixable

    def analyze_project_directory(self, project_path) -> Dict[str, Any]:
        """
        Analyze a local project directory to understand structure.

        Args:
            project_path: Path to the project directory

        Returns:
            Dictionary with project analysis information
        """
        from pathlib import Path

        project_path = Path(project_path)

        if not project_path.exists() or not project_path.is_dir():
            raise ValueError(f"Invalid project path: {project_path}")

        analysis = {
            "path": str(project_path),
            "total_files": 0,
            "python_files": 0,
            "javascript_files": 0,
            "java_files": 0,
            "other_files": 0,
            "directories": [],
            "has_sonar_config": False,
            "has_git": False,
            "potential_source_dirs": []
        }

        # Check for common configuration files
        config_files = ["sonar-project.properties", ".sonarcloud.properties"]
        analysis["has_sonar_config"] = any((project_path / cfg).exists() for cfg in config_files)
        analysis["has_git"] = (project_path / ".git").exists()

        # Analyze file structure
        for file_path in project_path.rglob("*"):
            if file_path.is_file():
                analysis["total_files"] += 1
                suffix = file_path.suffix.lower()

                if suffix == ".py":
                    analysis["python_files"] += 1
                elif suffix in [".js", ".jsx", ".ts", ".tsx"]:
                    analysis["javascript_files"] += 1
                elif suffix in [".java", ".kotlin", ".scala"]:
                    analysis["java_files"] += 1
                else:
                    analysis["other_files"] += 1

            elif file_path.is_dir() and not file_path.name.startswith("."):
                relative_path = file_path.relative_to(project_path)
                analysis["directories"].append(str(relative_path))

                # Identify potential source directories
                if file_path.name in ["src", "source", "app", "lib", "core"]:
                    analysis["potential_source_dirs"].append(str(relative_path))


        return analysis

    def close(self):
        """Close the session and release resources."""
        self.session.close()
        logger.debug("SonarCloud API session closed")

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()