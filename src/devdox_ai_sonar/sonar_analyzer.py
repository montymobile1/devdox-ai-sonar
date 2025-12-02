"""SonarCloud analyzer using direct REST API (production-ready)."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urljoin
import re
import time
import json
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

    def fetch_all_rules(self, languages: Optional[List[str]] = None) -> Dict[str, Any]:

        """

        Fetch all SonarCloud rules with pagination.



        Args:

            languages: List of languages to filter by (e.g., ['java', 'python', 'javascript'])

                      If None, fetches rules for all languages



        Returns:

            Dictionary containing all rules with root causes and fix guidance

        """

        logger.info("Starting to fetch all SonarCloud rules...")

        all_rules = []

        page = 1

        page_size = 500

        total_pages = None

        # Build API URL

        url = urljoin(self.base_url, "/api/rules/search")

        while True:

            logger.info(f"Fetching rules page {page}...")

            params = {

                'ps': page_size,  # Page size

                'p': page,  # Page number

                'organization': self.organization
            }

            # Add language filter if specified

            if languages:
                params['languages'] = ','.join(languages)

            try:
                response = self.session.get(

                    url,

                    params=params,

                    timeout=self.timeout

                )

                response.raise_for_status()

                data = response.json()

                rules = data.get('rules', [])

                if not rules:
                    break

                all_rules.extend(rules)

                # Calculate total pages on first request

                if total_pages is None:
                    total_count = data.get('total', 0)

                    total_pages = (total_count + page_size - 1) // page_size

                    logger.info(f"Total rules: {total_count}, Total pages: {total_pages}")

                # Check if we've reached the end

                if page >= total_pages:
                    break

                page += 1

                time.sleep(0.1)  # Rate limiting



            except requests.exceptions.RequestException as e:

                logger.error(f"Error fetching rules page {page}: {e}")

                break

        logger.info(f"Fetched {len(all_rules)} total rules")

        return self._process_rules(all_rules)

    def _process_rules(self, raw_rules: List[Dict]) -> Dict[str, Any]:

        """

        Process raw rules into structured format with root causes and fixes.



        Args:

            raw_rules: List of raw rule dictionaries from SonarCloud API



        Returns:

            Processed rules dictionary with metadata

        """

        processed_rules = {}

        # Group by language for statistics

        languages = {}

        for rule in raw_rules:

            rule_key = rule.get('key', '')

            # Extract rule information

            processed_rule = {

                'name': rule.get('name', ''),

                'language': rule.get('lang', 'Generic'),

                'category': rule.get('type', 'Unknown'),

                'severity': rule.get('severity', 'INFO'),

                'status': rule.get('status', 'READY'),

                'description': self._clean_html_description(rule.get('htmlDesc', '')),

                'tags': rule.get('tags', []),

                'system_tags': rule.get('sysTags', []),

                'created_at': rule.get('createdAt', ''),

                'parameters': rule.get('params', []),

                'root_cause': self._infer_root_cause(rule),

                'how_to_fix': self._generate_fix_guidance(rule)

            }

            processed_rules[rule_key] = processed_rule

            # Group by language for statistics

            lang = processed_rule['language']

            if lang not in languages:
                languages[lang] = []

            languages[lang].append(rule_key)

        return {

            'rules': processed_rules,

            'metadata': {

                'total_rules': len(processed_rules),

                'languages': {lang: len(rules) for lang, rules in languages.items()},

                'categories': self._get_category_stats(processed_rules),

                'severities': self._get_severity_stats(processed_rules),

                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),

                'organization': self.organization

            }

        }

    def _clean_html_description(self, html_desc: str) -> str:

        """

        Clean HTML description to extract meaningful text.



        Args:

            html_desc: HTML description from SonarCloud API



        Returns:

            Cleaned text description

        """

        if not html_desc:
            return ""

        # Remove HTML tags

        text = re.sub(r'<[^>]+>', ' ', html_desc)

        # Clean up whitespace

        text = re.sub(r'\s+', ' ', text).strip()

        # Truncate if too long

        max_length = 1000

        if len(text) > max_length:
            text = text[:max_length] + '...'

        return text

    def _infer_root_cause(self, rule: Dict) -> str:
        """
        Infer root cause based on rule information.

        Args:
        rule: Rule dictionary from SonarCloud API

        Returns:
        Root cause description
        """
        name = rule.get('name', '').lower()
        desc = rule.get('htmlDesc', '').lower()
        tags = [tag.lower() for tag in rule.get('tags', [])]
        rule_type = rule.get('type', '').lower()

        # Pattern matching for root causes
        return self._match_root_cause(name, desc, tags, rule_type)

    def _match_root_cause(self, name: str, desc: str, tags: List[str], rule_type: str) -> str:
        if self._contains_keywords(name, desc, ['unused', 'dead', 'never used']):
            return "Unused code creates clutter and indicates incomplete implementation or copy-paste errors"
        elif self._contains_keywords(name, desc, ['null', 'npe', 'null pointer']):
            return "Null pointer access causes NullPointerException at runtime, crashing the application"
        elif self._contains_keywords(name, desc, ['sql', 'injection', 'query']):
            return "Improper input handling allows malicious SQL injection attacks that can compromise data"
        elif self._contains_keywords(name, desc, ['password', 'secret', 'credential', 'api key']):
            return "Hard-coded credentials in source code create security vulnerabilities and make rotation impossible"
        elif self._contains_keywords(name, desc, ['complex', 'cognitive', 'cyclomatic']):
            return "High code complexity makes code difficult to understand, test, debug, and maintain"
        elif self._contains_keywords(name, desc, ['duplicate', 'repeated', 'copy']):
            return "Code duplication increases maintenance burden and creates consistency risks when changes are needed"
        elif self._contains_keywords(name, desc, ['empty', 'blank']):
            return "Empty code blocks indicate incomplete implementation or missing error handling"
        elif self._contains_keywords(name, desc, ['resource', 'leak', 'close']):
            return "Unclosed resources (files, connections, streams) cause memory leaks and resource exhaustion"
        elif self._contains_keywords(name, desc, ['thread', 'synchroniz', 'concurrenc']):
            return "Improper thread handling can cause race conditions, deadlocks, and data corruption"
        elif self._contains_keywords(name, desc, ['exception', 'error', 'catch']):
            return "Poor exception handling can hide errors, cause unexpected behavior, or crash the application"
        elif 'security' in tags or 'vulnerability' in rule_type:
            return "Security-sensitive code requires careful review and proper security controls to prevent attacks"
        elif rule_type == 'bug':
            return "Coding error that can cause incorrect behavior, crashes, or unexpected results at runtime"
        elif rule_type == 'code_smell':
            return "Code quality issue that affects readability, maintainability, or follows poor practices"
        else:
            return "Unknown rule type or insufficient data for analysis"

    def _contains_keywords(self, name: str, desc: str, keywords: List[str]) -> bool:
                return any(keyword in name or keyword in desc for keyword in keywords)

    def _generate_fix_guidance(self, rule: Dict) -> Dict[str, Any]:

        """

        Generate comprehensive fix guidance based on rule type and patterns.



        Args:

            rule: Rule dictionary from SonarCloud API



        Returns:

            Fix guidance dictionary with description, steps, and examples

        """

        name = rule.get('name', '').lower()

        rule_type = rule.get('type', '').lower()

        desc = rule.get('htmlDesc', '').lower()

        # Pattern-based fix guidance

        if 'unused' in name:

            return {

                'description': 'Remove unused code or implement its intended functionality',

                'steps': [

                    'Identify all unused elements (variables, methods, imports, etc.)',

                    'Verify they are truly not needed by checking references',

                    'Remove unused elements or implement their intended purpose',

                    'Run tests to ensure no functionality is broken'

                ],

                'priority': 'Medium',

                'effort': 'Low'

            }



        elif 'null' in name or 'npe' in name:

            return {

                'description': 'Add null checks before dereferencing objects',

                'steps': [

                    'Identify all potential null dereferences',

                    'Add null checks using if statements or Optional classes',

                    'Handle null cases appropriately (return, throw exception, use default)',

                    'Consider using null-safe operators where available'

                ],

                'priority': 'High',

                'effort': 'Medium'

            }



        elif 'sql' in name or 'injection' in name:

            return {

                'description': 'Use parameterized queries instead of string concatenation',

                'steps': [

                    'Identify dynamic SQL query construction',

                    'Replace string concatenation with parameterized queries',

                    'Use prepared statements or ORM frameworks',

                    'Validate and sanitize all user inputs'

                ],

                'priority': 'Critical',

                'effort': 'Medium'

            }



        elif 'password' in name or 'secret' in name or 'credential' in name:

            return {

                'description': 'Move credentials to environment variables or secure storage',

                'steps': [

                    'Remove hard-coded credentials from source code',

                    'Store credentials in environment variables',

                    'Use secure credential management systems',

                    'Update deployment scripts to set environment variables',

                    'Remove credentials from version control history'

                ],

                'priority': 'Critical',

                'effort': 'Medium'

            }



        elif 'complex' in name or 'cognitive' in name:

            return {

                'description': 'Refactor complex code into smaller, focused methods',

                'steps': [

                    'Identify the most complex parts of the method',

                    'Extract complex logic into separate methods with descriptive names',

                    'Use early returns to reduce nesting levels',

                    'Simplify conditional expressions using guard clauses',

                    'Consider using strategy pattern for complex conditional logic'

                ],

                'priority': 'Medium',

                'effort': 'High'

            }



        elif 'duplicate' in name or 'repeated' in name:

            return {

                'description': 'Extract common code into reusable methods or constants',

                'steps': [

                    'Identify all instances of duplicated code',

                    'Extract common logic into a shared method or constant',

                    'Replace all duplicated instances with calls to the extracted code',

                    'Ensure the extracted code handles all use cases correctly'

                ],

                'priority': 'Medium',

                'effort': 'Medium'

            }



        elif 'empty' in name:

            return {

                'description': 'Implement proper logic or remove unnecessary empty blocks',

                'steps': [

                    'Determine the intended purpose of the empty block',

                    'Either implement the missing functionality',

                    'Or remove the empty block if not needed',

                    'Add TODO comments for future implementation if appropriate'

                ],

                'priority': 'High',

                'effort': 'Low'

            }



        elif rule_type == 'vulnerability':

            return {

                'description': 'Address security vulnerability following secure coding practices',

                'steps': [

                    'Review the security implications of the vulnerable code',

                    'Apply appropriate security controls and validation',

                    'Follow security best practices for the specific vulnerability type',

                    'Test security fixes thoroughly',

                    'Consider security code review'

                ],

                'priority': 'Critical',

                'effort': 'High'

            }



        elif rule_type == 'bug':

            return {

                'description': 'Fix logical error or potential runtime issue',

                'steps': [

                    'Understand the root cause of the bug',

                    'Implement the correct logic',

                    'Add comprehensive tests to prevent regression',

                    'Verify the fix doesn\'t introduce new issues'

                ],

                'priority': 'High',

                'effort': 'Medium'

            }



        else:

            return {

                'description': 'Improve code quality following best practices',

                'steps': [

                    'Review the rule documentation for specific guidance',

                    'Apply the recommended changes',

                    'Verify improvements don\'t break functionality',

                    'Consider similar issues elsewhere in the codebase'

                ],

                'priority': 'Low',

                'effort': 'Low'

            }

    def _get_category_stats(self, rules: Dict) -> Dict[str, int]:

        """Get statistics by category."""

        stats = {}

        for rule in rules.values():
            category = rule['category']

            stats[category] = stats.get(category, 0) + 1

        return stats

    def _get_severity_stats(self, rules: Dict) -> Dict[str, int]:

        """Get statistics by severity."""

        stats = {}

        for rule in rules.values():
            severity = rule['severity']

            stats[severity] = stats.get(severity, 0) + 1

        return stats

    def get_rule_by_key(self, rule_key: str) -> Optional[Dict[str, Any]]:

        """

        Get detailed information for a specific rule.



        Args:

            rule_key: SonarCloud rule key (e.g., 'java:S1066')



        Returns:

            Rule information dictionary or None if not found

        """

        url = urljoin(self.base_url, "/api/rules/show")

        params = {

            'key': rule_key,

            'organization': self.organization

        }

        try:

            response = self.session.get(

                url,

                params=params,

                timeout=self.timeout

            )

            response.raise_for_status()

            data = response.json()

            rule_data = data.get('rule', {})

            if not rule_data:
                return None

            # Process single rule

            processed = self._process_rules([rule_data])

            return processed['rules'].get(rule_key)



        except requests.exceptions.RequestException as e:

            logger.error(f"Error fetching rule {rule_key}: {e}")

            return None

    def get_rules_for_language(self, language: str) -> Dict[str, Any]:

        """

        Get all rules for a specific programming language.



        Args:

            language: Language code (e.g., 'java', 'python', 'javascript')



        Returns:

            Dictionary containing rules for the specified language

        """

        logger.info(f"Fetching rules for language: {language}")

        return self.fetch_all_rules(languages=[language])

    def get_rules_by_severity(self, severity: str) -> List[Dict[str, Any]]:

        """

        Get all rules filtered by severity level.



        Args:

            severity: Severity level ('BLOCKER', 'CRITICAL', 'MAJOR', 'MINOR', 'INFO')



        Returns:

            List of rules matching the severity level

        """

        all_rules = self.fetch_all_rules()

        filtered_rules = []

        for rule_key, rule_data in all_rules['rules'].items():

            if rule_data['severity'] == severity:
                rule_data['key'] = rule_key

                filtered_rules.append(rule_data)

        return filtered_rules

    def export_rules_to_json(self, filename: str, languages: Optional[List[str]] = None):

        """

        Export all rules to a JSON file.



        Args:

            filename: Output filename

            languages: Optional list of languages to filter by

        """



        logger.info(f"Exporting rules to {filename}")

        rules_data = self.fetch_all_rules(languages=languages)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {rules_data['metadata']['total_rules']} rules to {filename}")

    def get_project_issues(
            self,
            project_key: str,
            branch: str = "",
            pull_request_number: Optional[int] = 0,
            statuses: List[str] = None,
            severities: List[str] = None,
            types: List[str] = None
        ) -> Optional[AnalysisResult]:
            logger.info(f"Fetching issues for project {project_key} on branch {branch} with pull request {pull_request_number}")

            if statuses is None:
                statuses = ["OPEN"]

            url = urljoin(self.base_url, "/api/issues/search")

            try:
                params = self._build_query_params(project_key, branch, pull_request_number, statuses, severities, types)
                issues = self._fetch_issues(url, params)
                parsed_issues = self._parse_issues(issues)
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
            except requests.RequestException as e:
                return self._handle_exceptions(e, project_key)
            except Exception as e:
                logger.error(f"Unexpected error fetching issues for {project_key}: {e}", exc_info=True)
                return None

    def _build_query_params(self, project_key, branch, pull_request_number, statuses, severities, types):
        params = {
            'componentKeys': project_key,
            'organization': self.organization,
            'issueStatuses': ','.join(statuses),
            'ps': 500  # Page size (max 500)
        }
        if branch != '':
            params['branch'] = branch
        else:
            if pull_request_number != 0:
                params['pullRequest'] = str(pull_request_number)
            else:
                params['branch'] = 'main'
        if severities:
            params['severities'] = ','.join(severities)
        if types:
            params['types'] = ','.join(types)
        return params


    def _fetch_issues(self, url, params):
        all_issues = []
        page = 1
        while True:
            params['p'] = page
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            issues = data.get('issues', [])
            all_issues.extend(issues)
            paging = data.get('paging', {})
            total = paging.get('total', 0)
            if len(all_issues) >= total:
                break
            page += 1
        return all_issues


    def _handle_exceptions(self, e, project_key):
            status_code = e.response.status_code if hasattr(e, 'response') else None
            logger.error(f"Error fetching issues for {project_key}: {e}", exc_info=True)
            if isinstance(e, requests.Timeout):
                logger.warning(f"Request timed out while fetching issues for {project_key}.")
            elif isinstance(e, requests.HTTPError):
                error_text = e.response.text
                if status_code == 401:
                    logger.error("Authentication failed. Check your SonarCloud token.")
                elif status_code == 403:
                    logger.error("Access forbidden. Check organization and project permissions.")
                elif status_code == 404:
                    logger.error(f"Project '{project_key}' not found in organization '{self.organization}'.")
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
            branch: str = "",
            pull_request: Optional[int] = 0,
            max_issues: Optional[int] = None,
            severities: Optional[List[str]] = None,
            types_list: Optional[List[str]] = None
    ) -> List[SonarIssue]:
        """
        Get issues that are potentially fixable by LLM.

        Args:
            project_key: SonarCloud project key
            branch: Branch to analyze
            max_issues: Maximum number of issues to return
            types_list: Optional list of issue types to filter by

        Returns:
            List of fixable SonarIssue objects
        """
        analysis = self.get_project_issues(project_key, branch,pull_request_number=pull_request,severities=severities, types=types_list)

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