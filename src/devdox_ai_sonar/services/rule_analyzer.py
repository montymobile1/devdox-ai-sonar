from typing import List, Optional, Dict, Any, Union
from urllib.parse import urljoin
import time
import logging
import requests
import json
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from devdox_ai_sonar.models.sonar import (
    ProcessedRules,
)
logger = logging.getLogger(__name__)

class RuleAnalyzer:
    """Analyzes and processes SonarCloud rules"""

    def __init__(
            self,
            token: Optional[str] = None,
            organization: Optional[str] = None,
            base_url: str = "https://sonarcloud.io",
            timeout: int = 30,
            max_retries: int = 3,
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
        self.organization = str(organization)
        self.base_url = base_url
        self.timeout = timeout

        # Create session with connection pooling
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1,  # Wait 1s, 2s, 4s between retries
        )

        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Number of connection pools
            pool_maxsize=20,  # Connections per pool
        )

        self.session.mount("https://", adapter)

        # Set authentication header (secure - not in process list)
        self.session.headers.update(
            {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        )

    def fetch_all_rules(self, languages: Optional[List[str]] = None) -> ProcessedRules:
        """

        Fetch all SonarCloud rules with pagination.



        Args:

            languages: List of languages to filter by (e.g., ['java', 'python', 'javascript'])

                      If None, fetches rules for all languages



        Returns:

            Dictionary containing all rules with root causes and fix guidance

        """

        all_rules = []

        page = 1

        page_size = 500

        total_pages = None

        # Build API URL

        url = urljoin(self.base_url, "/api/rules/search")

        while True:
            params: Dict[str, Union[str, int]] = {
                "ps": page_size,  # Page size
                "p": page,  # Page number
                "organization": self.organization,
            }

            # Add language filter if specified

            if languages:
                params["languages"] = ",".join(languages)

            try:
                response = self.session.get(url, params=params, timeout=self.timeout)

                response.raise_for_status()

                data = response.json()

                rules = data.get("rules", [])

                if not rules:
                    break

                all_rules.extend(rules)

                # Calculate total pages on first request

                if total_pages is None:
                    total_count = data.get("total", 0)

                    total_pages = (total_count + page_size - 1) // page_size

                # Check if we've reached the end

                if page >= total_pages:
                    break

                page += 1

                time.sleep(0.1)  # Rate limiting

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching rules page {page}: {e}")

                break

        return self._process_rules(all_rules)

    def _infer_root_cause(self, rule: Dict) -> str:
        """
        Infer root cause based on rule information.

        Args:
        rule: Rule dictionary from SonarCloud API

        Returns:
        Root cause description
        """
        name = rule.get("name", "").lower()
        desc = rule.get("htmlDesc", "").lower()
        tags = [tag.lower() for tag in rule.get("tags", [])]
        rule_type = rule.get("type", "").lower()

        # Pattern matching for root causes
        return self._match_root_cause(name, desc, tags, rule_type)

    def _match_root_cause(
        self, name: str, desc: str, tags: List[str], rule_type: str
    ) -> str:
        if self._contains_keywords(name, desc, ["unused", "dead", "never used"]):
            return "Unused code creates clutter and indicates incomplete implementation or copy-paste errors"
        elif self._contains_keywords(name, desc, ["null", "npe", "null pointer"]):
            return "Null pointer access causes NullPointerException at runtime, crashing the application"
        elif self._contains_keywords(name, desc, ["sql", "injection", "query"]):
            return "Improper input handling allows malicious SQL injection attacks that can compromise data"
        elif self._contains_keywords(
            name, desc, ["password", "secret", "credential", "api key"]
        ):
            return "Hard-coded credentials in source code create security vulnerabilities and make rotation impossible"
        elif self._contains_keywords(
            name, desc, ["complex", "cognitive", "cyclomatic"]
        ):
            return "High code complexity makes code difficult to understand, test, debug, and maintain"
        elif self._contains_keywords(name, desc, ["duplicate", "repeated", "copy"]):
            return "Code duplication increases maintenance burden and creates consistency risks when changes are needed"
        elif self._contains_keywords(name, desc, ["empty", "blank"]):
            return "Empty code blocks indicate incomplete implementation or missing error handling"
        elif self._contains_keywords(name, desc, ["resource", "leak", "close"]):
            return "Unclosed resources (files, connections, streams) cause memory leaks and resource exhaustion"
        elif self._contains_keywords(
            name, desc, ["thread", "synchroniz", "concurrenc"]
        ):
            return "Improper thread handling can cause race conditions, deadlocks, and data corruption"
        elif self._contains_keywords(name, desc, ["exception", "error", "catch"]):
            return "Poor exception handling can hide errors, cause unexpected behavior, or crash the application"
        elif "security" in tags or "vulnerability" in rule_type:
            return "Security-sensitive code requires careful review and proper security controls to prevent attacks"
        elif rule_type == "bug":
            return "Coding error that can cause incorrect behavior, crashes, or unexpected results at runtime"
        elif rule_type == "code_smell":
            return "Code quality issue that affects readability, maintainability, or follows poor practices"
        else:
            return "Unknown rule type or insufficient data for analysis"

    def _contains_keywords(self, name: str, desc: str, keywords: List[str]) -> bool:
        return any(keyword in name or keyword in desc for keyword in keywords)


    def _process_rules(self, raw_rules: List[Dict]) -> ProcessedRules:
        """

        Process raw rules into structured format with root causes and fixes.



        Args:

            raw_rules: List of raw rule dictionaries from SonarCloud API



        Returns:

            Processed rules dictionary with metadata

        """

        processed_rules = {}

        # Group by language for statistics

        languages: Dict[str, List[str]] = {}

        for rule in raw_rules:
            rule_key = rule.get("key", "")

            # Extract rule information

            processed_rule = {
                "name": rule.get("name", ""),
                "language": rule.get("lang", "Generic"),
                "category": rule.get("type", "Unknown"),
                "severity": rule.get("severity", "INFO"),
                "status": rule.get("status", "READY"),
                "description": self._clean_html_description(rule.get("htmlDesc", "")),
                "tags": rule.get("tags", []),
                "system_tags": rule.get("sysTags", []),
                "created_at": rule.get("createdAt", ""),
                "parameters": rule.get("params", []),
                "root_cause": self._infer_root_cause(rule),
                "how_to_fix": self._generate_fix_guidance(rule),
            }

            processed_rules[rule_key] = processed_rule

            # Group by language for statistics

            lang = processed_rule["language"]

            if lang not in languages:
                languages[lang] = []

            languages[lang].append(rule_key)

        return {
            "rules": processed_rules,
            "metadata": {
                "total_rules": len(processed_rules),
                "languages": {lang: len(rules) for lang, rules in languages.items()},
                "categories": self._get_category_stats(processed_rules),
                "severities": self._get_severity_stats(processed_rules),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "organization": self.organization,
            },
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

        text = re.sub(r"<[^>]+>", " ", html_desc)

        # Clean up whitespace

        text = re.sub(r"\s+", " ", text).strip()

        # Truncate if too long

        max_length = 1000

        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text

    def _generate_fix_guidance(self, rule: Dict) -> Dict[str, Any]:
        """

        Generate comprehensive fix guidance based on rule type and patterns.



        Args:

            rule: Rule dictionary from SonarCloud API



        Returns:

            Fix guidance dictionary with description, steps, and examples

        """

        name = str(rule.get("name", "")).lower()

        rule_type = str(rule.get("type", "")).lower()

        # Pattern-based fix guidance

        if "unused" in name:
            return {
                "description": "Remove unused code or implement its intended functionality",
                "steps": [
                    "Identify all unused elements (variables, methods, imports, etc.)",
                    "Verify they are truly not needed by checking references",
                    "Remove unused elements or implement their intended purpose",
                    "Run tests to ensure no functionality is broken",
                ],
                "priority": "Medium",
                "effort": "Low",
            }

        elif "null" in name or "npe" in name:
            return {
                "description": "Add null checks before dereferencing objects",
                "steps": [
                    "Identify all potential null dereferences",
                    "Add null checks using if statements or Optional classes",
                    "Handle null cases appropriately (return, throw exception, use default)",
                    "Consider using null-safe operators where available",
                ],
                "priority": "High",
                "effort": "Medium",
            }

        elif "sql" in name or "injection" in name:
            return {
                "description": "Use parameterized queries instead of string concatenation",
                "steps": [
                    "Identify dynamic SQL query construction",
                    "Replace string concatenation with parameterized queries",
                    "Use prepared statements or ORM frameworks",
                    "Validate and sanitize all user inputs",
                ],
                "priority": "Critical",
                "effort": "Medium",
            }

        elif "password" in name or "secret" in name or "credential" in name:
            return {
                "description": "Move credentials to environment variables or secure storage",
                "steps": [
                    "Remove hard-coded credentials from source code",
                    "Store credentials in environment variables",
                    "Use secure credential management systems",
                    "Update deployment scripts to set environment variables",
                    "Remove credentials from version control history",
                ],
                "priority": "Critical",
                "effort": "Medium",
            }

        elif "complex" in name or "cognitive" in name:
            return {
                "description": "Refactor complex code into smaller, focused methods",
                "steps": [
                    "Identify the most complex parts of the method",
                    "Extract complex logic into separate methods with descriptive names",
                    "Use early returns to reduce nesting levels",
                    "Simplify conditional expressions using guard clauses",
                    "Consider using strategy pattern for complex conditional logic",
                ],
                "priority": "Medium",
                "effort": "High",
            }

        elif "duplicate" in name or "repeated" in name:
            return {
                "description": "Extract common code into reusable methods or constants",
                "steps": [
                    "Identify all instances of duplicated code",
                    "Extract common logic into a shared method or constant",
                    "Replace all duplicated instances with calls to the extracted code",
                    "Ensure the extracted code handles all use cases correctly",
                ],
                "priority": "Medium",
                "effort": "Medium",
            }

        elif "empty" in name:
            return {
                "description": "Implement proper logic or remove unnecessary empty blocks",
                "steps": [
                    "Determine the intended purpose of the empty block",
                    "Either implement the missing functionality",
                    "Or remove the empty block if not needed",
                    "Add TODO comments for future implementation if appropriate",
                ],
                "priority": "High",
                "effort": "Low",
            }

        elif rule_type == "vulnerability":
            return {
                "description": "Address security vulnerability following secure coding practices",
                "steps": [
                    "Review the security implications of the vulnerable code",
                    "Apply appropriate security controls and validation",
                    "Follow security best practices for the specific vulnerability type",
                    "Test security fixes thoroughly",
                    "Consider security code review",
                ],
                "priority": "Critical",
                "effort": "High",
            }

        elif rule_type == "bug":
            return {
                "description": "Fix logical error or potential runtime issue",
                "steps": [
                    "Understand the root cause of the bug",
                    "Implement the correct logic",
                    "Add comprehensive tests to prevent regression",
                    "Verify the fix doesn't introduce new issues",
                ],
                "priority": "High",
                "effort": "Medium",
            }

        else:
            return {
                "description": "Improve code quality following best practices",
                "steps": [
                    "Review the rule documentation for specific guidance",
                    "Apply the recommended changes",
                    "Verify improvements don't break functionality",
                    "Consider similar issues elsewhere in the codebase",
                ],
                "priority": "Low",
                "effort": "Low",
            }

    def _get_category_stats(self, rules: Dict) -> Dict[str, int]:
        """Get statistics by category."""

        stats: Dict[str, int] = {}

        for rule in rules.values():
            category = rule["category"]

            stats[category] = stats.get(category, 0) + 1

        return stats

    def _get_severity_stats(self, rules: Dict) -> Dict[str, int]:
        """Get statistics by severity."""

        stats: Dict[str, int] = {}

        for rule in rules.values():
            severity = rule["severity"]

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

        params = {"key": rule_key, "organization": self.organization}

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)

            response.raise_for_status()

            data = response.json()

            rule_data = data.get("rule", {})

            if not rule_data:
                return None

            # Process single rule

            processed = self._process_rules([rule_data])

            return processed["rules"].get(rule_key)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching rule {rule_key}: {e}")

            return None

    def get_rules_for_language(self, language: str) -> ProcessedRules:
        """

        Get all rules for a specific programming language.



        Args:

            language: Language code (e.g., 'java', 'python', 'javascript')



        Returns:

            Dictionary containing rules for the specified language

        """

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

        for rule_key, rule_data in all_rules["rules"].items():
            if rule_data["severity"] == severity:
                rule_data["key"] = rule_key

                filtered_rules.append(rule_data)

        return filtered_rules

    def export_rules_to_json(
            self, filename: str, languages: Optional[List[str]] = None
    ) -> None:
        """

        Export all rules to a JSON file.



        Args:

            filename: Output filename

            languages: Optional list of languages to filter by

        """

        rules_data = self.fetch_all_rules(languages=languages)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)

