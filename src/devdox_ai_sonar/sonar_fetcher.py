"""
Script to fetch ALL SonarQube rules via API and create comprehensive JSON
"""

from devdox_ai_sonar.sonar_analyzer import SonarCloudAnalyzer
from devdox_ai_sonar.config import settings


def main() -> None:
    """Main function to fetch all SonarQube rules"""

    try:
        print("Initializing SonarCloud Analyzer...")
        print(settings.SONAR_TOKEN)
        print(settings.SONAR_ORGANIZATION)
        analyzer = SonarCloudAnalyzer(
            token=settings.SONAR_TOKEN, organization=settings.SONAR_ORGANIZATION
        )

        print("🚀 Enhanced SonarCloud Analyzer Examples")
        print("=" * 50)

        # Example 1: Fetch all rules
        print("\n1. Fetching ALL SonarCloud rules...")
        # all_rules = analyzer.fetch_all_rules()
        # print(f"   ✅ Fetched {all_rules['metadata']['total_rules']} rules")
        # print(f"   📊 Languages: {list(all_rules['metadata']['languages'].keys())}")

        # Example 2: Get rules for specific language
        # print("\n2. Fetching Python-specific rules...")
        # python_rules = analyzer.get_rules_for_language('python')
        # python_count = python_rules['metadata']['languages'].get('python', 0)
        # print(f"   ✅ Found {python_count} Python rules")
        #
        # # Example 3: Get rules by severity
        # print("\n3. Fetching critical severity rules...")
        # critical_rules = analyzer.get_rules_by_severity('CRITICAL')
        # print(f"   ✅ Found {len(critical_rules)} critical rules")
        #
        # # Example 4: Get detailed rule information
        # print("\n4. Getting detailed rule information...")
        rule_info = analyzer.get_rule_by_key("python:S1481")  # type: ignore[attr-defined]  # Unused variables
        if rule_info:
            print(f"   📋 Rule: {rule_info['name']}")
            print(f"   🎯 Root Cause: {rule_info['root_cause']}")
            print(f"   🔧 Fix Priority: {rule_info['how_to_fix']['priority']}")

        # Example 5: Export rules to JSON
        # print("\n5. Exporting rules to JSON file...")
        # analyzer.export_rules_to_json('all_sonarcloud_rules.json')
        # print("   ✅ Rules exported to all_sonarcloud_rules.json")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Run examples
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
