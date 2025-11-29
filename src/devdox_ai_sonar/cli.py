"""Command-line interface for SonarCloud Analyzer."""

import os
import sys
from pathlib import Path
from typing import Optional, List


import click
from rich.console import Console
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID

from . import __version__
from .sonar_analyzer import SonarCloudAnalyzer
from .llm_fixer import LLMFixer
from .models import Severity, AnalysisResult


console = Console()

BOLD_MAGENTA = "bold magenta"  # Define constant for repeated literal

@click.group()
@click.version_option(__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """
    SonarCloud Analyzer - Analyze and fix SonarCloud issues using LLM.

    This tool helps you:
    - Fetch issues from SonarCloud projects
    - Analyze code quality metrics
    - Generate AI-powered fix suggestions
    - Apply fixes to your codebase
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
@click.option("--token", "-t", required=True, help="SonarCloud authentication token")
@click.option("--organization", "--org", required=True, help="SonarCloud organization key")
@click.option("--project", "-p", required=True, help="SonarCloud project key")
@click.option("--branch", "-b", default="", help="Branch to analyze (default: main)")
@click.option("--pull-request", "-pr", type=int,default=0, help="Pull request number to analyze (optional)")
@click.option("--severity", multiple=True, type=click.Choice(["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]), help="Filter by severity")
@click.option("--type", "issue_types", multiple=True, type=click.Choice(["BUG", "VULNERABILITY", "CODE_SMELL", "SECURITY_HOTSPOT"]), help="Filter by issue type")
@click.option("--output", "-o", type=click.Path(), help="Output file for results (JSON format)")
@click.option("--limit", type=int, help="Limit number of issues to display")
@click.pass_context
def analyze(
    ctx: click.Context,
    token: str,
    organization: str,
    project: str,
    branch: str,
    pull_request: Optional[str],
    severity: List[str],
    issue_types: List[str],
    output: Optional[str],
    limit: Optional[int]
) -> None:
    """Analyze a SonarCloud project and display issues."""

    try:
        analyzer = SonarCloudAnalyzer(token, organization)

        with Progress() as progress:
            task = progress.add_task("Fetching issues from SonarCloud...", total=None)

            result = analyzer.get_project_issues(
                project_key=project,
                branch=branch,
                pull_request_number=int(pull_request) if pull_request else 0,
                severities=list(severity) if severity else None,
                types=list(issue_types) if issue_types else None
            )

            progress.remove_task(task)

        if not result:
            console.print("[red]Failed to fetch issues from SonarCloud[/red]")
            sys.exit(1)

        # Display results
        _display_analysis_results(result, limit)

        # Save to file if requested
        if output:
            _save_results(result, output)
            console.print(f"\n[green]Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

def select_fixes_interactively(fixes: List):
    click.echo("\nAvailable fixes:\n")


    table = Table(show_header=True, header_style=BOLD_MAGENTA)
    table.add_column("Number", width=10)
    table.add_column("Issue", width=20)
    table.add_column("Original", width=50)
    table.add_column("Fixed", width=100,overflow="crop")
    table.add_column("Confidence", width=15)

    for idx, fix in enumerate(fixes, start=1):
        confidence_str = f"{fix.confidence:.2f}"
        table.add_row(
            str(idx),
            fix.issue_key[-15:],  # Show last 20 chars of issue key
            fix.original_code[:47] + "..." if len(fix.original_code) > 50 else fix.original_code,
            fix.fixed_code[:97] + "..." if len(fix.fixed_code) > 100 else fix.fixed_code,
            confidence_str
        )

    console.print("\n")
    console.print(table)

    choice = click.prompt(
        "\nEnter fix numbers to apply (e.g., 1,3,5) or 'all' or 'none'",
        default="all"
    ).strip().lower()

    if choice == "all":
        return fixes

    if choice == "none" or choice == "":
        return []

    try:
        # Convert "1,3,5" → [1,3,5]
        selected_indices = [int(x.strip()) for x in choice.split(",")]
    except ValueError:
        click.echo("❌ Invalid input. Expected numbers separated by commas.")
        return []

    # Filter fixes based on user input
    selected = [
        fix for idx, fix in enumerate(fixes, start=1)
        if idx in selected_indices
    ]

    return selected

@main.command()
@click.option("--token", "-t", required=True, help="SonarCloud authentication token")
@click.option("--organization", "--org", required=True, help="SonarCloud organization key")
@click.option("--project", "-p", required=True, help="SonarCloud project key")
@click.option("--project-path", required=True, type=click.Path(exists=True, path_type=Path), help="Path to local project directory")
@click.option("--branch", "-b", default="", help="Branch to analyze (default: main)")
@click.option("--pull-request", "-pr", type=int,default=0, help="Pull request number to analyze (optional)")
@click.option("--provider", type=click.Choice(["openai", "gemini","togetherai"]), default="togetherai", help="LLM provider")
@click.option("--types",type=str,help="Comma-separated issue types (BUG, VULNERABILITY, CODE_SMELL, SECURITY_HOTSPOT)")
@click.option("--severity", type=str,help="Comma-separated severities (BLOCKER, CRITICAL, MAJOR, MINOR)")
@click.option("--model", help="LLM model name")
@click.option("--api-key", help="LLM API key (or set environment variable)")
@click.option("--max-fixes", type=int, default=10, help="Maximum number of fixes to generate (default: 10)")
@click.option("--apply", is_flag=True, help="Apply fixes to the codebase")
@click.option("--dry-run", is_flag=True, help="Show what would be changed without applying fixes")
@click.option("--backup/--no-backup", default=True, help="Create backup before applying fixes (default: true)")
@click.pass_context
def fix(
    ctx: click.Context,
    token: str,
    organization: str,
    project: str,
    project_path: Path,
    branch: str,
    pull_request: int,
    provider: str,
    types: Optional[str],
    severity:Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    max_fixes: int,
    apply: bool,
    dry_run: bool,
    backup: bool
) -> None:
    """Generate and optionally apply LLM-powered fixes for SonarCloud issues."""
    VALID_TYPES = {"BUG", "VULNERABILITY", "CODE_SMELL", "SECURITY_HOTSPOT"}
    VALID_SEVERETIES = {"BLOCKER", "CRITICAL", "MAJOR", "MINOR"}
    try:
        severity_list=None
        types_list=None
        if severity and severity!="":
            severity_list = [t.strip() for t in severity.split(",")]
            unknown = set(severity_list) - VALID_SEVERETIES
            if unknown:
                raise click.BadParameter(f"Invalid severities: {', '.join(unknown)}")

        if types and types!="":
            types_list = [t.strip() for t in types.split(",")]
            unknown = set(types_list) - VALID_TYPES
            if unknown:
                raise click.BadParameter(f"Invalid issue types: {', '.join(unknown)}")
        # Initialize analyzer
        analyzer = SonarCloudAnalyzer(token, organization)

        # Initialize LLM fixer
        fixer = LLMFixer(
            provider=provider,
            model=model,
            api_key=api_key
        )

        console.print(f"[blue]Analyzing project: {project}[/blue]")
        console.print(f"[blue]Local path: {project_path}[/blue]")

        # Get fixable issues
        with Progress() as progress:
            task = progress.add_task("Fetching fixable issues...", total=None)

            fixable_issues = analyzer.get_fixable_issues(
                project_key=project,
                branch=branch,
                pull_request=pull_request,
                max_issues=max_fixes,
                severities=severity_list if severity_list else None,
                types_list=types_list if types else None
            )

            progress.remove_task(task)

        if not fixable_issues:
            console.print("[yellow]No fixable issues found[/yellow]")
            return

        console.print(f"\n[green]Found {len(fixable_issues)} fixable issues[/green]")

        # Generate fixes
        fixes = []
        with Progress() as progress:
            task = progress.add_task("Generating fixes...", total=len(fixable_issues))

            for issue in fixable_issues:
                rule_info = analyzer.get_rule_by_key(issue.rule)
                fix = fixer.generate_fix(issue, project_path,rule_info)
                if fix:
                    fixes.append(fix)

                progress.advance(task)
        if not fixes:
            console.print("[yellow]No fixes could be generated[/yellow]")
            return



        # Apply fixes if requested
        if apply or dry_run:
            if not dry_run:
                selected_fixes = fixes if dry_run else select_fixes_interactively(fixes)

                if not selected_fixes:
                    click.echo("No fixes selected. Exiting.")
                    return

                if not click.confirm(f"Apply {len(selected_fixes)} fixes to the codebase?"):
                    return
            result = fixer.apply_fixes_with_validation(fixes=selected_fixes,issues=fixable_issues, project_path=project_path,

                                                       create_backup=backup and not dry_run,dry_run=dry_run,use_validator=True,
                                                       validator_provider=provider,validator_model=model,validator_api_key=api_key)
            _display_fix_results(result)


    except Exception as e:

        console.print(f"Error: {str(e)}", style="red", markup=False)
        if ctx.obj.get("verbose"):
            console.print_exception()
        sys.exit(1)


@main.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path))
def inspect(project_path: Path) -> None:
    """Inspect a local project directory structure."""

    try:
        analyzer = SonarCloudAnalyzer("dummy", "dummy")  # Token not needed for local analysis
        analysis = analyzer.analyze_project_directory(project_path)

        # Display project analysis
        console.print(Panel.fit(f"[bold]Project Analysis: {project_path}[/bold]"))

        table = Table(show_header=True, header_style=BOLD_MAGENTA)
        table.add_column("Property")
        table.add_column("Value")

        table.add_row("Total Files", str(analysis["total_files"]))
        table.add_row("Python Files", str(analysis["python_files"]))
        table.add_row("JavaScript Files", str(analysis["javascript_files"]))
        table.add_row("Java Files", str(analysis["java_files"]))
        table.add_row("Other Files", str(analysis["other_files"]))
        table.add_row("Has SonarCloud Config", "✓" if analysis["has_sonar_config"] else "✗")
        table.add_row("Has Git Repository", "✓" if analysis["has_git"] else "✗")

        console.print(table)

        if analysis["potential_source_dirs"]:
            console.print(f"\n[bold]Potential Source Directories:[/bold]")
            for src_dir in analysis["potential_source_dirs"]:
                console.print(f"  • {src_dir}")

    except Exception as e:
        console.print(f"[red]Error inspecting project: {e}[/red]")
        sys.exit(1)


def _display_analysis_results(result: AnalysisResult, limit: Optional[int]) -> None:
    """Display analysis results in a formatted table."""

    console.print(Panel.fit(f"[bold]SonarCloud Analysis Results[/bold]"))
    console.print(f"Project: {result.project_key}")
    console.print(f"Organization: {result.organization}")
    console.print(f"Branch: {result.branch}")
    console.print(f"Total Issues: {result.total_issues}")

    if result.metrics:
        console.print(f"\n[bold]Project Metrics:[/bold]")
        if result.metrics.lines_of_code:
            console.print(f"Lines of Code: {result.metrics.lines_of_code:,}")
        if result.metrics.coverage:
            console.print(f"Test Coverage: {result.metrics.coverage:.1f}%")
        if result.metrics.bugs:
            console.print(f"Bugs: {result.metrics.bugs}")
        if result.metrics.vulnerabilities:
            console.print(f"Vulnerabilities: {result.metrics.vulnerabilities}")
        if result.metrics.code_smells:
            console.print(f"Code Smells: {result.metrics.code_smells}")

    # Display issues by severity
    severity_counts = result.issues_by_severity
    console.print(f"\n[bold]Issues by Severity:[/bold]")
    for severity in Severity:
        color="green"
        count = len(severity_counts[severity])
        if count > 0:
            color = _get_severity_color(severity)
            console.print(f"  {severity.value}: {count}", style=color, markup=False)

    # Display issues table
    if result.issues:

        issues_to_show = result.issues[:limit] if limit else result.issues

        table = Table(show_header=True, header_style=BOLD_MAGENTA)
        table.add_column("Severity", width=10)
        table.add_column("Type", width=12)
        table.add_column("File", width=30)
        table.add_column("Line", width=6)
        table.add_column("Message", width=50)

        for issue in issues_to_show:
            severity_color = _get_severity_color(Severity(issue.severity))
            severity_text = Text(issue.severity, style=severity_color)
            table.add_row(
                severity_text,
                issue.type,
                issue.file or "N/A",
                str(issue.first_line) if issue.first_line else "N/A",
                issue.message[:47] + "..." if len(issue.message) > 50 else issue.message  # Still need to escape this
            )


        console.print("\n")
        console.print(table)

        if limit and len(result.issues) > limit:
            console.print(f"\n[dim]... and {len(result.issues) - limit} more issues[/dim]")


def _display_fix_results(result) -> None:
    """Display fix application results."""

    console.print(f"\n[bold]Fix Results:[/bold]")
    console.print(f"Fixes Attempted: {result.total_fixes_attempted}")
    console.print(f"Successful: [green]{len(result.successful_fixes)}[/green]")
    console.print(f"Failed: [red]{len(result.failed_fixes)}[/red]")
    console.print(f"[red] Failed: {(result.failed_fixes)}[/red]")
    console.print(f"Success Rate: {result.success_rate:.1%}")

    if result.backup_created:
        console.print(f"Backup Created: [blue]{result.backup_path}[/blue]")

    if result.failed_fixes:
        console.print(f"\n[bold red]Failed Fixes:[/bold red]")
        for failed in result.failed_fixes:
            console.print(f"  • {failed.get('error', 'Unknown error')}")


def _save_results(result: AnalysisResult, output_path: str) -> None:
    """Save analysis results to JSON file."""
    import json

    # Convert to serializable format
    data = result.model_dump()

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def _get_severity_color(severity: Severity) -> str:
    """Get Rich color for severity level."""
    color_map = {
        Severity.BLOCKER: "red",
        Severity.CRITICAL: "red",
        Severity.MAJOR: "yellow",
        Severity.MINOR: "blue",
        Severity.INFO: "green"
    }
    return color_map.get(severity, "white")


if __name__ == "__main__":
    main()