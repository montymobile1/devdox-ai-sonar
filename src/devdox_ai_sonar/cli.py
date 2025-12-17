"""Command-line interface for SonarCloud Analyzer."""

import os
import sys
from pathlib import Path
from typing import Optional, List, Any, Union, cast, Sequence, Dict, Tuple
from rich.prompt import Confirm, Prompt
import rich_click as click
from rich.console import Console
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress


from devdox_ai_sonar import __version__
from devdox_ai_sonar.sonar_analyzer import SonarCloudAnalyzer
from devdox_ai_sonar.services.rule_analyzer import RuleAnalyzer
from devdox_ai_sonar.llm_fixer import LLMFixer
from devdox_ai_sonar.models.llm_config import ConfigManager
from devdox_ai_sonar.utils.validator import InputValidator
from devdox_ai_sonar.utils.exceptions import (ValidationError)
from devdox_ai_sonar.services.configuration import ConfigService, AuthConfig, LLMConfig

from devdox_ai_sonar.models.sonar import (
    Severity,
    AnalysisResult,
    SonarSecurityIssue,
    FixSuggestion,
    SonarIssue,
    FixResult,
)
from devdox_ai_sonar.utils.provider_config import (
    ProviderConfigUI,
    ProviderValidator,
    ProviderConfigManager,
)

from devdox_ai_sonar.config import settings


console = Console()


BOLD_MAGENTA = "bold magenta"  # Define constant for repeated literal
AUTHENTICATION_NOT_FOUND = "[red]❌ No authentication configuration found[/red]"
DEVDOX_SONAR_CONFIG="[yellow]💡 Run 'devdox_sonar_config' to configure authentication[/yellow]"
NO_BRANCH_OR_PR_SPECIFIED = "[red]❌ No branch or pull request specified[/red]"
AUTH_CONFIG_LOADED = "[green]✓[/green] Authentication config loaded"

VALID_TYPES = {"BUG", "VULNERABILITY", "CODE_SMELL", "SECURITY_HOTSPOT"}
VALID_SEVERITIES = {"BLOCKER", "CRITICAL", "MAJOR", "MINOR"}


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
@click.option(
    "--severity",
    multiple=True,
    type=click.Choice(["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]),
    help="Filter by severity",
)
@click.option(
    "--type",
    "issue_types",
    multiple=True,
    type=click.Choice(["BUG", "VULNERABILITY", "CODE_SMELL", "SECURITY_HOTSPOT"]),
    help="Filter by issue type",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    severity: List[str],
    issue_types: List[str],
) -> None:
    """Analyze a SonarCloud project and display issues."""

    try:
        ui = ProviderConfigUI()
        config_service = ConfigService(sonar_path=Path(settings.auth_file_path))
        auth_config = config_service.load_auth_config()

        if not auth_config:
            console.print(AUTHENTICATION_NOT_FOUND)
            console.print(
                DEVDOX_SONAR_CONFIG
            )
            sys.exit(1)
        auth_config = AuthConfig(**auth_config)
        # Validate auth config
        is_valid, error_msg = auth_config.validate()
        if not is_valid:
            console.print(f"[red]❌ Configuration error: {error_msg}[/red]")
            sys.exit(1)

        console.print(AUTH_CONFIG_LOADED)
        console.print(f"  Project: [cyan]{auth_config.project}[/cyan]")
        console.print(f"  Organization: [cyan]{auth_config.organization}[/cyan]")

        manager = ConfigManager(config_path=settings.config_file_path)
        manager.load_config()
        validator = ProviderValidator()

        provider_manager = ProviderConfigManager(manager, ui, validator)
        branch, pull_request = provider_manager.branch_or_pr_prompt()
        if not branch and not pull_request:
            console.print(NO_BRANCH_OR_PR_SPECIFIED)
            sys.exit(1)

        limit = Prompt.ask(
            f"Max issues to fetch where max issues is {settings.MAX_FIXES_LIMIT}", default=settings.MAX_FIXES_LIMIT
        )
        try:
            limit = int(limit)
            if limit <= 0:
                raise ValueError
            if limit > settings.MAX_FIXES_LIMIT:
                limit = settings.MAX_FIXES_LIMIT

        except ValueError:
            console.print("\n[red]❌ You must enter an integer! [/red]")
            raise click.Abort()

        analyzer = SonarCloudAnalyzer(auth_config.token, auth_config.organization)

        with Progress() as progress:
            task = progress.add_task("Fetching issues from SonarCloud...", total=None)

            result = analyzer.get_project_issues(
                project_key=auth_config.project,
                branch=branch,
                max_issues=limit,
                pull_request_number=int(pull_request) if pull_request else 0,
                severities=list(severity) if severity else None,
                types=list(issue_types) if issue_types else None,
            )

            progress.remove_task(task)

        if not result:
            console.print("[red]Failed to fetch issues from SonarCloud[/red]")
            sys.exit(1)

        # Display results
        _display_analysis_results(result, limit)

        # Save results in a file
        if Confirm.ask("Do you want to save the results to a file?", default=False):
            output_file = click.prompt(
                "Enter output file path", default="analysis.json"
            )
            _save_results(result, output_file)
            console.print(f"\n[green]Results saved to {output_file}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)



@main.command(name="fix_issues")
@click.option(
    "--types",
    type=str,
    help="Comma-separated issue types (BUG, VULNERABILITY, CODE_SMELL, SECURITY_HOTSPOT)",
)
@click.option(
    "--severity",
    type=str,
    help="Comma-separated severities (BLOCKER, CRITICAL, MAJOR, MINOR)",
)
@click.option(
    "--max-fixes",
    type=click.IntRange(1, settings.MAX_FIXES_LIMIT),
    default=settings.DEFAULT_MAX_FIXES,
    help=f"Maximum number of fixes to generate (1-{settings.MAX_FIXES_LIMIT}, default: {settings.DEFAULT_MAX_FIXES})",
    show_default=True,
)
@click.option("--apply", is_flag=True, help="Apply fixes to the codebase")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be changed without applying fixes"
)
@click.pass_context
def fix_issues(ctx: click.Context, **options: Any) -> None:
    try:
        # Step 1: Load and validate configuration
        auth_config, llm_config, branch, pull_request = _load_and_validate_config()

        # Step 2: Get and validate user inputs
        fix_params = _configure_fix_parameters(options)

        # Step 3: Process issues and apply fixes
        _process_and_fix_issues(
            ctx,
            auth_config,
            llm_config,
            branch,
            pull_request,
            fix_params
        )

    except ValidationError as e:
        console.print(f"\n[red]❌ {e.message}[/red]")
        raise click.Abort()
    except Exception as e:
        console.print(f"Error: {str(e)}", style="red", markup=False)
        if ctx.obj.get("verbose"):
            console.print_exception()
        sys.exit(1)


# def fix_issues(ctx: click.Context, **options: Any) -> None:
#     """
#     Generate and optionally apply LLM-powered fixes for SonarCloud issues.
#
#     This command:
#     1. Loads authentication and LLM configuration
#     2. Validates user inputs
#     3. Fetches issues from SonarCloud
#     4. Generates fixes using LLM
#     5. Optionally applies fixes to your codebase
#
#     """
#
#     # ============================================================================
#     # STEP 1: Load Configuration
#     # ============================================================================
#
#     console.print("\n[bold cyan]Step 1: Loading Configuration[/bold cyan]")
#     ui = ProviderConfigUI()
#     validator = ProviderValidator()
#     # Load authentication config
#     config_service = ConfigService(sonar_path=Path(settings.auth_file_path))
#     auth_config = config_service.load_auth_config()
#
#     if not auth_config:
#         console.print(AUTHENTICATION_NOT_FOUND)
#         console.print(
#             DEVDOX_SONAR_CONFIG
#         )
#         sys.exit(1)
#
#     # Validate auth config
#     auth_config = AuthConfig(**auth_config)
#     # Validate auth config
#     is_valid, error_msg = auth_config.validate()
#
#     if not is_valid:
#         console.print(f"[red]❌ Configuration error: {error_msg}[/red]")
#         sys.exit(1)
#
#     console.print(AUTH_CONFIG_LOADED)
#     console.print(f"  Project: [cyan]{auth_config.project}[/cyan]")
#     console.print(f"  Organization: [cyan]{auth_config.organization}[/cyan]")
#
#     manager = ConfigManager(config_path=settings.config_file_path)
#     manager.load_config()
#
#     # Load LLM config
#     llm_config = config_service.load_llm_config(manager)
#     if not llm_config:
#         console.print("[red]❌ No LLM providers configured[/red]")
#         console.print("[yellow]💡 Run 'devdox_sonar init_config' to configure LLM[/yellow]")
#         sys.exit(1)
#
#     provider_manager = ProviderConfigManager(manager, ui, validator)
#     branch, pull_request = provider_manager.branch_or_pr_prompt()
#     if not branch and not pull_request:
#         console.print(NO_BRANCH_OR_PR_SPECIFIED)
#         sys.exit(1)
#
#     console.print(AUTH_CONFIG_LOADED)
#     # Interactive model selection if enabled
#     if not Confirm.ask(f"Do you want to use the default model '{llm_config.model}'?", default=True):
#         default_model = ui.select_provider_from_list(llm_config.models, llm_config.provider)
#         if not default_model:
#             console.print("[yellow]⚠ Model selection cancelled[/yellow]")
#             return None
#
#     api_key = llm_config.api_key
#
#     if not api_key:
#         console.print("[red]❌ No API key found for LLM provider[/red]")
#         sys.exit(1)
#     # ============================================================================
#     # STEP 2: Get and Validate User Inputs
#     # ============================================================================
#
#     console.print("\n[bold cyan]Step 2: Configuring Fix Parameters[/bold cyan]")
#
#     dry_run = options.get("dry_run", False)
#     # Get max fixes
#     max_fixes = options.get("max_fixes", settings.DEFAULT_MAX_FIXES)
#     if not options.get("max_fixes"):
#         # Ask user
#         max_fixes_input = Prompt.ask(
#             f"Maximum fixes to generate (1-{settings.MAX_FIXES_LIMIT})",
#             default=str(settings.DEFAULT_MAX_FIXES)
#         )
#         try:
#             max_fixes = InputValidator.validate_max_issues(
#                 max_fixes_input,
#                 settings.MAX_FIXES_LIMIT,
#                 field_name="max_fixes"
#             )
#         except ValidationError as e:
#             console.print(f"\n[red]❌ {e.message}[/red]")
#             raise click.Abort()
#
#     console.print(f"[green]✓[/green] Max fixes: [cyan]{max_fixes}[/cyan]")
#
#     # Parse and validate types
#     types_list: Optional[List[str]] = None
#     if options.get("types"):
#         try:
#             types_list = InputValidator.validate_issue_types(options["types"])
#             console.print(f"[green]✓[/green] Issue types: [cyan]{', '.join(types_list)}[/cyan]")
#         except ValidationError as e:
#             console.print(f"\n[red]❌ {e.message}[/red]")
#             raise click.Abort()
#
#     # Parse and validate severities
#     severities_list: Optional[List[str]] = None
#     if options.get("severity"):
#         try:
#             severities_list = InputValidator.validate_severities(options["severity"])
#             console.print(f"[green]✓[/green] Severities: [cyan]{', '.join(severities_list)}[/cyan]")
#         except ValidationError as e:
#             console.print(f"\n[red]❌ {e.message}[/red]")
#             raise click.Abort()
#
#     # ============================================================================
#     # STEP 3: Create Services and Config
#     # ============================================================================
#
#     console.print("\n[bold cyan]Step 3: Initializing Services[/bold cyan]")
#
#     # Create analyzer
#     analyzer = SonarCloudAnalyzer(
#         token=auth_config.token,
#         organization=auth_config.organization
#     )
#     ruler = RuleAnalyzer( token=auth_config.token,
#         organization=auth_config.organization)
#     console.print("[green]✓[/green] SonarCloud analyzer initialized")
#     try:
#         # Create fixer
#         fixer = LLMFixer(
#             provider=llm_config.provider,
#             model=llm_config.model,
#             api_key=llm_config.api_key
#         )
#
#
#         console.print(f"[blue]Analyzing project: {auth_config.project}[/blue]")
#         console.print(f"[blue]Local path: {auth_config.project_path}[/blue]")
#         fixable_issues = _fetch_fixable_issues(
#                     analyzer,
#                     auth_config.project,
#                     branch,
#                     pull_request,
#                     max_fixes,
#                     severities_list,
#                     types_list,
#                 )
#
#         if not fixable_issues:
#             console.print("[yellow]No fixable issues found[/yellow]")
#             return
#
#         console.print(f"\n[green]Found {len(fixable_issues)} fixable issues[/green]")
#         for file_path, issues in fixable_issues.items():
#                     fixes = _generate_fixes(fixer, ruler, issues, Path(str(auth_config.project_path)))
#
#                     if not fixes:
#                         console.print("[yellow]No fixes could be generated[/yellow]")
#                         return
#                     backup = Confirm.ask("Do you want to save a backup before applying fixes?", default=False)
#                     _apply_fixes_if_requested(
#                         True,
#                         bool(dry_run),
#                         fixes,
#                         cast(List[SonarIssue], issues),
#                         fixer,
#                         Path(str(auth_config.project_path)),
#                         backup,
#                     )
#
#     except Exception as e:
#             console.print(f"Error: {str(e)}", style="red", markup=False)
#             if ctx.obj.get("verbose"):
#                 console.print_exception()
#             sys.exit(1)



@main.command()
def inspect() -> None:
    """Inspect a local project directory structure."""

    try:
        config_service = ConfigService(sonar_path=Path(settings.auth_file_path))
        auth_config = config_service.load_auth_config()

        if not auth_config:
            console.print(AUTHENTICATION_NOT_FOUND)
            console.print(
                DEVDOX_SONAR_CONFIG
            )
            sys.exit(1)

        # Validate auth config
        auth_config = AuthConfig(**auth_config)
        # Validate auth config
        is_valid, error_msg = auth_config.validate()

        if not is_valid:
            console.print(f"[red]❌ Configuration error: {error_msg}[/red]")
            sys.exit(1)
        token = auth_config.token
        organization = auth_config.organization
        project_path = auth_config.project_path
        analyzer = SonarCloudAnalyzer(
            token, organization
        )  # Token not needed for local analysis
        analysis = analyzer.analyze_project_directory(str(project_path))

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
        table.add_row(
            "Has SonarCloud Config", "✓" if analysis["has_sonar_config"] else "✗"
        )
        table.add_row("Has Git Repository", "✓" if analysis["has_git"] else "✗")

        console.print(table)

        if analysis["potential_source_dirs"]:
            console.print("\n[bold]Potential Source Directories:[/bold]")
            for src_dir in analysis["potential_source_dirs"]:
                console.print(f"  • {src_dir}")

    except Exception as e:
        console.print(f"[red]Error inspecting project: {e}[/red]")
        sys.exit(1)



@main.command(name="fix_security_issues")
@click.option(
    "--max-fixes",
    type=int,
    default=settings.DEFAULT_MAX_FIXES,
    help=f"Maximum number of fixes to generate (default: {settings.DEFAULT_MAX_FIXES})",
)
@click.option("--apply", is_flag=True, help="Apply fixes to the codebase")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be changed without applying fixes"
)
@click.option(
    "--backup/--no-backup",
    default=True,
    help="Create backup before applying fixes (default: true)",
)
@click.pass_context
def fix_security_issues(ctx: click.Context, **options: Any) -> None:
    """
    Generate and optionally apply LLM-powered fixes for security issues.

    This is a specialized version of fix_issues that:
    - Filters for security-related issues only
    - Uses security-specific fetching logic
    - Maintains same configuration workflow
    """
    try:
        # Reuse configuration loading from fix_issues
        auth_config, llm_config, branch, pull_request = _load_and_validate_config()

        # Get fix parameters (respecting user input)
        fix_params = _configure_security_fix_parameters(options)

        # Process security issues specifically
        _process_security_issues(
            ctx,
            auth_config,
            llm_config,
            branch,
            pull_request,
            fix_params
        )

    except ValidationError as e:
        console.print(f"\n[red]❌ {e.message}[/red]")
        raise click.Abort()
    except Exception as e:
        console.print(f"Error: {str(e)}", style="red", markup=False)
        if ctx.obj.get("verbose"):
            console.print_exception()
        sys.exit(1)


def _load_and_validate_config() -> Tuple[AuthConfig, LLMConfig, Optional[str], Optional[str]]:
    """
    Load and validate authentication and LLM configuration.

    Returns:
        Tuple of (auth_config, llm_config, branch, pull_request)

    Raises:
        SystemExit: If configuration is invalid or missing
    """
    console.print("\n[bold cyan]Step 1: Loading Configuration[/bold cyan]")

    # Load authentication
    auth_config = _load_auth_config()

    # Load LLM config and select model
    llm_config = _load_and_configure_llm()

    # Get branch or PR
    branch, pull_request = _get_branch_or_pr()

    console.print(AUTH_CONFIG_LOADED)
    return auth_config, llm_config, branch, pull_request


def _load_auth_config() -> AuthConfig:
    """Load and validate authentication configuration."""
    config_service = ConfigService(sonar_path=Path(settings.auth_file_path))
    auth_config_dict = config_service.load_auth_config()

    if not auth_config_dict:
        console.print(AUTHENTICATION_NOT_FOUND)
        console.print(DEVDOX_SONAR_CONFIG)
        sys.exit(1)

    auth_config = AuthConfig(**auth_config_dict)
    is_valid, error_msg = auth_config.validate()

    if not is_valid:
        console.print(f"[red]❌ Configuration error: {error_msg}[/red]")
        sys.exit(1)

    console.print(AUTH_CONFIG_LOADED)
    console.print(f"  Project: [cyan]{auth_config.project}[/cyan]")
    console.print(f"  Organization: [cyan]{auth_config.organization}[/cyan]")

    return auth_config


def _configure_security_fix_parameters(options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Configure parameters specific to security issue fixing.

    Key differences from general fix_issues:
    - No type/severity filtering (all security issues)
    - Respects user's apply/dry-run/backup flags
    - Validates max_fixes within allowed range

    Returns:
        Dictionary containing:
        - max_fixes: int (validated range)
        - apply: bool (from user input)
        - dry_run: bool (from user input)
        - backup: bool (from user input)
    """
    console.print("\n[bold cyan]Configuring Security Fix Parameters[/bold cyan]")

    # Get and validate max_fixes
    max_fixes = options.get("max_fixes", settings.DEFAULT_MAX_FIXES)
    console.print(f"[green]✓[/green] Max fixes: [cyan]{max_fixes}[/cyan]")

    # Get user preferences for application
    apply = options.get("apply", False)
    dry_run = options.get("dry_run", False)
    backup = options.get("backup", True)

    # Validate conflicting flags
    if apply and dry_run:
        console.print("[yellow]⚠ Both --apply and --dry-run specified. Using dry-run mode.[/yellow]")
        apply = False

    console.print(f"[green]✓[/green] Apply fixes: [cyan]{apply}[/cyan]")
    console.print(f"[green]✓[/green] Dry run: [cyan]{dry_run}[/cyan]")
    console.print(f"[green]✓[/green] Backup: [cyan]{backup}[/cyan]")

    return {
        "max_fixes": max_fixes,
        "apply": apply,
        "dry_run": dry_run,
        "backup": backup
    }


def _process_security_issues(
        ctx: click.Context,
        auth_config: AuthConfig,
        llm_config: LLMConfig,
        branch: Optional[str],
        pull_request: Optional[str],
        fix_params: Dict[str, Any]
) -> None:
    """
    Initialize services and process security issues specifically.

    Args:
        ctx: Click context
        auth_config: Authentication configuration
        llm_config: LLM configuration
        branch: Branch name (if applicable)
        pull_request: Pull request ID (if applicable)
        fix_params: Fix parameters (max_fixes, apply, dry_run, backup)
    """
    console.print("\n[bold cyan]Initializing Security Analysis Services[/bold cyan]")

    # Initialize services
    services = _initialize_services(auth_config, llm_config)

    console.print("[green]✓[/green] Services initialized")
    console.print(f"[blue]Analyzing project: {auth_config.project}[/blue]")
    console.print(f"[blue]Local path: {auth_config.project_path}[/blue]")

    # Fetch security-specific issues
    fixable_issues = _fetch_fixable_security_issues(
        services["analyzer"],
        auth_config.project,
        branch,
        pull_request,
        fix_params["max_fixes"],
    )

    if not fixable_issues:
        console.print("[yellow]No fixable security issues found[/yellow]")
        return

    console.print(f"\n[green]Found {len(fixable_issues)} fixable security issues[/green]")

    # Generate and apply fixes
    _generate_and_apply_security_fixes(
        fixable_issues,
        services,
        auth_config.project_path,
        fix_params
    )


def _load_and_configure_llm() -> LLMConfig:
    """Load LLM configuration and optionally select model interactively."""
    manager = ConfigManager(config_path=settings.config_file_path)
    manager.load_config()

    config_service = ConfigService(sonar_path=Path(settings.auth_file_path))
    llm_config = config_service.load_llm_config(manager)

    if not llm_config:
        console.print("[red]❌ No LLM providers configured[/red]")
        console.print("[yellow]💡 Run 'devdox_sonar init_config' to configure LLM[/yellow]")
        sys.exit(1)

    # Interactive model selection if user wants to change default
    if not Confirm.ask(
            f"Do you want to use the default model '{llm_config.model}'?",
            default=True
    ):
        ui = ProviderConfigUI()
        default_model = ui.select_provider_from_list(
            llm_config.models,
            llm_config.provider
        )
        if not default_model:
            console.print("[yellow]⚠ Model selection cancelled[/yellow]")
            sys.exit(1)
        llm_config.model = default_model

    if not llm_config.api_key:
        console.print("[red]❌ No API key found for LLM provider[/red]")
        sys.exit(1)

    return llm_config


def _get_branch_or_pr() -> Tuple[Optional[str], Optional[str]]:
    """Prompt user for branch or pull request."""
    ui = ProviderConfigUI()
    validator = ProviderValidator()
    manager = ConfigManager(config_path=settings.config_file_path)
    manager.load_config()

    provider_manager = ProviderConfigManager(manager, ui, validator)
    branch, pull_request = provider_manager.branch_or_pr_prompt()

    if not branch and not pull_request:
        console.print(NO_BRANCH_OR_PR_SPECIFIED)
        sys.exit(1)

    return branch, pull_request


def _initialize_services(
        auth_config: AuthConfig,
        llm_config: LLMConfig
) -> Dict[str, Any]:
    """
    Initialize analyzer, ruler, and fixer services.

    Returns:
        Dictionary containing initialized services:
        - analyzer: SonarCloudAnalyzer
        - ruler: RuleAnalyzer
        - fixer: LLMFixer
    """
    analyzer = SonarCloudAnalyzer(
        token=auth_config.token,
        organization=auth_config.organization
    )

    ruler = RuleAnalyzer(
        token=auth_config.token,
        organization=auth_config.organization
    )

    fixer = LLMFixer(
        provider=llm_config.provider,
        model=llm_config.model,
        api_key=llm_config.api_key
    )

    return {
        "analyzer": analyzer,
        "ruler": ruler,
        "fixer": fixer
    }


def _generate_and_apply_security_fixes(
        fixable_issues: Dict[str, List[SonarIssue]],
        services: Dict[str, Any],
        project_path: Path,
        fix_params: Dict[str, Any]
) -> None:
    """
    Generate and apply fixes for security issues.

    Args:
        fixable_issues: Dictionary mapping file paths to issues
        services: Dictionary containing analyzer, ruler, fixer
        project_path: Path to project root
        fix_params: Fix parameters (apply, dry_run, backup)
    """
    for file_path, issues in fixable_issues.items():
        console.print(f"\n[blue]Processing {file_path}...[/blue]")

        fixes = _generate_fixes(
            services["fixer"],
            services["ruler"],
            issues,
            Path(str(project_path))
        )

        if not fixes:
            console.print("[yellow]No fixes could be generated for this file[/yellow]")
            continue

        _apply_fixes_if_requested(
            fix_params["apply"],
            fix_params["dry_run"],
            fixes,
            cast(List[SonarIssue], issues),
            services["fixer"],
            Path(str(project_path)),
            fix_params["backup"],
        )

def _process_and_fix_issues(
        ctx: click.Context,
        auth_config: AuthConfig,
        llm_config: LLMConfig,
        branch: Optional[str],
        pull_request: Optional[str],
        fix_params: Dict[str, Any]
) -> None:
    """
    Initialize services, fetch issues, generate and apply fixes.

    Args:
        ctx: Click context
        auth_config: Authentication configuration
        llm_config: LLM configuration
        branch: Branch name (if applicable)
        pull_request: Pull request ID (if applicable)
        fix_params: Fix parameters (max_fixes, types_list, severities_list, dry_run)
    """
    console.print("\n[bold cyan]Step 3: Initializing Services[/bold cyan]")

    # Initialize services
    analyzer = SonarCloudAnalyzer(
        token=auth_config.token,
        organization=auth_config.organization
    )
    ruler = RuleAnalyzer(
        token=auth_config.token,
        organization=auth_config.organization
    )
    fixer = LLMFixer(
        provider=llm_config.provider,
        model=llm_config.model,
        api_key=llm_config.api_key
    )

    console.print("[green]✓[/green] SonarCloud analyzer initialized")
    console.print(f"[blue]Analyzing project: {auth_config.project}[/blue]")
    console.print(f"[blue]Local path: {auth_config.project_path}[/blue]")

    # Fetch fixable issues
    fixable_issues = _fetch_fixable_issues(
        analyzer,
        auth_config.project,
        branch,
        pull_request,
        fix_params["max_fixes"],
        fix_params["severities_list"],
        fix_params["types_list"],
    )

    if not fixable_issues:
        console.print("[yellow]No fixable issues found[/yellow]")
        return

    console.print(f"\n[green]Found {len(fixable_issues)} fixable issues[/green]")

    # Process each file's issues
    _process_fixable_issues(
        fixable_issues,
        fixer,
        ruler,
        auth_config.project_path,
        fix_params["dry_run"]
    )


def _process_fixable_issues(
        fixable_issues: Dict[str, List[SonarIssue]],
        fixer: LLMFixer,
        ruler: RuleAnalyzer,
        project_path: Path,
        dry_run: bool
) -> None:
    """Process and apply fixes for each file's issues."""
    for file_path, issues in fixable_issues.items():
        fixes = _generate_fixes(fixer, ruler, issues, Path(str(project_path)))

        if not fixes:
            console.print("[yellow]No fixes could be generated[/yellow]")
            continue

        backup = Confirm.ask(
            "Do you want to save a backup before applying fixes?",
            default=False
        )

        _apply_fixes_if_requested(
            True,  # apply flag
            dry_run,
            fixes,
            cast(List[SonarIssue], issues),
            fixer,
            Path(str(project_path)),
            backup,
        )


def _apply_fixes_if_requested(
    apply: bool,
    dry_run: bool,
    fixes: List[FixSuggestion],
    issues: Sequence[Union[SonarIssue, SonarSecurityIssue]],
    fixer: LLMFixer,
    project_path: Path,
    backup: bool,
) -> None:
    if not (apply or dry_run):
        return
    if not dry_run:
        selected_fixes = fixes if dry_run else select_fixes_interactively(fixes,issues)
        if not selected_fixes:
            click.echo("No fixes selected. Exiting.")
            return
        if not click.confirm(f"Apply {len(selected_fixes)} fixes to the codebase?"):
            return
    else:
        selected_fixes = fixes
    result = fixer.apply_fixes_with_validation(
        fixes=selected_fixes,
        issues=issues,
        project_path=project_path,
        create_backup=backup and not dry_run,
        dry_run=dry_run,
        use_validator=True,
        validator_provider=fixer.provider,
        validator_model=fixer.model,
        validator_api_key=fixer.api_key,
    )
    _display_fix_results(result)


def _configure_fix_parameters(options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get and validate user inputs for fix parameters.

    Returns:
        Dictionary containing validated parameters:
        - max_fixes: int
        - types_list: Optional[List[str]]
        - severities_list: Optional[List[str]]
        - dry_run: bool
    """
    console.print("\n[bold cyan]Step 2: Configuring Fix Parameters[/bold cyan]")

    # Get max fixes
    max_fixes = _get_max_fixes(options)
    console.print(f"[green]✓[/green] Max fixes: [cyan]{max_fixes}[/cyan]")

    # Validate types
    types_list = _validate_issue_types(options.get("types"))
    if types_list:
        console.print(f"[green]✓[/green] Issue types: [cyan]{', '.join(types_list)}[/cyan]")

    # Validate severities
    severities_list = _validate_severities(options.get("severity"))
    if severities_list:
        console.print(f"[green]✓[/green] Severities: [cyan]{', '.join(severities_list)}[/cyan]")

    return {
        "max_fixes": max_fixes,
        "types_list": types_list,
        "severities_list": severities_list,
        "dry_run": options.get("dry_run", False)
    }


def _get_max_fixes(options: Dict[str, Any]) -> int:
    """Get and validate max_fixes parameter."""
    max_fixes = options.get("max_fixes")

    if not max_fixes:
        max_fixes_input = Prompt.ask(
            f"Maximum fixes to generate (1-{settings.MAX_FIXES_LIMIT})",
            default=str(settings.DEFAULT_MAX_FIXES)
        )
        max_fixes = InputValidator.validate_max_issues(
            max_fixes_input,
            settings.MAX_FIXES_LIMIT,
            field_name="max_fixes"
        )

    return max_fixes

def _validate_issue_types(types_str: Optional[str]) -> Optional[List[str]]:
    """Validate and parse issue types string."""
    if not types_str:
        return None
    return InputValidator.validate_issue_types(types_str)


def _validate_severities(severity_str: Optional[str]) -> Optional[List[str]]:
    """Validate and parse severities string."""
    if not severity_str:
        return None
    return InputValidator.validate_severities(severity_str)



def _display_analysis_results(result: AnalysisResult, limit: Optional[int]) -> None:
    """Display analysis results in a formatted table."""

    console.print(Panel.fit("[bold]SonarCloud Analysis Results[/bold]"))
    console.print(f"Project: {result.project_key}")
    console.print(f"Organization: {result.organization}")
    console.print(f"Branch: {result.branch}")
    console.print(f"Total Issues: {result.total_issues}")

    if result.metrics:
        console.print("\n[bold]Project Metrics:[/bold]")
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
    console.print("\n[bold]Issues by Severity:[/bold]")
    for severity in Severity:
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
                (
                    issue.message[:47] + "..."
                    if len(issue.message) > 50
                    else issue.message
                ),
            )

        console.print("\n")
        console.print(table)

        if limit and len(result.issues) > limit:
            console.print(
                f"\n[dim]... and {len(result.issues) - limit} more issues[/dim]"
            )


def _display_fix_results(result: FixResult) -> None:
    """Display fix application results."""

    console.print("\n[bold]Fix Results:[/bold]")
    console.print(f"Fixes Attempted: {result.total_fixes_attempted}")
    console.print(f"Successful: [green]{len(result.successful_fixes)}[/green]")
    console.print(f"Failed: [red]{len(result.failed_fixes)}[/red]")
    console.print(f"[red] Failed: {(result.failed_fixes)}[/red]")
    console.print(f"Success Rate: {result.success_rate:.1%}")

    if result.backup_created:
        console.print(f"Backup Created: [blue]{result.backup_path}[/blue]")

    if result.failed_fixes:
        console.print("\n[bold red]Failed Fixes:[/bold red]")
        for failed in result.failed_fixes:
            console.print(f"  • {failed.get('error', 'Unknown error')}")


def _save_results(result: AnalysisResult, output_path: str) -> None:
    """Save analysis results to JSON file."""
    import json

    # Convert to serializable format
    data = result.model_dump()

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _get_severity_color(severity: Severity) -> str:
    """Get Rich color for severity level."""
    color_map = {
        Severity.BLOCKER: "red",
        Severity.CRITICAL: "red",
        Severity.MAJOR: "yellow",
        Severity.MINOR: "blue",
        Severity.INFO: "green",
    }
    return color_map.get(severity, "white")



def _fetch_fixable_issues(
    analyzer: SonarCloudAnalyzer,
    project_key: str,
    branch: str = "",
    pull_request: int = 0,
    max_issues: int = 10,
    severity_list: Optional[List[str]] = None,
    types_list: Optional[List[str]] = None,
) -> Dict[str, Union[SonarIssue, SonarSecurityIssue]]:
    with Progress() as progress:
        task = progress.add_task("Fetching fixable issues...", total=None)
        issues_per_files = analyzer.get_fixable_issues_by_files(
            project_key=project_key,
            branch=branch,
            pull_request=pull_request,
            max_issues=max_issues,
            severities=severity_list if severity_list else None,
            types_list=types_list if types_list else None,
        )

        progress.remove_task(task)
    return issues_per_files


def _fetch_fixable_security_issues(
    analyzer: SonarCloudAnalyzer,
    project_key: str,
    branch: str = "",
    pull_request: int = 0,
    max_issues: int = 10,
) -> List[SonarSecurityIssue]:
    with Progress() as progress:
        task = progress.add_task("Fetching fixable Security issues...", total=None)
        issues = analyzer.get_fixable_security_issues(
            project_key=project_key,
            branch=branch,
            pull_request=pull_request,
            max_issues=max_issues,
        )
        progress.remove_task(task)
    return issues


def _generate_fixes(
    fixer: LLMFixer,
    ruler: RuleAnalyzer,
    issues: Sequence[Union[SonarIssue, SonarSecurityIssue]],
    project_path: Path,
) -> List[FixSuggestion]:
    fixes = []
    with Progress() as progress:
        rule_info_list: Dict[str, Dict[str, str]] = {}
        for issue in issues:
            rule_info = ruler.get_rule_by_key(issue.rule)
            rule_info_list[issue.rule] = rule_info
        fix = fixer.generate_fix_by_file(issues, project_path, rule_info_list)
        if fix:
            fixes.append(fix)
        task = progress.add_task("Generating fixes...", total=len(fixes))
        progress.advance(task)
    return fixes


def select_fixes_interactively(fixes: List[FixSuggestion],issues:Sequence[Union[SonarIssue, SonarSecurityIssue]]) -> List[FixSuggestion]:
    click.echo("\nAvailable fixes:\n")

    table = Table(show_header=True, header_style=BOLD_MAGENTA)
    table.add_column("Number", width=10)
    table.add_column("File and line", width=50)
    table.add_column(
        "Fixed",
        width=100,
        no_wrap=False,
        overflow="fold",
        max_width=None,
        min_width=None,
    )
    table.add_column("Confidence", width=15)

    for idx, fix in enumerate(fixes, start=1):
        confidence_str = f"{fix.confidence:.2f}"
        file_path = os.path.basename(fix.file_path) if fix.file_path else "N/A"
        line_info = (
            f"{file_path}:{fix.sonar_line_number}"
            if fix.sonar_line_number
            else file_path
        )

        messages = "\n -> ".join(
            f"{issue.message} (line {issue.first_line} - {issue.last_line})" if issue.first_line else issue.message
            for issue in issues
        )
        line_info = f"{line_info} \n → {messages}"
        table.add_row(
            str(idx),
            line_info,
            fix.fixed_code[:500],
            confidence_str,
        )

    console.print("\n")
    console.print(table)

    choice = (
        click.prompt(
            "\nEnter fix numbers to apply (e.g., 1,3,5) or 'all' or 'none'",
            default="all",
        )
        .strip()
        .lower()
    )

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
        fix for idx, fix in enumerate(fixes, start=1) if idx in selected_indices
    ]

    return selected




if __name__ == "__main__":
    main()
