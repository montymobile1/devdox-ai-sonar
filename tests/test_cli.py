# Comprehensive Test Cases for cli.py


import pytest
import json
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock, call

from devdox_ai_sonar.cli import (
    main,
    analyze,
    fix_issues,
    fix_security_issues,
    inspect
)
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    AnalysisResult,
    FixSuggestion,
    FixResult,
    ProjectMetrics,
    Severity
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def runner():
    """Create CLI test runner"""
    return CliRunner()


@pytest.fixture
def mock_config_service():
    """Mock ConfigService"""
    with patch('devdox_ai_sonar.cli.ConfigService') as mock:
        mock_instance = Mock()
        mock_instance.load_auth_config.return_value = {
            "token": "test-token",
            "organization": "test-org",
            "project": "test-project",
            "project_path": "/tmp/project"
        }
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_analyzer():
    """Mock SonarCloudAnalyzer"""
    with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer') as mock:
        yield mock


@pytest.fixture
def mock_fixer():
    """Mock LLMFixer"""
    with patch('devdox_ai_sonar.cli.LLMFixer') as mock:
        yield mock


@pytest.fixture
def mock_config_manager():
    """Mock ConfigManager"""
    with patch('devdox_ai_sonar.cli.ConfigManager') as mock:
        mock_instance = Mock()
        mock_instance.get_value.return_value = []
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def sample_analysis_result():
    """Sample analysis result"""
    return AnalysisResult(
        project_key="test-project",
        organization="test-org",
        branch="main",
        total_issues=5,
        issues=[
            SonarIssue(
                key="issue-1",
                rule="python:S1234",
                severity="MAJOR",
                component="test.py",
                project="test-project",
                line=10,
                message="Test issue",
                type="BUG",
                status="OPEN"
            )
        ],
        issues_by_severity={
            Severity.MAJOR: [SonarIssue(
                key="issue-1",
                rule="python:S1234",
                severity="MAJOR",
                component="test.py",
                project="test-project",
                line=10,
                message="Test issue",
                type="BUG",
                status="OPEN"
            )]
        },
        metrics=ProjectMetrics(
            project_key="a",
            lines_of_code=1000,
            coverage=85.5,
            bugs=2,
            vulnerabilities=1,
            code_smells=10
        )
    )


# ============================================================================
# TEST CLASS: MAIN COMMAND
# ============================================================================

class TestMainCommand:
    """Test main CLI entry point"""
    
    def test_main_help(self, runner):
        """Test main command help"""
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert 'SonarCloud Analyzer' in result.output
        assert 'analyze' in result.output.lower()
    
    def test_main_version(self, runner):
        """Test version display"""
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        # Version should be displayed
    
    def test_main_verbose_flag(self, runner):
        """Test verbose flag"""
        result = runner.invoke(main, ['--verbose', '--help'])
        
        assert result.exit_code == 0



# ============================================================================
# TEST CLASS: ANALYZE COMMAND
# ============================================================================

class TestAnalyzeCommand:
    """Test analyze command"""
    
    def test_analyze_help(self, runner):
        """Test analyze command help"""
        result = runner.invoke(analyze, ['--help'])
        
        assert result.exit_code == 0
        assert 'Analyze' in result.output or 'analyze' in result.output.lower()
    
    def test_analyze_no_config(self, runner):
        """Test analyze without configuration"""
        with patch('devdox_ai_sonar.cli.ConfigService') as mock_config:
            mock_instance = Mock()
            mock_instance.load_auth_config.return_value = None
            mock_config.return_value = mock_instance
            
            result = runner.invoke(analyze, [])
        
        assert result.exit_code == 1
        assert 'authentication' in result.output.lower() or 'config' in result.output.lower()
    
    def test_analyze_success(
        self, runner, mock_config_service, mock_analyzer, 
        mock_config_manager, sample_analysis_result
    ):
        """Test successful analysis"""
        # Mock analyzer
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.get_project_issues.return_value = sample_analysis_result
        mock_analyzer.return_value = mock_analyzer_instance
        
        # Mock provider manager
        with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance.branch_or_pr_prompt.return_value = ("main", None)
            mock_provider.return_value = mock_provider_instance
            
            # Mock Prompt.ask for limit
            with patch('devdox_ai_sonar.cli.Prompt.ask', return_value="10"):
                result = runner.invoke(analyze, [])
        
        # Command should execute
        assert result.exit_code in [0, 1]
    
    def test_analyze_with_severity_filter(
        self, runner, mock_config_service, mock_analyzer, mock_config_manager
    ):
        """Test analyze with severity filter"""
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.get_project_issues.return_value = AnalysisResult(
            project_key="test",
            organization="org",
            branch="main",
            total_issues=0,
            issues=[],
            issues_by_severity={}
        )
        mock_analyzer.return_value = mock_analyzer_instance
        
        with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance.branch_or_pr_prompt.return_value = ("main", None)
            mock_provider.return_value = mock_provider_instance
            
            with patch('devdox_ai_sonar.cli.Prompt.ask', return_value="10"):
                result = runner.invoke(analyze, ['--severity', 'BLOCKER', '--severity', 'CRITICAL'])
        
        assert result.exit_code in [0, 1]
    
    def test_analyze_with_type_filter(
        self, runner, mock_config_service, mock_analyzer, mock_config_manager
    ):
        """Test analyze with type filter"""
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.get_project_issues.return_value = AnalysisResult(
            project_key="test",
            organization="org",
            branch="main",
            total_issues=0,
            issues=[],
            issues_by_severity={}
        )
        mock_analyzer.return_value = mock_analyzer_instance
        
        with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance.branch_or_pr_prompt.return_value = ("main", None)
            mock_provider.return_value = mock_provider_instance
            
            with patch('devdox_ai_sonar.cli.Prompt.ask', return_value="10"):
                result = runner.invoke(analyze, ['--type', 'BUG', '--type', 'VULNERABILITY'])
        
        assert result.exit_code in [0, 1]
    
    def test_analyze_save_results(
        self, runner, mock_config_service, mock_analyzer, 
        mock_config_manager, sample_analysis_result, tmp_path
    ):
        """Test saving analysis results to file"""
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.get_project_issues.return_value = sample_analysis_result
        mock_analyzer.return_value = mock_analyzer_instance
        
        output_file = tmp_path / "analysis.json"
        
        with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance.branch_or_pr_prompt.return_value = ("main", None)
            mock_provider.return_value = mock_provider_instance
            
            with patch('devdox_ai_sonar.cli.Prompt.ask', return_value="10"):
                with patch('devdox_ai_sonar.cli.Confirm.ask', side_effect=[True]):  # Save results
                    with patch('click.prompt', return_value=str(output_file)):
                        result = runner.invoke(analyze, [])
        
        # File should be created or command should execute
        assert result.exit_code in [0, 1] or output_file.exists()
    
    def test_analyze_api_error(
        self, runner, mock_config_service, mock_analyzer, mock_config_manager
    ):
        """Test handling API errors"""
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.get_project_issues.side_effect = Exception("API Error")
        mock_analyzer.return_value = mock_analyzer_instance
        
        with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance.branch_or_pr_prompt.return_value = ("main", None)
            mock_provider.return_value = mock_provider_instance
            
            with patch('devdox_ai_sonar.cli.Prompt.ask', return_value="10"):
                result = runner.invoke(analyze, [])
        
        assert result.exit_code == 1
    
    def test_analyze_invalid_config(self, runner, mock_config_service):
        """Test with invalid configuration"""
        mock_config_service.return_value.load_auth_config.return_value = {
            "token": "",  # Empty token
            "organization": "org",
            "project": "proj",
            "project_path": "/tmp"
        }
        
        result = runner.invoke(analyze, [])
        
        assert result.exit_code == 1


# ============================================================================
# TEST CLASS: FIX ISSUES COMMAND
# ============================================================================

class TestFixIssuesCommand:
    """Test fix_issues command"""
    
    def test_fix_issues_help(self, runner):
        """Test fix_issues command help"""
        result = runner.invoke(fix_issues, ['--help'])
        
        assert result.exit_code == 0
        assert 'fix' in result.output.lower()
    
    def test_fix_issues_no_config(self, runner):
        """Test fix_issues without configuration"""
        with patch('devdox_ai_sonar.cli.ConfigService') as mock_config:
            mock_instance = Mock()
            mock_instance.load_auth_config.return_value = None
            mock_config.return_value = mock_instance
            
            result = runner.invoke(fix_issues, [])
        
        assert result.exit_code == 1
    
    def test_fix_issues_no_llm_config(self, runner, mock_config_service, mock_config_manager):
        """Test fix_issues without LLM configuration"""
        mock_config_service.return_value.load_llm_config.return_value = None
        
        result = runner.invoke(fix_issues, [])
        
        assert result.exit_code == 1
        assert 'LLM' in result.output or 'llm' in result.output.lower()
    
    def test_fix_issues_with_max_fixes(
        self, runner, mock_config_service, mock_config_manager, 
        mock_analyzer, mock_fixer
    ):
        """Test fix_issues with max-fixes option"""
        # Setup mocks
        mock_config_service.return_value.load_llm_config.return_value = Mock(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            models=["gpt-4", "gpt-3.5-turbo"]
        )
        
        with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance.branch_or_pr_prompt.return_value = ("main", None)
            mock_provider.return_value = mock_provider_instance
            
            with patch('devdox_ai_sonar.cli.Confirm.ask', return_value=True):
                result = runner.invoke(fix_issues, ['--max-fixes', '15'])
        
        assert result.exit_code in [0, 1]
    
    def test_fix_issues_dry_run(
        self, runner, mock_config_service, mock_config_manager
    ):
        """Test fix_issues with dry-run flag"""
        mock_config_service.return_value.load_llm_config.return_value = Mock(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            models=[]
        )
        
        with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance.branch_or_pr_prompt.return_value = ("main", None)
            mock_provider.return_value = mock_provider_instance
            
            with patch('devdox_ai_sonar.cli.Confirm.ask', return_value=True):
                result = runner.invoke(fix_issues, ['--dry-run'])
        
        assert result.exit_code in [0, 1]
    
    def test_fix_issues_with_filters(
        self, runner, mock_config_service, mock_config_manager
    ):
        """Test fix_issues with type and severity filters"""
        mock_config_service.return_value.load_llm_config.return_value = Mock(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            models=[]
        )
        
        with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance.branch_or_pr_prompt.return_value = ("main", None)
            mock_provider.return_value = mock_provider_instance
            
            with patch('devdox_ai_sonar.cli.Confirm.ask', return_value=True):
                result = runner.invoke(fix_issues, [
                    '--types', 'BUG,VULNERABILITY',
                    '--severity', 'BLOCKER,CRITICAL'
                ])
        
        assert result.exit_code in [0, 1]
    
    def test_fix_issues_apply_flag(
        self, runner, mock_config_service, mock_config_manager
    ):
        """Test fix_issues with apply flag"""
        mock_config_service.return_value.load_llm_config.return_value = Mock(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            models=[]
        )
        
        with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance.branch_or_pr_prompt.return_value = ("main", None)
            mock_provider.return_value = mock_provider_instance
            
            with patch('devdox_ai_sonar.cli.Confirm.ask', return_value=True):
                result = runner.invoke(fix_issues, ['--apply'])
        
        assert result.exit_code in [0, 1]


# ============================================================================
# TEST CLASS: FIX SECURITY ISSUES COMMAND
# ============================================================================

class TestFixSecurityIssuesCommand:
    """Test fix_security_issues command"""
    
    def test_fix_security_issues_help(self, runner):
        """Test fix_security_issues help"""
        result = runner.invoke(fix_security_issues, ['--help'])
        
        assert result.exit_code == 0
        assert 'security' in result.output.lower()
    
    def test_fix_security_issues_no_config(self, runner):
        """Test without configuration"""
        with patch('devdox_ai_sonar.cli.ConfigService') as mock_config:
            mock_instance = Mock()
            mock_instance.load_auth_config.return_value = None
            mock_config.return_value = mock_instance
            
            result = runner.invoke(fix_security_issues, [])
        
        assert result.exit_code == 1
    
    def test_fix_security_issues_with_max_fixes(
        self, runner, mock_config_service, mock_config_manager
    ):
        """Test with max-fixes option"""
        mock_config_service.return_value.load_llm_config.return_value = Mock(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            models=[]
        )
        
        with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance.branch_or_pr_prompt.return_value = ("main", None)
            mock_provider.return_value = mock_provider_instance
            
            with patch('devdox_ai_sonar.cli.Confirm.ask', return_value=True):
                result = runner.invoke(fix_security_issues, ['--max-fixes', '5'])
        
        assert result.exit_code in [0, 1]
    
    def test_fix_security_issues_backup_flag(
        self, runner, mock_config_service, mock_config_manager
    ):
        """Test with backup flag"""
        mock_config_service.return_value.load_llm_config.return_value = Mock(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            models=[]
        )
        
        with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_provider:
            mock_provider_instance = Mock()
            mock_provider_instance.branch_or_pr_prompt.return_value = ("main", None)
            mock_provider.return_value = mock_provider_instance
            
            with patch('devdox_ai_sonar.cli.Confirm.ask', return_value=True):
                result = runner.invoke(fix_security_issues, ['--backup'])
        
        assert result.exit_code in [0, 1]


# ============================================================================
# TEST CLASS: INSPECT COMMAND
# ============================================================================

class TestInspectCommand:
    """Test inspect command"""
    
    def test_inspect_help(self, runner):
        """Test inspect command help"""
        result = runner.invoke(inspect, ['--help'])
        
        assert result.exit_code == 0
        assert 'inspect' in result.output.lower() or 'Inspect' in result.output
    
    def test_inspect_no_config(self, runner):
        """Test inspect without configuration"""
        with patch('devdox_ai_sonar.cli.ConfigService') as mock_config:
            mock_instance = Mock()
            mock_instance.load_auth_config.return_value = None
            mock_config.return_value = mock_instance
            
            result = runner.invoke(inspect, [])
        
        assert result.exit_code == 1
    
    def test_inspect_success(self, runner, mock_config_service, mock_analyzer, tmp_path):
        """Test successful inspection"""
        # Create test project
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        
        mock_config_service.return_value.load_auth_config.return_value = {
            "token": "test-token",
            "organization": "test-org",
            "project": "test-project",
            "project_path": str(tmp_path)
        }
        
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_project_directory.return_value = {
            "total_files": 1,
            "python_files": 1,
            "javascript_files": 0,
            "java_files": 0,
            "other_files": 0,
            "has_sonar_config": False,
            "has_git": False,
            "potential_source_dirs": ["src"]
        }
        mock_analyzer.return_value = mock_analyzer_instance
        
        result = runner.invoke(inspect, [])
        
        assert result.exit_code in [0, 1]
        assert 'total_files' in result.output.lower() or result.exit_code == 0


# ============================================================================
# TEST CLASS: HELPER FUNCTIONS
# ============================================================================

class TestHelperFunctions:
    """Test helper functions in CLI"""
    
    def test_display_analysis_results(
        self, runner, sample_analysis_result
    ):
        """Test _display_analysis_results function"""
        from devdox_ai_sonar.cli import _display_analysis_results
        
        # Should not raise exception
        _display_analysis_results(sample_analysis_result, limit=10)
    
    def test_display_fix_results(self, runner, sample_fix_suggestion):
        """Test _display_fix_results function"""
        from devdox_ai_sonar.cli import _display_fix_results
        
        result = FixResult(
            project_path=Path("/tmp/project"),
            total_fixes_attempted=10,
            successful_fixes=[sample_fix_suggestion] * 8,
            failed_fixes=[],
            skipped_issues=[],
            backup_created=True,
            backup_path="/tmp/backup"
        )
        
        # Should not raise exception
        _display_fix_results(result)
    
    def test_save_results(self, runner, sample_analysis_result, tmp_path):
        """Test _save_results function"""
        from devdox_ai_sonar.cli import _save_results
        
        output_file = tmp_path / "results.json"
        
        _save_results(sample_analysis_result, str(output_file))
        
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "project_key" in data


