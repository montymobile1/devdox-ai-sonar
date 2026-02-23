from devdox_ai_sonar.utils.supported_programming_languages import is_file_processable


class TestIsFileProcessable:
    """Test suite for is_file_processable function."""

    # --- Suffix tests ---

    def test_allows_matching_suffix(self):
        assert is_file_processable("src/foo/bar.py", allowed_suffixes={".py"}) is True

    def test_rejects_non_matching_suffix(self):
        assert is_file_processable("src/foo/bar.js", allowed_suffixes={".py"}) is False

    def test_rejects_excluded_suffix(self):
        assert is_file_processable("foo.pyc", excluded_suffixes={".pyc"}) is False

    def test_multiple_allowed_suffixes(self):
        assert (
            is_file_processable("foo.js", allowed_suffixes={".py", ".js"}) is True
        )

    def test_empty_allowed_suffixes_allows_all(self):
        assert is_file_processable("foo.rs") is True

    def test_empty_excluded_suffixes_excludes_none(self):
        assert is_file_processable("foo.pyc") is True

    # --- Prefix tests ---

    def test_rejects_excluded_prefix(self):
        assert (
            is_file_processable("test_bar.py", excluded_prefixes={"test_"}) is False
        )

    def test_allows_file_without_excluded_prefix(self):
        assert is_file_processable("bar.py", excluded_prefixes={"test_"}) is True

    def test_requires_allowed_prefix(self):
        assert (
            is_file_processable("api_handler.py", allowed_prefixes={"api_"}) is True
        )

    def test_rejects_missing_allowed_prefix(self):
        assert (
            is_file_processable("handler.py", allowed_prefixes={"api_"}) is False
        )

    def test_empty_allowed_prefixes_allows_all(self):
        assert is_file_processable("anything.py") is True

    def test_multiple_excluded_prefixes(self):
        assert (
            is_file_processable(
                "conftest_foo.py", excluded_prefixes={"test_", "conftest_"}
            )
            is False
        )

    # --- Combined & path tests ---

    def test_both_checks_combined(self):
        """Suffix passes (.py) but prefix fails (test_) -> rejected."""
        assert (
            is_file_processable(
                "src/foo/test_bar.py",
                allowed_suffixes={".py"},
                excluded_prefixes={"test_"},
            )
            is False
        )

    def test_nested_path_checks_filename_only(self):
        """Directory named test_utils should NOT trigger the prefix filter."""
        assert (
            is_file_processable(
                "src/test_utils/helper.py",
                allowed_suffixes={".py"},
                excluded_prefixes={"test_"},
            )
            is True
        )

    def test_absolute_path_works(self):
        assert (
            is_file_processable(
                "/home/user/src/foo/bar.py", allowed_suffixes={".py"}
            )
            is True
        )

    def test_all_filters_pass(self):
        """File passes all four filters simultaneously."""
        assert (
            is_file_processable(
                "src/api_handler.py",
                allowed_suffixes={".py"},
                excluded_suffixes={".pyc"},
                allowed_prefixes={"api_"},
                excluded_prefixes={"test_"},
            )
            is True
        )

    def test_excluded_suffix_overrides_allowed(self):
        """If a suffix is in both allowed and excluded, excluded wins."""
        assert (
            is_file_processable(
                "foo.pyc",
                allowed_suffixes={".pyc", ".py"},
                excluded_suffixes={".pyc"},
            )
            is False
        )
