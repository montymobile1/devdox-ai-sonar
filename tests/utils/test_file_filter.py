from devdox_ai_sonar.utils.supported_programming_languages import FileFilter


def _make_filter(**kwargs):
    """Helper to build a FileFilter from list-style kwargs (matching old API)."""
    return FileFilter(
        allowed_suffixes=set(kwargs.get("allowed_suffixes", [])),
        excluded_suffixes=set(kwargs.get("excluded_suffixes", [])),
        allowed_prefixes=set(kwargs.get("allowed_prefixes", [])),
        excluded_prefixes=set(kwargs.get("excluded_prefixes", [])),
    )


class TestFileFilterIsProcessable:
    """Test suite for FileFilter.is_processable (replaces is_file_processable)."""

    # --- Suffix tests ---

    def test_allows_matching_suffix(self):
        f = _make_filter(allowed_suffixes=[".py"])
        assert f.is_processable("src/foo/bar.py") is True

    def test_rejects_non_matching_suffix(self):
        f = _make_filter(allowed_suffixes=[".py"])
        assert f.is_processable("src/foo/bar.js") is False

    def test_rejects_excluded_suffix(self):
        f = _make_filter(excluded_suffixes=[".pyc"])
        assert f.is_processable("foo.pyc") is False

    def test_multiple_allowed_suffixes(self):
        f = _make_filter(allowed_suffixes=[".py", ".js"])
        assert f.is_processable("foo.js") is True

    def test_empty_allowed_suffixes_allows_all(self):
        f = _make_filter()
        assert f.is_processable("foo.rs") is True

    def test_empty_excluded_suffixes_excludes_none(self):
        f = _make_filter()
        assert f.is_processable("foo.pyc") is True

    # --- Prefix tests ---

    def test_rejects_excluded_prefix(self):
        f = _make_filter(excluded_prefixes=["test_"])
        assert f.is_processable("test_bar.py") is False

    def test_allows_file_without_excluded_prefix(self):
        f = _make_filter(excluded_prefixes=["test_"])
        assert f.is_processable("bar.py") is True

    def test_requires_allowed_prefix(self):
        f = _make_filter(allowed_prefixes=["api_"])
        assert f.is_processable("api_handler.py") is True

    def test_rejects_missing_allowed_prefix(self):
        f = _make_filter(allowed_prefixes=["api_"])
        assert f.is_processable("handler.py") is False

    def test_empty_allowed_prefixes_allows_all(self):
        f = _make_filter()
        assert f.is_processable("anything.py") is True

    def test_multiple_excluded_prefixes(self):
        f = _make_filter(excluded_prefixes=["test_", "conftest_"])
        assert f.is_processable("conftest_foo.py") is False

    # --- Combined & path tests ---

    def test_both_checks_combined(self):
        """Suffix passes (.py) but prefix fails (test_) -> rejected."""
        f = _make_filter(allowed_suffixes=[".py"], excluded_prefixes=["test_"])
        assert f.is_processable("src/foo/test_bar.py") is False

    def test_nested_path_checks_filename_only(self):
        """Directory named test_utils should NOT trigger the prefix filter."""
        f = _make_filter(allowed_suffixes=[".py"], excluded_prefixes=["test_"])
        assert f.is_processable("src/test_utils/helper.py") is True

    def test_absolute_path_works(self):
        f = _make_filter(allowed_suffixes=[".py"])
        assert f.is_processable("/home/user/src/foo/bar.py") is True

    def test_all_filters_pass(self):
        """File passes all four filters simultaneously."""
        f = _make_filter(
            allowed_suffixes=[".py"],
            excluded_suffixes=[".pyc"],
            allowed_prefixes=["api_"],
            excluded_prefixes=["test_"],
        )
        assert f.is_processable("src/api_handler.py") is True

    def test_excluded_suffix_overrides_allowed(self):
        """If a suffix is in both allowed and excluded, excluded wins (eval order)."""
        f = _make_filter(
            allowed_suffixes=[".pyc", ".py"],
            excluded_suffixes=[".pyc"],
        )
        assert f.is_processable("foo.pyc") is False
