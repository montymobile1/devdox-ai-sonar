# ********************************************************************
# YOU NEED TO COMMENT THIS IF YOU WILL USE THIS MODULE IN REAL TESTS
# ********************************************************************
import pytest

skip_message = (
    "Live / end-to-end tests are disabled in CI/CD. "
    "They perform actual non-mocked operations and are intended "
    "for manual testing only. Comment out the skip below to run them locally."
)
pytest.skip(skip_message, allow_module_level=True)
# ********************************************************************
# ********************************************************************
# ********************************************************************
