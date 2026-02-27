
# ********************************************************************
# YOU NEED TO COMMENT THIS IF YOU WILL USE THIS MODULE IN REAL TESTS
# ********************************************************************
import pytest

skip_message = (
    "Live filesystem tests are disabled in CI/CD. "
    "These tests perform real I/O (temp directory creation, git-clone "
    "simulation, permission changes) and are intended for manual or "
    "local execution only. To run them, comment out the pytest.skip() "
    "call below."
)
pytest.skip(skip_message, allow_module_level=True)
# ********************************************************************
# ********************************************************************
# ********************************************************************
