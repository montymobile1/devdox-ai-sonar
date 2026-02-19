"""Live filesystem tests for TmpCloneManager cleanup behaviour.

These tests verify that temporary directories created by TmpCloneManager
are **actually deleted from disk** under every exit scenario identified
in the project design document (S-1 through S-13).

Unlike the unit tests in ``tests/utils/test_file_indentation.py`` which
use mocks to verify call sequences, every test in this module performs
real filesystem operations — creating directories, populating them with
files (including read-only git objects), and asserting that the path no
longer exists after the context manager exits.

Scenario table (from project design doc)
=========================================

======  ===================================  ========================================
ID      Scenario                             How / Guard
======  ===================================  ========================================
S-1     Branch not determined                tmp dir never created
S-2     Download fails -> click.Abort()      __aexit__ fires
S-3     Exception in _fetch_issues           __aexit__ fires
S-4     Exception in _process_files          __aexit__ fires
S-5     KeyboardInterrupt (Ctrl+C)           __aexit__ fires
S-6     SIGTERM -> SystemExit                __aexit__ fires
S-7     SIGKILL                              startup sweep cleans orphans
S-8     asyncio.CancelledError               __aexit__ fires (BaseException catch)
S-9     SystemExit (explicit exit())         __aexit__ fires
S-10    OOM kill                             startup sweep cleans orphans
S-11    Disk full during clone               __aexit__ cleans partial clone
S-12    Network failure during clone         __aexit__ cleans partial clone
S-13    Concurrent invocation                each manager cleans its own dir
======  ===================================  ========================================
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import click
import pytest

from devdox_ai_sonar.utils.file_indentation import (
    TmpCloneManager,
    generate_tmp_path,
    sweep_orphaned_tmp_dirs,
)


# ============================================================================
# Helpers — shared directory builders used across tests
# ============================================================================


def _populate(path: Path) -> None:
    """Create a small but realistic source-project tree inside *path*.

    Structure::

        path/
        ├── src/
        │   ├── main.py
        │   └── utils.py
        └── README.md
    """
    (path / "src").mkdir(exist_ok=True)
    (path / "src" / "main.py").write_text("print('hello')\n")
    (path / "src" / "utils.py").write_text("x = 1\n")
    (path / "README.md").write_text("# readme\n")


def _populate_git_clone(path: Path) -> None:
    """Simulate a git-clone directory with read-only object files.

    Git stores pack files with mode 0o444 (read-only).  On Windows and
    occasionally macOS, ``shutil.rmtree`` fails on these unless a custom
    ``onerror`` handler calls ``os.chmod`` first.  This helper creates
    that exact file layout so we can verify the cleanup handles it.

    Structure::

        path/
        ├── .git/
        │   ├── objects/
        │   │   └── pack/
        │   │       └── pack-abc123.idx  (mode 0o444)
        │   └── refs/
        │       └── heads/
        │           └── main             (mode 0o444)
        └── src/
            └── app.py
    """
    objects = path / ".git" / "objects" / "pack"
    objects.mkdir(parents=True)
    idx = objects / "pack-abc123.idx"
    idx.write_text("binary data")
    idx.chmod(0o444)

    refs = path / ".git" / "refs" / "heads"
    refs.mkdir(parents=True)
    (refs / "main").write_text("abc123\n")
    (refs / "main").chmod(0o444)

    (path / "src" / "app.py").parent.mkdir(parents=True, exist_ok=True)
    (path / "src" / "app.py").write_text("import os\n")


# ============================================================================
# S-0  Happy path — normal exit, no exception
# ============================================================================


class TestS00HappyPath:
    """Verify cleanup on the normal, exception-free exit path.

    This is the baseline: the context manager is entered, work is done
    inside the ``async with`` block, and the block exits normally.  The
    temp directory must be removed.
    """

    async def test_normal_exit_with_populated_dir(self):
        """S-0: Normal exit after populating the temp directory.

        Workflow:
            1. ``TmpCloneManager()`` creates a real temp directory via
               ``generate_tmp_path()``.
            2. The test populates it with a source tree (src/, README.md).
            3. The ``async with`` block exits normally.
            4. ``__aexit__`` calls ``remove_tmp_files()`` in a thread
               executor.
            5. Assertion: the temp directory no longer exists on disk.

        This is the most common exit path in production — the CLI
        successfully clones, processes issues, and finishes.
        """
        saved = None
        async with TmpCloneManager() as tmp:
            saved = tmp
            _populate(tmp)
            assert saved.exists()
        assert not saved.exists()

    async def test_normal_exit_with_git_clone_content(self):
        """S-0 sub: Normal exit with read-only git-clone files.

        Same as the baseline, but the directory contains read-only
        ``.git/objects/`` files that would cause ``shutil.rmtree`` to
        fail without the ``_handle_remove_readonly`` onerror handler.

        Verifies that the cross-platform read-only fix works on the
        happy path, not just on exception paths.
        """
        saved = None
        async with TmpCloneManager() as tmp:
            saved = tmp
            _populate_git_clone(tmp)
        assert not saved.exists()

    async def test_normal_exit_empty_dir(self):
        """S-0 sub: Normal exit with an empty temp directory.

        Edge case: the context manager is entered but nothing is
        written to the temp directory.  Cleanup must still succeed
        (``shutil.rmtree`` on an empty directory is valid).
        """
        saved = None
        async with TmpCloneManager() as tmp:
            saved = tmp
            assert saved.exists()
        assert not saved.exists()

    async def test_normal_exit_with_on_cleanup_callback(self):
        """S-0 sub: Normal exit with a successful on_cleanup callback.

        Verifies that when the optional ``on_cleanup`` callback is
        provided and runs successfully, the cleanup still proceeds and
        the temp directory is deleted.
        """
        callback_called_with = []

        def callback(p: str) -> None:
            callback_called_with.append(p)

        saved = None
        async with TmpCloneManager(on_cleanup=callback) as tmp:
            saved = tmp
            _populate(tmp)

        assert not saved.exists()
        assert len(callback_called_with) == 1
        assert callback_called_with[0] == str(saved)


# ============================================================================
# S-1  Branch not determined — tmp dir never created
# ============================================================================


class TestS01BranchNotDetermined:
    """Verify that no temp directory leaks when the context manager is
    never entered.

    In production, the CLI determines the branch before entering the
    ``async with TmpCloneManager()`` block.  If the user cancels branch
    selection or the branch is unavailable, the context manager is never
    entered and ``__aenter__`` is never called.
    """

    async def test_no_dir_created_before_aenter(self):
        """S-1: TmpCloneManager instantiated but never entered.

        Workflow:
            1. ``TmpCloneManager()`` is constructed (no I/O happens).
            2. ``__aenter__`` is never called.
            3. ``_tmp_path`` remains ``None``.
            4. No temp directory exists to leak.

        This verifies that the constructor is side-effect-free and that
        ``_tmp_path`` is only set inside ``__aenter__``.
        """
        manager = TmpCloneManager()
        assert manager._tmp_path is None

    async def test_aexit_with_no_aenter_is_safe(self):
        """S-1 sub: Calling __aexit__ directly without __aenter__.

        Edge case: if through some bug or refactoring ``__aexit__`` is
        called without a preceding ``__aenter__``, it must not crash.
        The guard ``if not self._tmp_path: return False`` handles this.
        """
        manager = TmpCloneManager()
        result = await manager.__aexit__(None, None, None)
        assert result is False


# ============================================================================
# S-2  Download fails -> click.Abort()
# ============================================================================


class TestS02DownloadFailsClickAbort:
    """Verify cleanup when the git clone download fails and ``click.Abort()``
    is raised.

    In production, ``cli.py`` raises ``click.Abort()`` when the user
    selects an empty branch or when ``download_latest_version()`` returns
    ``None``.  ``click.Abort`` is a ``BaseException`` (not ``Exception``),
    so the old ``except Exception`` handler would miss it.  After the bug
    fix, ``except BaseException`` catches it and the synchronous fallback
    ensures cleanup.
    """

    async def test_click_abort_with_git_clone_deletes(self):
        """S-2: click.Abort after partial git clone cleans up.

        Workflow:
            1. Context manager creates a temp directory.
            2. A simulated git clone populates it with read-only objects.
            3. ``click.Abort()`` is raised (simulating failed download).
            4. ``__aexit__`` catches the ``BaseException`` and cleans up.
            5. Assertion: temp directory is gone despite read-only files.
        """
        saved = None
        with pytest.raises(click.Abort):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate_git_clone(tmp)
                raise click.Abort()
        assert not saved.exists()

    async def test_click_abort_with_empty_dir_deletes(self):
        """S-2 sub: click.Abort before any files are written.

        Simulates the scenario where the branch selection is empty and
        ``click.Abort()`` is raised before any clone starts.  The temp
        directory exists (created by ``__aenter__``) but is empty.
        """
        saved = None
        with pytest.raises(click.Abort):
            async with TmpCloneManager() as tmp:
                saved = tmp
                raise click.Abort()
        assert not saved.exists()


# ============================================================================
# S-3  Exception in _fetch_issues_by_type()
# ============================================================================


class TestS03FetchIssuesException:
    """Verify cleanup when the SonarCloud API call fails.

    In production, ``_fetch_issues_by_type()`` calls the SonarCloud API.
    If the API returns an error, times out, or returns malformed data,
    a ``RuntimeError`` or similar exception propagates.  The context
    manager must clean up the cloned repository.
    """

    async def test_runtime_error_during_fetch_deletes(self):
        """S-3: RuntimeError from SonarCloud API cleans up.

        Workflow:
            1. Git clone content is created in the temp directory.
            2. ``RuntimeError("SonarCloud API error")`` is raised
               (simulating a failed API call).
            3. ``__aexit__`` catches the exception and removes the dir.
        """
        saved = None
        with pytest.raises(RuntimeError):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate_git_clone(tmp)
                raise RuntimeError("SonarCloud API error")
        assert not saved.exists()

    async def test_timeout_error_during_fetch_deletes(self):
        """S-3 sub: TimeoutError from SonarCloud API cleans up.

        ``TimeoutError`` is a subclass of ``Exception``, so this tests
        the normal exception path (not the ``BaseException`` fallback).
        """
        saved = None
        with pytest.raises(TimeoutError):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate_git_clone(tmp)
                raise TimeoutError("SonarCloud API timed out")
        assert not saved.exists()


# ============================================================================
# S-4  Exception in _process_files_with_issues()
# ============================================================================


class TestS04ProcessFilesException:
    """Verify cleanup when issue processing fails.

    In production, ``_process_files_with_issues()`` calls the LLM,
    parses the response, validates the fix, and applies it.  Failures
    can occur at any step: invalid JSON from the LLM, file I/O errors,
    validation failures, etc.
    """

    async def test_value_error_during_processing_deletes(self):
        """S-4: ValueError from LLM response parsing cleans up.

        Simulates the LLM returning invalid JSON that causes a
        ``ValueError`` during deserialization.
        """
        saved = None
        with pytest.raises(ValueError):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate_git_clone(tmp)
                raise ValueError("LLM returned invalid JSON")
        assert not saved.exists()

    async def test_ioerror_during_file_processing_deletes(self):
        """S-4 sub: IOError during file I/O cleans up.

        Simulates a failure when reading or writing the file being fixed
        (e.g., file was deleted between the SonarCloud report and the
        fix attempt, or permission denied on the cloned file).
        """
        saved = None
        with pytest.raises(IOError):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate(tmp)
                raise IOError("Permission denied: src/main.py")
        assert not saved.exists()

    async def test_type_error_during_validation_deletes(self):
        """S-4 sub: TypeError during fix validation cleans up.

        Simulates a bug in the validation logic that produces an
        unexpected ``TypeError``.  Even unexpected exceptions must
        trigger cleanup.
        """
        saved = None
        with pytest.raises(TypeError):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate(tmp)
                raise TypeError("unsupported operand type(s)")
        assert not saved.exists()


# ============================================================================
# S-5  KeyboardInterrupt (Ctrl+C)
# ============================================================================


class TestS05KeyboardInterrupt:
    """Verify cleanup when the user presses Ctrl+C.

    ``KeyboardInterrupt`` is a ``BaseException`` (not ``Exception``).
    Before the bug fix, the ``except Exception`` handler in
    ``__aexit__`` would miss it.  The fix changed to
    ``except BaseException`` with a synchronous fallback.
    """

    async def test_keyboard_interrupt_with_git_clone_deletes(self):
        """S-5: Ctrl+C after git clone cleans up.

        Workflow:
            1. Git clone content (including read-only objects) is created.
            2. ``KeyboardInterrupt`` is raised (simulating Ctrl+C).
            3. ``__aexit__`` catches it via ``except BaseException``.
            4. If the executor await is interrupted, the synchronous
               fallback calls ``remove_tmp_files()`` directly.
            5. ``_handle_remove_readonly`` handles the read-only files.
        """
        saved = None
        with pytest.raises(KeyboardInterrupt):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate_git_clone(tmp)
                raise KeyboardInterrupt()
        assert not saved.exists()

    async def test_keyboard_interrupt_with_regular_files_deletes(self):
        """S-5 sub: Ctrl+C with regular (non-git) files cleans up.

        Verifies cleanup when the directory contains only normal files
        (no read-only git objects).
        """
        saved = None
        with pytest.raises(KeyboardInterrupt):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate(tmp)
                raise KeyboardInterrupt()
        assert not saved.exists()


# ============================================================================
# S-6  SIGTERM -> SystemExit
# ============================================================================


class TestS06SigtermSystemExit:
    """Verify cleanup when the process receives SIGTERM.

    Python's default SIGTERM handler raises ``SystemExit``, which is a
    ``BaseException``.  The ``__aexit__`` handler must catch it.
    """

    async def test_system_exit_nonzero_deletes(self):
        """S-6: SystemExit(1) from SIGTERM cleans up.

        ``SystemExit(1)`` is what Python raises when a SIGTERM signal
        is received.  The ``except BaseException`` block catches it,
        and the synchronous fallback ensures cleanup.
        """
        saved = None
        with pytest.raises(SystemExit):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate_git_clone(tmp)
                raise SystemExit(1)
        assert not saved.exists()

    async def test_system_exit_with_string_message_deletes(self):
        """S-6 sub: SystemExit with a string message cleans up.

        ``SystemExit`` can be raised with any argument, not just an
        integer.  Some frameworks raise ``SystemExit("error message")``.
        """
        saved = None
        with pytest.raises(SystemExit):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate(tmp)
                raise SystemExit("fatal error")
        assert not saved.exists()


# ============================================================================
# S-7  SIGKILL — startup sweep cleans orphans
# ============================================================================


class TestS07SigkillOrphanSweep:
    """Verify that SIGKILL-orphaned directories are cleaned by the
    startup sweep.

    SIGKILL cannot be caught by any Python handler — the process is
    killed instantly and ``__aexit__`` never runs.  The safety net is
    ``sweep_orphaned_tmp_dirs()``, which runs at CLI startup (cli.py
    line 713) and removes any ``devdox_*_test`` directory older than
    1 hour.
    """

    async def test_sigkill_orphan_with_git_clone_cleaned(self):
        """S-7: Orphaned dir with git-clone content cleaned by sweep.

        Workflow:
            1. Manually create a ``devdox_*_test`` directory (bypassing
               the context manager, simulating SIGKILL).
            2. Populate with git-clone content including read-only files.
            3. Set mtime to 2 hours ago (older than 1-hour threshold).
            4. Call ``sweep_orphaned_tmp_dirs(max_age_seconds=3600)``.
            5. Assertion: the orphaned directory is removed.

        Note: the sweep uses ``shutil.rmtree`` directly (without the
        ``onerror`` handler), so this also validates that the sweep
        can handle typical orphaned directories.
        """
        orphan = Path(generate_tmp_path())
        _populate_git_clone(orphan)
        assert orphan.exists()

        old_time = time.time() - 7200
        os.utime(str(orphan), (old_time, old_time))

        removed = sweep_orphaned_tmp_dirs(max_age_seconds=3600)
        assert removed >= 1
        assert not orphan.exists()

    async def test_sigkill_orphan_with_regular_files_cleaned(self):
        """S-7 sub: Orphaned dir with regular files cleaned by sweep.

        Same as the main test but with regular (non-read-only) files
        to verify the sweep works regardless of file permissions.
        """
        orphan = Path(generate_tmp_path())
        _populate(orphan)
        old_time = time.time() - 7200
        os.utime(str(orphan), (old_time, old_time))

        removed = sweep_orphaned_tmp_dirs(max_age_seconds=3600)
        assert removed >= 1
        assert not orphan.exists()

    async def test_fresh_dir_not_cleaned_by_sweep(self):
        """S-7 sub: Recently-created dir is NOT cleaned by sweep.

        Verifies the age threshold: a directory created just now
        (mtime < 1 hour) must not be removed, even if its name
        matches the ``devdox_*_test`` pattern.  This prevents the
        sweep from deleting an active session's directory.
        """
        fresh = Path(generate_tmp_path())
        _populate(fresh)
        try:
            removed = sweep_orphaned_tmp_dirs(max_age_seconds=3600)
            assert fresh.exists(), "Fresh dir should NOT be cleaned"
        finally:
            # Manual cleanup since we bypassed the context manager
            import shutil
            shutil.rmtree(fresh, ignore_errors=True)


# ============================================================================
# S-8  asyncio.CancelledError
# ============================================================================


class TestS08AsyncioCancelledError:
    """Verify cleanup when an asyncio task is cancelled.

    ``asyncio.CancelledError`` is a ``BaseException`` (since Python 3.9),
    which means it bypasses ``except Exception``.  The bug fix changed
    the handler to ``except BaseException`` to catch it.

    In production, task cancellation can happen when the CLI sets a
    timeout on the processing loop, or when the user cancels from an
    outer scope.
    """

    async def test_task_cancellation_deletes(self):
        """S-8: asyncio task cancellation cleans up temp directory.

        Workflow:
            1. An async task enters the ``TmpCloneManager`` context.
            2. Inside, it calls ``await asyncio.sleep(60)`` to block.
            3. The test cancels the task after 50ms.
            4. ``CancelledError`` propagates through ``__aexit__``.
            5. The ``except BaseException`` block catches it; if the
               executor await is itself cancelled, the synchronous
               fallback runs ``remove_tmp_files()`` directly.
            6. Assertion: the temp directory is removed.
        """
        holder = [None]

        async def worker():
            async with TmpCloneManager() as tmp:
                holder[0] = tmp
                _populate_git_clone(tmp)
                await asyncio.sleep(60)

        task = asyncio.create_task(worker())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert holder[0] is not None
        assert not holder[0].exists()

    async def test_cancelled_error_raised_directly_deletes(self):
        """S-8 sub: CancelledError raised directly (not via task.cancel()).

        Tests the case where ``asyncio.CancelledError`` is raised
        explicitly inside the ``async with`` block, rather than via
        external task cancellation.  This can happen in coroutines
        that check for cancellation manually.
        """
        saved = None
        with pytest.raises(asyncio.CancelledError):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate(tmp)
                raise asyncio.CancelledError()
        assert not saved.exists()


# ============================================================================
# S-9  SystemExit (explicit exit() call)
# ============================================================================


class TestS09ExplicitSystemExit:
    """Verify cleanup when ``sys.exit()`` or ``exit()`` is called.

    ``sys.exit()`` raises ``SystemExit``, a ``BaseException``.  This
    scenario covers explicit exit calls in user code, as opposed to
    S-6 where SIGTERM triggers it.
    """

    async def test_system_exit_zero_deletes(self):
        """S-9: SystemExit(0) (clean exit) triggers cleanup.

        Even a "successful" exit (code 0) must clean up.  This is
        relevant when the CLI calls ``sys.exit(0)`` after completing
        its work but before the context manager naturally exits.
        """
        saved = None
        with pytest.raises(SystemExit):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate(tmp)
                raise SystemExit(0)
        assert not saved.exists()

    async def test_system_exit_custom_code_deletes(self):
        """S-9 sub: SystemExit with custom exit code (42) triggers cleanup.

        Verifies that arbitrary exit codes don't affect cleanup behaviour.
        """
        saved = None
        with pytest.raises(SystemExit):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate_git_clone(tmp)
                raise SystemExit(42)
        assert not saved.exists()


# ============================================================================
# S-10  OOM kill — startup sweep cleans orphans
# ============================================================================


class TestS10OomKillOrphanSweep:
    """Verify that OOM-killed sessions leave orphans cleaned by sweep.

    When the Linux OOM killer terminates the process, it behaves like
    SIGKILL — no Python cleanup runs.  The mechanism is identical to
    S-7: the startup sweep removes stale directories.
    """

    async def test_oom_orphan_cleaned_by_startup_sweep(self):
        """S-10: OOM-orphaned directory cleaned by sweep.

        Workflow identical to S-7:
            1. Create a ``devdox_*_test`` directory manually.
            2. Populate with typical content.
            3. Set mtime to 2 hours ago.
            4. Run ``sweep_orphaned_tmp_dirs()``.
            5. Assert the directory was removed.

        The test is separate from S-7 to make the scenario mapping
        explicit, even though the mechanism is the same.
        """
        orphan = Path(generate_tmp_path())
        _populate(orphan)
        old_time = time.time() - 7200
        os.utime(str(orphan), (old_time, old_time))

        removed = sweep_orphaned_tmp_dirs(max_age_seconds=3600)
        assert removed >= 1
        assert not orphan.exists()

    async def test_oom_multiple_orphans_cleaned(self):
        """S-10 sub: Multiple OOM-orphaned directories all cleaned.

        If the tool was invoked multiple times and the machine OOM-killed
        each session, there may be several orphaned directories.  The
        sweep must clean all of them, not just the first.
        """
        orphans = []
        for _ in range(3):
            orphan = Path(generate_tmp_path())
            _populate(orphan)
            old_time = time.time() - 7200
            os.utime(str(orphan), (old_time, old_time))
            orphans.append(orphan)

        removed = sweep_orphaned_tmp_dirs(max_age_seconds=3600)
        assert removed >= 3
        for orphan in orphans:
            assert not orphan.exists()


# ============================================================================
# S-11  Disk full during clone — partial clone cleaned up
# ============================================================================


class TestS11DiskFullDuringClone:
    """Verify cleanup when disk-full errors interrupt the git clone.

    When ``git clone`` runs out of disk space, it leaves a partial
    directory tree: some ``.git/`` directories exist but the clone is
    incomplete.  The ``OSError`` propagates and ``__aexit__`` must
    remove whatever was partially created.
    """

    async def test_partial_clone_cleaned_on_oserror(self):
        """S-11: Partial git clone directory cleaned after OSError.

        Workflow:
            1. Temp directory created by context manager.
            2. Partially populate it: ``.git/objects/``, ``HEAD``,
               ``src/`` exist, simulating a clone that stopped mid-way.
            3. ``OSError("No space left on device")`` is raised.
            4. ``__aexit__`` cleans up the partial tree.
        """
        saved = None
        with pytest.raises(OSError):
            async with TmpCloneManager() as tmp:
                saved = tmp
                (tmp / ".git" / "objects").mkdir(parents=True)
                (tmp / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
                (tmp / "src").mkdir()
                raise OSError("No space left on device")
        assert not saved.exists()

    async def test_empty_dir_cleaned_on_disk_full_before_any_write(self):
        """S-11 sub: Disk full before any files written.

        Edge case: the temp directory is created (by ``mkdtemp``) but
        the very first file write fails.  The directory exists but is
        empty.  Cleanup must still succeed.
        """
        saved = None
        with pytest.raises(OSError):
            async with TmpCloneManager() as tmp:
                saved = tmp
                raise OSError("No space left on device")
        assert not saved.exists()


# ============================================================================
# S-12  Network failure during clone — partial clone cleaned up
# ============================================================================


class TestS12NetworkFailureDuringClone:
    """Verify cleanup when network errors interrupt the git clone.

    Network failures during ``git clone`` raise ``ConnectionError``,
    ``TimeoutError``, or ``git.exc.GitCommandError``.  The partial
    clone must be cleaned up.
    """

    async def test_connection_error_cleans_partial_clone(self):
        """S-12: ConnectionError during clone cleans up partial tree.

        Simulates a network timeout or DNS failure mid-clone: some
        git metadata files exist but the repository is incomplete.
        """
        saved = None
        with pytest.raises(ConnectionError):
            async with TmpCloneManager() as tmp:
                saved = tmp
                (tmp / ".git" / "objects" / "pack").mkdir(parents=True)
                (tmp / ".git" / "config").write_text("[core]\n")
                raise ConnectionError("Connection timed out")
        assert not saved.exists()

    async def test_connection_refused_cleans_up(self):
        """S-12 sub: ConnectionRefusedError during clone cleans up.

        ``ConnectionRefusedError`` is a subclass of ``ConnectionError``
        and ``OSError``.  Verifies the same cleanup path works for this
        more specific error.
        """
        saved = None
        with pytest.raises(ConnectionRefusedError):
            async with TmpCloneManager() as tmp:
                saved = tmp
                (tmp / ".git").mkdir()
                raise ConnectionRefusedError("Connection refused")
        assert not saved.exists()

    async def test_timeout_error_during_clone_cleans_up(self):
        """S-12 sub: TimeoutError during git clone cleans up.

        ``TimeoutError`` is a subclass of ``OSError``.  Tests the
        scenario where the connection hangs and times out.
        """
        saved = None
        with pytest.raises(TimeoutError):
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate_git_clone(tmp)
                raise TimeoutError("Connection timed out after 30s")
        assert not saved.exists()


# ============================================================================
# S-13  Concurrent invocation — each manager cleans its own dir
# ============================================================================


class TestS13ConcurrentInvocations:
    """Verify that multiple concurrent TmpCloneManager instances each
    clean up their own directory without interfering with each other.

    Each ``TmpCloneManager`` instance has its own ``_tmp_path``.  There
    is no shared mutable state between instances.  This test verifies
    isolation under concurrent execution.
    """

    async def test_concurrent_managers_all_clean(self):
        """S-13: Five concurrent managers all clean up successfully.

        Workflow:
            1. Launch 5 async tasks, each using its own
               ``TmpCloneManager``.
            2. Each task populates its directory and does a short sleep
               (to simulate async work like API calls).
            3. All tasks complete normally.
            4. Assertion: all 5 temp directories are removed.
        """
        async def run_one(index: int) -> Path:
            async with TmpCloneManager() as tmp:
                _populate(tmp)
                (tmp / f"worker_{index}.txt").write_text(f"worker {index}\n")
                await asyncio.sleep(0.01)
                return tmp

        tasks = [asyncio.create_task(run_one(i)) for i in range(5)]
        paths = await asyncio.gather(*tasks)

        for p in paths:
            assert not p.exists(), f"Leaked dir: {p}"

    async def test_concurrent_mixed_success_and_failure(self):
        """S-13 sub: Concurrent managers where some succeed and some fail.

        Verifies isolation when some tasks complete normally and others
        raise exceptions.  Every temp directory — whether its task
        succeeded or failed — must be cleaned up.

        Tasks with even indices succeed; tasks with odd indices raise
        ``ValueError``.
        """
        results = {}

        async def run_one(index: int) -> Path:
            async with TmpCloneManager() as tmp:
                _populate(tmp)
                results[index] = tmp
                if index % 2 == 1:
                    raise ValueError(f"task {index} failed")
                await asyncio.sleep(0.01)
                return tmp

        tasks = [asyncio.create_task(run_one(i)) for i in range(6)]
        done = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in results.items():
            assert not result.exists(), f"Leaked dir for task {idx}: {result}"

    async def test_concurrent_with_cancellation(self):
        """S-13 sub: Concurrent managers where one task is cancelled.

        Verifies that task cancellation of one manager doesn't affect
        the cleanup of other managers.
        """
        paths = {}

        async def run_slow(index: int):
            async with TmpCloneManager() as tmp:
                paths[index] = tmp
                _populate(tmp)
                await asyncio.sleep(60)  # will be cancelled

        async def run_fast(index: int):
            async with TmpCloneManager() as tmp:
                paths[index] = tmp
                _populate(tmp)
                await asyncio.sleep(0.01)

        slow_task = asyncio.create_task(run_slow(0))
        fast_tasks = [asyncio.create_task(run_fast(i)) for i in range(1, 4)]

        await asyncio.sleep(0.05)
        slow_task.cancel()

        await asyncio.gather(*fast_tasks)
        with pytest.raises(asyncio.CancelledError):
            await slow_task

        for idx, p in paths.items():
            assert not p.exists(), f"Leaked dir for task {idx}: {p}"


# ============================================================================
# Cross-platform edge cases: read-only and permission-locked files
# ============================================================================


class TestCrossPlatformEdgeCases:
    """Verify cleanup handles cross-platform filesystem quirks.

    Git creates read-only files in ``.git/objects/``.  On Windows,
    ``shutil.rmtree`` fails with ``PermissionError`` on these files
    unless a custom ``onerror`` handler calls ``os.chmod()`` first.
    On macOS, the same issue can occur with SIP-protected or
    quarantined files.

    The ``_handle_remove_readonly`` function in ``file_indentation.py``
    provides this handler.  These tests verify it works on the current
    platform.
    """

    async def test_readonly_git_objects_deleted(self):
        """Read-only pack files (0o444) in .git/objects/ are removed.

        Creates a simulated git clone with read-only ``.idx`` and
        ``refs/heads/main`` files, then verifies the entire tree is
        deleted on context manager exit.

        This is the regression test for the original bug report:
        "my team leader tested it on his mac and did not delete".
        """
        saved = None
        async with TmpCloneManager() as tmp:
            saved = tmp
            _populate_git_clone(tmp)
            idx = tmp / ".git" / "objects" / "pack" / "pack-abc123.idx"
            assert idx.exists()
            assert not os.access(str(idx), os.W_OK) or os.getuid() == 0
        assert not saved.exists()

    async def test_readonly_directory_deleted(self):
        """A directory with mode 0o555 (read+execute, no write) is removed.

        On some platforms, removing files from a read-only directory
        requires restoring write permission on the directory itself.
        The ``onerror`` handler in ``remove_tmp_files`` handles this.
        """
        saved = None
        async with TmpCloneManager() as tmp:
            saved = tmp
            sub = tmp / "locked_dir"
            sub.mkdir()
            (sub / "file.txt").write_text("data")
            sub.chmod(0o555)
        assert not saved.exists()

    async def test_mixed_permissions_deleted(self):
        """A tree with mixed read-only and writable files is removed.

        Real git clones contain a mix: ``.git/objects/`` is read-only
        while working tree files are writable.  This test verifies
        that ``_handle_remove_readonly`` only kicks in when needed and
        doesn't interfere with normal deletion.
        """
        saved = None
        async with TmpCloneManager() as tmp:
            saved = tmp
            _populate_git_clone(tmp)
            _populate(tmp)  # also adds writable files
        assert not saved.exists()


# ============================================================================
# Bug-fix regressions: on_cleanup callback isolation
# ============================================================================


class TestOnCleanupCallbackIsolation:
    """Verify that failures in the optional ``on_cleanup`` callback do
    not prevent the actual temp directory deletion.

    Bug history: the original ``__aexit__`` implementation had both
    ``on_cleanup`` and ``cleanup_fn`` in the same ``try`` block.  If
    ``on_cleanup`` raised, the ``cleanup_fn`` was skipped.

    The fix isolates ``on_cleanup`` in its own ``try/except`` so that
    a crashing callback cannot block deletion.
    """

    async def test_callback_crash_still_deletes(self):
        """on_cleanup raises RuntimeError — temp dir is still deleted.

        Workflow:
            1. A faulty ``on_cleanup`` callback is provided that raises.
            2. The ``async with`` block exits normally.
            3. ``__aexit__`` calls ``on_cleanup`` in its own
               ``try/except`` (the exception is swallowed).
            4. ``cleanup_fn`` runs and deletes the directory.
        """
        def bad_callback(p: str) -> None:
            raise RuntimeError("callback exploded")

        saved = None
        async with TmpCloneManager(on_cleanup=bad_callback) as tmp:
            saved = tmp
            _populate(tmp)
        assert not saved.exists()

    async def test_callback_crash_plus_body_exception_still_deletes(self):
        """on_cleanup crash + body exception: dir is still removed.

        Worst case: the ``async with`` body raises ``ValueError`` AND
        the ``on_cleanup`` callback also raises.  The body exception
        must propagate to the caller, and the temp dir must still be
        cleaned up.
        """
        def bad_callback(p: str) -> None:
            raise TypeError("callback broke")

        saved = None
        with pytest.raises(ValueError):
            async with TmpCloneManager(on_cleanup=bad_callback) as tmp:
                saved = tmp
                _populate(tmp)
                raise ValueError("body failed")
        assert not saved.exists()

    async def test_callback_crash_with_keyboard_interrupt_still_deletes(self):
        """on_cleanup crash + KeyboardInterrupt: dir is still removed.

        Combines Bug 1 (BaseException not caught) and Bug 2 (callback
        crash blocks cleanup).  Both must be handled correctly.
        """
        def bad_callback(p: str) -> None:
            raise RuntimeError("callback crashed")

        saved = None
        with pytest.raises(KeyboardInterrupt):
            async with TmpCloneManager(on_cleanup=bad_callback) as tmp:
                saved = tmp
                _populate_git_clone(tmp)
                raise KeyboardInterrupt()
        assert not saved.exists()


# ============================================================================
# Cleanup fallback: stderr warning on total failure
# ============================================================================


class TestCleanupFallbackStderrWarning:
    """Verify that when both the executor cleanup and synchronous
    fallback fail, a warning is printed to stderr.

    This is Bug 3 from the investigation: ``logger.exception()`` alone
    is invisible because the logger has no console handler.  The fix
    adds ``print(..., file=sys.stderr)`` so the user always sees the
    warning.
    """

    async def test_total_cleanup_failure_prints_stderr(self):
        """Both executor and sync cleanup fail -> stderr warning printed.

        Workflow:
            1. Provide a ``cleanup_fn`` that always raises ``OSError``.
            2. Enter the context manager and populate the directory.
            3. On exit, the executor cleanup fails (first call).
            4. The synchronous fallback also fails (second call).
            5. ``logger.exception()`` is called.
            6. ``print(..., file=sys.stderr)`` is called.
            7. The temp directory is NOT deleted (both attempts failed),
               but the context manager does NOT crash.

        We capture stderr to verify the warning message.
        """
        call_count = 0

        def always_fail(path: str) -> bool:
            nonlocal call_count
            call_count += 1
            raise OSError("Permission denied")

        with patch("devdox_ai_sonar.utils.file_indentation.logger"):
            with patch("builtins.print") as mock_print:
                async with TmpCloneManager(cleanup_fn=always_fail) as tmp:
                    saved = tmp
                    _populate(tmp)

        # Verify stderr warning was printed
        mock_print.assert_called()
        stderr_calls = [
            c for c in mock_print.call_args_list
            if c.kwargs.get("file") is sys.stderr
        ]
        assert len(stderr_calls) >= 1
        assert "could not remove temporary directory" in str(stderr_calls[0])

        # Manual cleanup since our cleanup_fn was intentionally broken
        import shutil
        shutil.rmtree(saved, ignore_errors=True)


# ============================================================================
# Stress / edge cases
# ============================================================================


class TestStressAndEdgeCases:
    """Verify cleanup under unusual but valid filesystem conditions.

    These tests push the cleanup logic to its limits with deep nesting,
    large file counts, and rapid sequential usage.
    """

    async def test_deeply_nested_directory_deleted(self):
        """A 20-level deep directory tree is fully removed.

        ``shutil.rmtree`` must recurse through all 20 levels.  On some
        platforms (Windows), paths longer than 260 characters can cause
        issues.  This test verifies that the cleanup handles deep nesting.
        """
        saved = None
        async with TmpCloneManager() as tmp:
            saved = tmp
            deep = tmp
            for i in range(20):
                deep = deep / f"level_{i}"
            deep.mkdir(parents=True)
            (deep / "leaf.txt").write_text("deep file")
        assert not saved.exists()

    async def test_many_files_deleted(self):
        """A directory with 200 files is fully removed.

        Verifies that cleanup performance is acceptable with a moderate
        number of files and that no files are missed.
        """
        saved = None
        async with TmpCloneManager() as tmp:
            saved = tmp
            for i in range(200):
                (tmp / f"file_{i:04d}.py").write_text(f"x = {i}\n")
        assert not saved.exists()

    async def test_sequential_runs_all_clean(self):
        """Five sequential context-manager uses all clean up.

        Verifies that the cleanup function doesn't leak state between
        invocations — each run creates a new temp dir and cleans it
        independently.
        """
        paths = []
        for _ in range(5):
            async with TmpCloneManager() as tmp:
                paths.append(tmp)
                _populate(tmp)
        for p in paths:
            assert not p.exists()

    async def test_symlink_inside_dir_deleted(self):
        """A temp directory containing a symlink is fully removed.

        ``shutil.rmtree`` should remove the symlink itself, not follow
        it and delete the target.  This test verifies that symlinks
        inside the temp directory don't cause issues.
        """
        saved = None
        external_file = Path(generate_tmp_path()) / "external.txt"
        external_file.parent.mkdir(parents=True, exist_ok=True)
        external_file.write_text("external data\n")
        try:
            async with TmpCloneManager() as tmp:
                saved = tmp
                _populate(tmp)
                link = tmp / "link_to_external"
                link.symlink_to(external_file)
            assert not saved.exists()
            # The external file should NOT have been deleted
            assert external_file.exists()
        finally:
            import shutil
            shutil.rmtree(external_file.parent, ignore_errors=True)
