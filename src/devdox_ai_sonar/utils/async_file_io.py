"""Async file I/O utilities for DevDox AI Sonar."""
import aiofiles
from pathlib import Path
import tomli
import tomli_w
from typing import List, Optional
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor


class AsyncFileReader:
    """Async file reading with proper error handling."""

    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def read_text(self, file_path: str | Path) -> str:
        """Read file content asynchronously."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except UnicodeDecodeError:
            # Fallback for non-UTF8 files
            async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                return await f.read()
        except Exception as e:
            raise IOError(f"Failed to read {file_path}: {e}")

    async def read_lines(self, file_path: str | Path) -> List[str]:
        """Read file lines asynchronously."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.readlines()

    async def read_json_file(self, file_path: str | Path) -> Optional[dict]:
        """Read JSON file asynchronously."""
        try:
            content = await self.read_text(file_path)
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    async def read_toml_file(self, file_path: str | Path) -> Optional[dict]:
        """Read TOML file asynchronously."""


        try:
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()
                return tomli.loads(content.decode('utf-8'))

        except Exception as e:
            raise IOError(f"Failed to read {file_path}: {e}")

    async def write_text(self, file_path: str | Path, content: str) -> None:
        """Write content to file asynchronously."""
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(content)

    async def read_multiple(self, file_paths: List[str | Path]) -> dict[str, str]:
        """Read multiple files concurrently."""
        tasks = [self.read_text(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            str(path): result if not isinstance(result, Exception) else None
            for path, result in zip(file_paths, results)
        }

    async def file_exists(self, file_path: str | Path) -> bool:
        """Check if file exists (async-friendly)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            Path(file_path).exists
        )


class AsyncFileWriter:
    """Async file writing with atomic operations."""

    async def write_atomic(self, file_path: str | Path, content: str) -> None:
        """Atomic write with temp file + rename."""
        import tempfile
        import os

        file_path = Path(file_path)
        temp_path = file_path.with_suffix('.tmp')

        try:
            # Write to temp file
            async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
                await f.write(content)

            # Atomic rename
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                os.replace,
                temp_path,
                file_path
            )
        except Exception as e:
            # Cleanup temp file on failure
            if temp_path.exists():
                await loop.run_in_executor(None, temp_path.unlink)
            raise IOError(f"Failed to write {file_path}: {e}")