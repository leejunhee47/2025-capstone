"""
Temporary file management for audio and video extraction
Provides context manager for safe temporary file handling
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import logging

from config import Config

logger = logging.getLogger(__name__)


class TempFileManager:
    """
    Context manager for temporary file operations
    Automatically creates and cleans up temporary directories
    """

    def __init__(self, prefix: Optional[str] = None, auto_cleanup: bool = True):
        """
        Initialize TempFileManager

        Args:
            prefix: Prefix for temporary directory name
            auto_cleanup: Automatically cleanup on exit
        """
        self.prefix = prefix or Config.TEMP_DIR_PREFIX
        self.auto_cleanup = auto_cleanup
        self.temp_dir: Optional[Path] = None
        self._created_files = []

    def __enter__(self):
        """Enter context manager - create temporary directory"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix=self.prefix))
        logger.info(f"Created temporary directory: {self.temp_dir}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - cleanup if enabled"""
        if self.auto_cleanup:
            self.cleanup()
        return False

    def get_temp_path(self, filename: str) -> Path:
        """
        Get path for a temporary file

        Args:
            filename: Name of the file

        Returns:
            Path object for the temporary file
        """
        if self.temp_dir is None:
            raise RuntimeError("TempFileManager not initialized. Use with context manager.")

        temp_path = self.temp_dir / filename
        self._created_files.append(temp_path)
        return temp_path

    def create_temp_file(self, suffix: str = '', prefix: str = '') -> Path:
        """
        Create a temporary file with given suffix/prefix

        Args:
            suffix: File suffix (e.g., '.wav')
            prefix: File prefix

        Returns:
            Path to created temporary file
        """
        if self.temp_dir is None:
            raise RuntimeError("TempFileManager not initialized. Use with context manager.")

        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=self.temp_dir)
        os.close(fd)  # Close file descriptor

        temp_path = Path(temp_path)
        self._created_files.append(temp_path)
        return temp_path

    def cleanup(self):
        """Remove temporary directory and all contents"""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory {self.temp_dir}: {e}")
            finally:
                self.temp_dir = None
                self._created_files.clear()

    def get_temp_dir(self) -> Path:
        """Get the temporary directory path"""
        if self.temp_dir is None:
            raise RuntimeError("TempFileManager not initialized. Use with context manager.")
        return self.temp_dir

    def exists(self) -> bool:
        """Check if temporary directory exists"""
        return self.temp_dir is not None and self.temp_dir.exists()

    def get_disk_usage(self) -> dict:
        """
        Get disk usage statistics for temporary directory

        Returns:
            Dict with total, used, and free disk space in bytes
        """
        if self.temp_dir is None:
            return {'total': 0, 'used': 0, 'free': 0}

        stat = shutil.disk_usage(self.temp_dir)
        return {
            'total': stat.total,
            'used': stat.used,
            'free': stat.free
        }

    def get_size(self) -> int:
        """
        Get total size of temporary directory in bytes

        Returns:
            Size in bytes
        """
        if self.temp_dir is None or not self.temp_dir.exists():
            return 0

        total_size = 0
        for file_path in self.temp_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    def list_files(self) -> list:
        """
        List all files in temporary directory

        Returns:
            List of Path objects
        """
        if self.temp_dir is None or not self.temp_dir.exists():
            return []

        return [f for f in self.temp_dir.rglob('*') if f.is_file()]
