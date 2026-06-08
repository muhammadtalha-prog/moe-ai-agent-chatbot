import os
from pathlib import Path

def get_workspace_dir() -> Path:
    """
    Returns the absolute path of the workspace directory.
    """
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).resolve()

def is_safe_path(path_str: str, base_dir: Path = None) -> bool:
    """
    Validates that a file path is safe and does not escape the workspace directory
    (preventing path traversal attacks like ../../).
    """
    if base_dir is None:
        base_dir = get_workspace_dir()
    
    try:
        # Resolve target absolute path
        target_path = Path(path_str).resolve()
        # Ensure target path starts with the base workspace path
        return base_dir in target_path.parents or target_path == base_dir
    except (ValueError, OSError):
        return False

def get_safe_path(path_str: str, base_dir: Path = None) -> Path:
    """
    Resolves and returns a safe Path object. Raises PermissionError if unsafe.
    """
    if base_dir is None:
        base_dir = get_workspace_dir()
    
    # Resolve absolute path
    target_path = Path(path_str).resolve()
    
    # Check safety
    if not is_safe_path(target_path, base_dir):
        raise PermissionError(
            f"Security Error: Access to path '{path_str}' is denied. It falls outside the allowed workspace '{base_dir}'."
        )
    
    return target_path
