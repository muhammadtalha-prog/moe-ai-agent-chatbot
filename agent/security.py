import os
import re
import logging
from pathlib import Path
from typing import Tuple, Union

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

def get_safe_path(file_path: Union[str, Path], base_dir: Union[str, Path] = None) -> Path:
    """
    Get a safe file path ensuring it's within the base directory.
    
    Args:
        file_path: Desired file path
        base_dir: Allowed base directory
        
    Returns:
        Resolved Path object guaranteed to be within base_dir
        
    Raises:
        ValueError: If path traversal is detected
    """
    if base_dir is None:
        base_dir = get_workspace_dir()
    base_dir = Path(base_dir).resolve()
    file_path = Path(file_path).resolve()
    
    # Check if file_path is within base_dir
    try:
        file_path.relative_to(base_dir)
    except ValueError:
        raise ValueError(f"Path traversal attempt: {file_path} is outside {base_dir}")
    
    return file_path

class SafeLogger:
    @staticmethod
    def log_error(error: Exception, context: str = ""):
        """Log error without exposing sensitive data like API keys"""
        error_str = str(error)
        # Mask Groq keys (gsk_...) and Gemini keys (AIzaSy...)
        masked = re.sub(r'gsk_[a-zA-Z0-9]{30,60}', '[GROQ_API_KEY_REDACTED]', error_str)
        masked = re.sub(r'AIzaSy[a-zA-Z0-9_\-]{33}', '[GEMINI_API_KEY_REDACTED]', masked)
        masked = re.sub(r'sk-[a-zA-Z0-9]{20,50}', '[API_KEY_REDACTED]', masked)
        
        # Mask other sensitive patterns
        sensitive_patterns = [
            (r'"[A-Za-z0-9]{32,}"', '[TOKEN_REDACTED]'),
            (r'key:[^,}]+', 'key:[REDACTED]'),
        ]
        for pattern, replacement in sensitive_patterns:
            masked = re.sub(pattern, replacement, masked)
            
        logging.error(f"{context}: {masked}" if context else masked)

def sanitize_user_input(user_input: str) -> str:
    """Sanitize user input to prevent prompt injection and restrict length."""
    if not user_input or not isinstance(user_input, str):
        return ""
    
    # Remove excess whitespace
    user_input = ' '.join(user_input.split())
    
    # Limit length (10,000 characters)
    max_length = 10000
    if len(user_input) > max_length:
        user_input = user_input[:max_length]
    
    # Strip non-printable/control characters, keeping standard printable keyboard symbols and spaces
    import string
    printable_chars = set(string.printable)
    user_input = ''.join(filter(lambda x: x in printable_chars, user_input))
    
    return user_input

def validate_and_secure_file(uploaded_file) -> Tuple[str, bytes]:
    """Validate file size, extension, and content MIME type safely."""
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024
    if uploaded_file.size > max_size:
        raise ValueError(f"File too large. Max size: {max_size // (1024 * 1024)}MB")
    
    file_content = uploaded_file.getvalue()
    orig_name = uploaded_file.name
    # Check for path traversal elements in filename
    if '..' in orig_name or '/' in orig_name or '\\' in orig_name:
        raise ValueError("Invalid filename contains path traversal separators.")
        
    filename = Path(orig_name).name
        
    # Check MIME type
    mime_type = None
    try:
        import magic
        mime_type = magic.from_buffer(file_content, mime=True)
    except Exception as e:
        # Fallback to mimetypes detection
        import mimetypes
        mime_type, _ = mimetypes.guess_type(filename)
        
    ALLOWED_MIME_TYPES = {
        'text/plain': ['.txt'],
        'application/pdf': ['.pdf'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
        'text/csv': ['.csv'],
        'application/json': ['.json'],
        'text/yaml': ['.yaml', '.yml'],
        'application/x-yaml': ['.yaml', '.yml'],
        'text/html': ['.html', '.htm'],
        'text/markdown': ['.md', '.markdown'],
    }
    
    if not mime_type:
        raise ValueError("Could not determine file type.")
        
    mime_type = mime_type.lower()
    ext = Path(filename).suffix.lower()
    
    # Check if the detected MIME type is allowed
    if mime_type not in ALLOWED_MIME_TYPES:
        # If it is text/plain but has another allowed extension, allow it
        if mime_type == 'text/plain' or mime_type.startswith('text/'):
            is_valid_ext = False
            for allowed_exts in ALLOWED_MIME_TYPES.values():
                if ext in allowed_exts:
                    is_valid_ext = True
                    break
            if not is_valid_ext:
                raise ValueError(f"Extension '{ext}' is not allowed for content of type '{mime_type}'.")
        else:
            raise ValueError(f"File type '{mime_type}' is not allowed.")
    else:
        # MIME is registered. Verify the extension is acceptable for it.
        allowed_exts = ALLOWED_MIME_TYPES[mime_type]
        if ext not in allowed_exts:
            # Let it pass if the mime type is a generic text type and the extension is a known text type
            if mime_type.startswith('text/') or mime_type == 'application/json':
                is_valid_ext = False
                for other_exts in ALLOWED_MIME_TYPES.values():
                    if ext in other_exts:
                        is_valid_ext = True
                        break
                if not is_valid_ext:
                    raise ValueError(f"Extension '{ext}' does not match detected file type '{mime_type}'.")
            else:
                raise ValueError(f"Extension '{ext}' does not match detected file type '{mime_type}'.")
            
    return filename, file_content

