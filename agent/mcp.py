import os
import platform
import psutil
import datetime
from pathlib import Path
from typing import Dict, Any, Callable, List
from duckduckgo_search import DDGS
from agent.security import get_workspace_dir, is_safe_path, get_safe_path

class MCPRegistry:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any]):
        """Decorator to register a tool with a specific schema."""
        def decorator(func: Callable):
            self.tools[name] = {
                "name": name,
                "description": description,
                "parameters": parameters,
                "func": func
            }
            return func
        return decorator

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Executes a tool by name with arguments."""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' is not registered.")
        
        try:
            return self.tools[name]["func"](**arguments)
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns JSON schema definitions of all registered tools for the LLM."""
        defs = []
        for name, info in self.tools.items():
            defs.append({
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"]
            })
        return defs

# Instantiate global registry
mcp_registry = MCPRegistry()

@mcp_registry.register_tool(
    name="web_search",
    description="Performs a web search to fetch the latest information on a given topic.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to search the web for."
            }
        },
        "required": ["query"]
    }
)
def web_search(query: str) -> str:
    """Executes a search query using DuckDuckGo."""
    try:
        results_list = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results_list.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n")
        
        if not results_list:
            return "No web results found."
        return "\n---\n".join(results_list)
    except Exception as e:
        return f"Web search failed: {str(e)}"

@mcp_registry.register_tool(
    name="get_system_info",
    description="Retrieves basic environment and hardware statistics of the host system.",
    parameters={
        "type": "object",
        "properties": {}
    }
)
def get_system_info() -> str:
    """Gets system information diagnostics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        mem_used = mem.used / (1024 ** 3)
        mem_total = mem.total / (1024 ** 3)
        
        info = [
            f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
            f"Python Version: {platform.python_version()}",
            f"CPU Usage: {cpu_percent}%",
            f"Memory: {mem_used:.2f} GB / {mem_total:.2f} GB ({mem.percent}%)",
            f"Current Local Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        return "\n".join(info)
    except Exception as e:
        return f"Failed to get system info: {str(e)}"

@mcp_registry.register_tool(
    name="search_files",
    description="Searches for a text pattern or word in all files under the workspace directory.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The text string to look for."
            }
        },
        "required": ["pattern"]
    }
)
def search_files(pattern: str) -> str:
    """Searches for files containing a specific pattern within the workspace."""
    workspace = get_workspace_dir()
    matches = []
    
    # Simple recursive text search, skipping large binary files
    try:
        for root, dirs, files in os.walk(workspace):
            # Exclude hidden files, directories, and vector_store.json
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv']]
            for f in files:
                if f.startswith('.') or f == 'vector_store.json':
                    continue
                file_path = os.path.join(root, f)
                if not is_safe_path(file_path, workspace):
                    continue
                
                try:
                    # Only search readable extensions
                    if Path(f).suffix.lower() in ['.txt', '.py', '.md', '.json', '.csv', '.html', '.css', '.js']:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file_content:
                            for i, line in enumerate(file_content, 1):
                                if pattern.lower() in line.lower():
                                    matches.append(f"{f}:{i} -> {line.strip()}")
                except Exception:
                    pass  # Ignore files that can't be read
                    
        if not matches:
            return f"No matches found for pattern '{pattern}'."
        return "\n".join(matches[:30])  # Cap results
    except Exception as e:
        return f"File search failed: {str(e)}"

@mcp_registry.register_tool(
    name="list_directory",
    description="Lists files in a given directory path safely (defaults to workspace root).",
    parameters={
        "type": "object",
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Subdirectory path relative to the workspace root (e.g. 'agent' or '')."
            }
        }
    }
)
def list_directory(relative_path: str = "") -> str:
    """Lists safe directory contents."""
    try:
        workspace = get_workspace_dir()
        target_dir = (workspace / relative_path).resolve()
        
        # Verify safety
        if not is_safe_path(target_dir, workspace):
            return "Error: Path falls outside the workspace directory."
            
        if not target_dir.exists():
            return f"Directory '{relative_path}' does not exist."
        if not target_dir.is_dir():
            return f"Path '{relative_path}' is not a directory."
            
        items = os.listdir(target_dir)
        formatted_items = []
        for item in items:
            full_path = target_dir / item
            is_dir = "DIR" if full_path.is_dir() else "FILE"
            size = f"{full_path.stat().st_size} bytes" if full_path.is_file() else ""
            formatted_items.append(f"[{is_dir}] {item} {size}")
            
        return "\n".join(formatted_items) if formatted_items else "Directory is empty."
    except Exception as e:
        return f"Listing failed: {str(e)}"
