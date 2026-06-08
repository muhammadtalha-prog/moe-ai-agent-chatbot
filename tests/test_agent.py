import os
import json
import pytest
import numpy as np
from pathlib import Path

# Add local path to sys.path to ensure imports work
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.security import is_safe_path, get_safe_path, get_workspace_dir
from agent.memory import VectorStore
from agent.file_handler import read_file, write_file
from agent.mcp import mcp_registry
from agent.experts.orchestrator import Orchestrator

# Test directories
@pytest.fixture
def temp_workspace(tmp_path):
    return tmp_path

def test_security_paths(temp_workspace):
    # Safe path in workspace
    safe_file = temp_workspace / "test.txt"
    assert is_safe_path(safe_file, temp_workspace) == True

    # Unsafe path outside workspace
    unsafe_file = temp_workspace / "../../unsafe.txt"
    assert is_safe_path(unsafe_file, temp_workspace) == False
    
    with pytest.raises(PermissionError):
        get_safe_path(unsafe_file, temp_workspace)

def test_file_handler(temp_workspace):
    # Mock workspace path for file operations
    test_txt_path = str(temp_workspace / "test.txt")
    test_json_path = str(temp_workspace / "test.json")
    test_csv_path = str(temp_workspace / "test.csv")
    
    # 1. Text file read/write
    text_content = "Hello, this is a test text file."
    # Temporarily override get_workspace_dir to return temp_workspace in test context
    import agent.security
    original_get_workspace = agent.security.get_workspace_dir
    agent.security.get_workspace_dir = lambda: temp_workspace
    
    try:
        write_file(test_txt_path, text_content)
        read_text = read_file(test_txt_path)
        assert read_text == text_content
        
        # 2. JSON file read/write
        json_data = {"key": "value", "number": 42}
        write_file(test_json_path, json.dumps(json_data))
        read_json_str = read_file(test_json_path)
        read_json_data = json.loads(read_json_str)
        assert read_json_data["key"] == "value"
        assert read_json_data["number"] == 42
        
        # 3. CSV file read/write
        csv_content = "name,age,city\nAlice,30,New York\nBob,25,San Francisco"
        write_file(test_csv_path, csv_content)
        read_csv = read_file(test_csv_path)
        assert "Alice, 30, New York" in read_csv
        assert "Bob, 25, San Francisco" in read_csv

        # 4. Word Document (.docx) read/write
        docx_content = "This is a word document paragraph.\nSecond paragraph content."
        test_docx_path = str(temp_workspace / "test.docx")
        write_file(test_docx_path, docx_content)
        read_docx = read_file(test_docx_path)
        assert "This is a word document paragraph." in read_docx
        assert "Second paragraph content." in read_docx

        # 5. Excel Spreadsheet (.xlsx) read/write
        excel_content = "item,count\napple,5\nbanana,10"
        test_xlsx_path = str(temp_workspace / "test.xlsx")
        write_file(test_xlsx_path, excel_content)
        read_xlsx = read_file(test_xlsx_path)
        assert "apple" in read_xlsx
        assert "banana" in read_xlsx
        
    finally:
        agent.security.get_workspace_dir = original_get_workspace

def test_vector_store(temp_workspace):
    db_file = str(temp_workspace / "vector_store.json")
    store = VectorStore(db_file)
    
    # Add items
    doc1_id = store.add_text("Python programming language", {"topic": "python"})
    doc2_id = store.add_text("Machine learning with neural networks", {"topic": "ai"})
    
    assert len(store.documents) == 2
    
    # Verify save / load
    store2 = VectorStore(db_file)
    assert len(store2.documents) == 2
    
    # Query (in mock/fallback mode, query uses deterministic vector matching)
    results = store.query("Python language", top_k=1)
    assert len(results) > 0
    doc, score = results[0]
    assert "id" in doc
    assert "text" in doc

def test_mcp_registry():
    # Verify pre-registered tools
    defs = mcp_registry.get_tool_definitions()
    tool_names = [d["name"] for d in defs]
    assert "web_search" in tool_names
    assert "get_system_info" in tool_names
    assert "search_files" in tool_names
    assert "list_directory" in tool_names

    # Test executing system_info
    sys_info = mcp_registry.execute_tool("get_system_info", {})
    assert "OS:" in sys_info
    assert "Python Version:" in sys_info

def test_orchestrator_routing():
    # Setup orchestrator (mock db)
    orchestrator = Orchestrator("temp_vector_store.json")
    
    # Test command routing
    route1 = orchestrator.determine_route("/read data.csv")
    assert route1["expert"] == "file"
    assert route1["action"] == "read"
    assert route1["file_path"] == "data.csv"
    
    route2 = orchestrator.determine_route("/sys")
    assert route2["expert"] == "mcp"
    assert route2["tool_name"] == "get_system_info"
    
    route3 = orchestrator.determine_route("/remember Agent 1 is cool")
    assert route3["expert"] == "memory"
    assert route3["action"] == "remember"
    assert "Agent 1 is cool" in route3["text_to_remember"]
    
    route4 = orchestrator.determine_route("/search what is MCP")
    assert route4["expert"] == "mcp"
    assert route4["tool_name"] == "web_search"
    assert route4["tool_arguments"]["query"] == "what is MCP"
    
    # Clean up test database if created
    if os.path.exists("temp_vector_store.json"):
        os.remove("temp_vector_store.json")
