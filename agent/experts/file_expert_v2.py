import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from agent.file_processor import FileProcessor
from agent.file_handler import read_file, write_file
from agent.config import get_groq_client, get_groq_key, GROQ_CHAT_MODEL_NAME

class FileExpertV2:
    """Advanced file processing expert with AI capabilities"""
    
    def __init__(self):
        self.processor = FileProcessor()
    
    def respond(self, user_input: str, history: List[Dict[str, str]], memory: Any, **kwargs) -> str:
        """Process file operations and return markdown response"""
        action = kwargs.get("action")
        file_path = kwargs.get("file_path")
        output_path = kwargs.get("output_path")
        instruction = kwargs.get("instruction") or user_input
        
        if not file_path and action != 'generate':
            return "FileExpert Error: No file path was specified."

        try:
            if action == 'read':
                return self._handle_read(file_path)
            elif action == 'summarize':
                return self._handle_summarize(file_path, instruction)
            elif action in ['rewrite', 'modify']:
                return self._handle_modify(file_path, output_path, instruction)
            elif action == 'compare':
                return self._handle_compare(file_path, output_path)
            elif action == 'generate':
                return self._handle_generate(instruction, output_path)
            elif action == 'analyze':
                return self._handle_analyze(file_path)
            else:
                return f"FileExpert: Unknown action '{action}'."
        except Exception as e:
            return f"FileExpert Error: {str(e)}"
            
    def _handle_read(self, file_path: str) -> str:
        result = self.processor.process_file(file_path)
        meta = result["metadata"]
        content = result["content"]["raw"]
        
        return f"""### 📄 File Read Success

**File:** `{meta['filename']}`
**Type:** `{meta['type']}`
**Size:** `{meta['size_readable']}`
**Lines:** {meta['line_count']}
**Words:** {meta['word_count']}

### Content Preview:
```
{content[:2000]}
```
{'... (truncated)' if len(content) > 2000 else ''}
"""

    def _handle_summarize(self, file_path: str, instruction: str) -> str:
        result = self.processor.process_file(file_path, {"summary_focus": instruction})
        meta = result["metadata"]
        summary = result["summary"]
        stats = result["statistics"]
        
        # If Groq is connected, get a smarter AI summary
        if get_groq_key():
            try:
                client = get_groq_client()
                prompt = (
                    f"Summarize the following content from file '{meta['filename']}'.\n"
                    f"Focus/Instruction: {instruction}\n\n"
                    f"--- Content ---\n"
                    f"{result['content']['raw']}"
                )
                completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=GROQ_CHAT_MODEL_NAME,
                    temperature=0.3
                )
                summary = completion.choices[0].message.content
            except:
                pass
                
        return f"""### 📊 File Summary

**File:** `{meta['filename']}`
**Type:** `{meta['type']}`
**Word Count:** {stats.get('word_count', 0)}
**Line Count:** {stats.get('line_count', 0)}

### Summary:
{summary}
"""

    def _handle_modify(self, file_path: str, output_path: str, instruction: str) -> str:
        result = self.processor.process_file(file_path)
        content = result["content"]["raw"]
        meta = result["metadata"]
        
        if not output_path:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_modified{ext}"
            
        # Use Groq to intelligently rewrite/modify content
        rewritten = content
        if get_groq_key():
            try:
                client = get_groq_client()
                prompt = (
                    f"Modify the following file content based on the user instruction.\n"
                    f"The file format is '{meta['extension']}'. Match the original formatting exactly.\n"
                    f"Do not add markdown formatting or conversation, just return raw file contents.\n\n"
                    f"Instruction: {instruction}\n\n"
                    f"--- Original Content ---\n"
                    f"{content}"
                )
                completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=GROQ_CHAT_MODEL_NAME,
                    temperature=0.2
                )
                rewritten = completion.choices[0].message.content.strip()
                # Clean potential markdown block formatting
                if rewritten.startswith("```"):
                    lines = rewritten.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    rewritten = "\n".join(lines).strip()
            except:
                pass
                
        write_file(output_path, rewritten)
        
        return f"""### ✏️ File Modified Successfully

**Original:** `{meta['filename']}`
**Output:** `{os.path.basename(output_path)}`
**Saved To:** `{output_path}`

### Preview of Changes:
```
{rewritten[:1000]}
```
{'... (truncated)' if len(rewritten) > 1000 else ''}
"""

    def _handle_compare(self, file_path: str, output_path: str) -> str:
        if not output_path:
            return "FileExpert Error: Please specify the second file path using the output_path parameter."
        result1 = self.processor.process_file(file_path)
        result2 = self.processor.process_file(output_path)
        
        content1 = result1["content"]["raw"]
        content2 = result2["content"]["raw"]
        
        lines1 = content1.split('\n')
        lines2 = content2.split('\n')
        common_lines = len(set(lines1) & set(lines2)) if lines1 and lines2 else 0
        
        return f"""### 🔍 File Comparison

**File 1:** `{result1['metadata']['filename']}`
**File 2:** `{result2['metadata']['filename']}`

- **Size Difference:** {result1['metadata']['size_bytes'] - result2['metadata']['size_bytes']} bytes
- **Line Count Difference:** {result1['metadata']['line_count'] - result2['metadata']['line_count']} lines
- **Common Lines:** {common_lines} lines
"""

    def _handle_generate(self, instruction: str, output_path: str) -> str:
        if not output_path:
            return "FileExpert Error: Please specify the target output file path using the output_path parameter."
        write_file(output_path, instruction)
        return f"""### 📝 File Generated Successfully

**File:** `{os.path.basename(output_path)}`
**Size:** {len(instruction)} bytes

### Preview:
```
{instruction[:500]}
```
"""

    def _handle_analyze(self, file_path: str) -> str:
        result = self.processor.process_file(file_path)
        meta = result["metadata"]
        analysis = result["analysis"]
        stats = result["statistics"]
        
        entities = analysis.get("entities", {})
        sentiment = analysis.get("sentiment", {})
        keywords = analysis.get("keywords", [])
        
        entity_section = ""
        if entities:
            entity_section = "\n### 📋 Extracted Entities\n"
            for k, v in entities.items():
                if v:
                    entity_section += f"- **{k.capitalize()}:** {', '.join(v[:10])}\n"
                    
        keyword_section = ""
        if keywords:
            keyword_section = "\n### 🔑 Key Keywords\n"
            keyword_section += ", ".join([f"**{kw['word']}** ({kw['count']})" for kw in keywords[:10]]) + "\n"
            
        return f"""### 🔬 Advanced File Analysis

**File:** `{meta['filename']}`
**Type:** `{meta['type']}`
**Readability:** `{analysis.get('readability', {}).get('level', 'unknown')}`
**Sentiment:** `{sentiment.get('label', 'neutral')} ({sentiment.get('score', 0):.1f}%)`

{entity_section}
{keyword_section}
### 📊 Statistics
- **Character Count:** {stats.get('character_count', 0)}
- **Word Count:** {stats.get('word_count', 0)}
- **Line Count:** {stats.get('line_count', 0)}
- **Complexity:** {stats.get('complexity', 0)*100:.1f}% unique words
"""
