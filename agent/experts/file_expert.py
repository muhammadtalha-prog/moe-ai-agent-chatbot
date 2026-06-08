import os
from typing import List, Dict, Any
from agent.experts.base_expert import BaseExpert
from agent.memory import VectorStore
from agent.config import get_groq_client, get_groq_key, GROQ_CHAT_MODEL_NAME
from agent.file_handler import read_file, write_file
from agent.security import get_safe_path

class FileExpert(BaseExpert):
    def respond(self, user_input: str, history: List[Dict[str, str]], memory: VectorStore, **kwargs) -> str:
        """
        Handles user commands relating to file manipulation.
        Kwargs can pass:
          - action: 'read', 'summarize', or 'rewrite'
          - file_path: Path of the source file
          - output_path: Optional path for output rewrite
          - instruction: Optional prompt instruction
        """
        action = kwargs.get("action")
        file_path = kwargs.get("file_path")
        output_path = kwargs.get("output_path")
        instruction = kwargs.get("instruction") or user_input
        
        if not file_path:
            return "FileExpert Error: No file path was specified."

        try:
            # 1. Read file
            content = read_file(file_path)
            
            # Save file snippet in vector store for future recall
            filename = os.path.basename(file_path)
            memory.add_text(
                text=f"Content of file {filename}:\n{content[:1500]}", 
                metadata={"source": "file_read", "filename": filename, "path": file_path}
            )
            
            if action == 'read':
                return f"Successfully read file `{filename}`. Content preview:\n\n{content}"
                
            elif action == 'summarize':
                summary = self.summarize_content(content, filename, instruction)
                # Save summary to vector memory
                memory.add_text(
                    text=f"Summary of file {filename}:\n{summary}",
                    metadata={"source": "file_summary", "filename": filename, "path": file_path}
                )
                return f"### Summary of `{filename}`:\n\n{summary}"
                
            elif action == 'rewrite':
                if not output_path:
                    # Default to overwriting or a suffix
                    base, ext = os.path.splitext(file_path)
                    output_path = f"{base}_rewritten{ext}"
                
                rewritten_content = self.rewrite_content(content, filename, instruction, file_path)
                write_file(output_path, rewritten_content)
                
                # Save rewritten to memory
                memory.add_text(
                    text=f"Rewritten version of file {filename} saved to {os.path.basename(output_path)}",
                    metadata={"source": "file_rewrite", "original_file": file_path, "new_file": output_path}
                )
                
                return f"Successfully processed and rewrote file `{filename}`.\nSaved to: `{output_path}`\n\nPreview of rewritten content:\n\n{rewritten_content[:1000]}"
            
            else:
                return f"FileExpert: Unknown action '{action}'."
                
        except Exception as e:
            return f"FileExpert Error: {str(e)}"

    def summarize_content(self, content: str, filename: str, instruction: str) -> str:
        """Summarizes the file content via Groq API."""
        if not get_groq_key():
            return f"[Offline Mock Mode] Summary of {filename} (first 100 chars): {content[:100]}..."
            
        prompt = (
            f"You are a summarizing expert. Summarize the following file content from the file named '{filename}'.\n"
            f"Instruction: {instruction}\n\n"
            f"--- File Content ---\n"
            f"{content}"
        )
        
        client = get_groq_client()
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_CHAT_MODEL_NAME,
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content

    def rewrite_content(self, content: str, filename: str, instruction: str, file_path: str) -> str:
        """Rewrites file content in its original format via Groq API."""
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        if not get_groq_key():
            return f"[Offline Mock Mode Rewritten] Original file: {filename}\nInstruction: {instruction}\nOriginal:\n{content}"
            
        prompt = (
            f"You are a file editing expert. Rewrite the following file content based on the instruction.\n"
            f"The original file is a '{ext}' format. You MUST write your response in the EXACT same format as the input.\n"
            f"Do not add conversational text, opening summaries, or markdown code blocks (like ```json or ```csv) in your reply, "
            f"only return the raw contents to be saved directly to the file.\n\n"
            f"Instruction: {instruction}\n\n"
            f"--- File Content ---\n"
            f"{content}"
        )
        
        client = get_groq_client()
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_CHAT_MODEL_NAME,
            temperature=0.2,
        )
        
        # Clean potential markdown fences from the response
        reply = chat_completion.choices[0].message.content.strip()
        if reply.startswith("```"):
            lines = reply.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            reply = "\n".join(lines).strip()
            
        return reply
