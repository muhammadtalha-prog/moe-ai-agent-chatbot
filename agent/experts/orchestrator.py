import json
import re
import os
from typing import List, Dict, Any, Optional
from agent.config import get_groq_client, get_groq_key, GROQ_CHAT_MODEL_NAME
from agent.memory import VectorMemory
from agent.mcp import mcp_registry


class Orchestrator:
    def __init__(self, vector_store_path: str = "vector_store.json") -> None:
        self.vector_store_path: str = vector_store_path
        self._memory: Optional[VectorMemory] = None
        self._chat_expert: Optional[Any] = None
        self._file_expert: Optional[Any] = None
        self._memory_expert: Optional[Any] = None

    @property
    def memory(self) -> VectorMemory:
        if self._memory is None:
            self._memory = VectorMemory(self.vector_store_path)
        return self._memory

    @property
    def chat_expert(self) -> Any:
        if self._chat_expert is None:
            from agent.experts.chat_expert import ChatExpert
            self._chat_expert = ChatExpert()
        return self._chat_expert

    @property
    def file_expert(self) -> Any:
        if self._file_expert is None:
            from agent.experts.file_expert import FileExpert
            self._file_expert = FileExpert()
        return self._file_expert

    @property
    def memory_expert(self) -> Any:
        if self._memory_expert is None:
            from agent.experts.memory_expert import MemoryExpert
            self._memory_expert = MemoryExpert()
        return self._memory_expert

    def route_and_process(self, user_input: str, history: List[Dict[str, str]]) -> str:
        """
        Main entry point for routing a user query.
        1. Query Vector Memory to find relevant context.
        2. Determine the best expert (MoE routing).
        3. Execute the expert and return response.
        4. Save user input and response to history/memory.
        """
        # Save user query to vector memory if it looks informative
        if len(user_input.strip()) > 10 and not user_input.startswith('/'):
            # Fetch relevant memories to enrich context
            relevant_memories = self.memory.query(user_input, top_k=2)
            mem_context = ""
            if relevant_memories:
                mem_context = "\n[Retrieved Memory Context]:\n" + "\n".join(
                    [f"- {doc['text']}" for doc, score in relevant_memories if score > 0.6]
                )
        else:
            mem_context = ""

        # Step 2: Route intent
        routing = self.determine_route(user_input)
        expert_name = routing.get("expert", "chat")
        
        # Add retrieved memory context if chat expert is chosen
        refined_input = user_input
        if expert_name == "chat" and mem_context:
            refined_input = f"{user_input}\n{mem_context}"
            
        response = ""
        
        # Step 3: Execute Expert
        if expert_name == "file":
            response = self.file_expert.respond(
                refined_input, 
                history, 
                self.memory, 
                action=routing.get("action"),
                file_path=routing.get("file_path"),
                output_path=routing.get("output_path"),
                instruction=routing.get("instruction")
            )
            
        elif expert_name == "memory":
            response = self.memory_expert.respond(
                refined_input,
                history,
                self.memory,
                action=routing.get("action"),
                text_to_remember=routing.get("text_to_remember"),
                query_text=routing.get("query_text")
            )
            
        elif expert_name == "mcp":
            tool_name = routing.get("tool_name")
            tool_args = routing.get("tool_arguments", {})
            
            tool_output = mcp_registry.execute_tool(tool_name, tool_args)
            
            # Feed tool output back to the ChatExpert to generate a natural response
            if get_groq_key():
                followup_prompt = (
                    f"The user query: '{user_input}' triggered the tool '{tool_name}' with arguments {tool_args}.\n"
                    f"The tool returned the following output:\n\n{tool_output}\n\n"
                    f"Please summarize or formulate a helpful final response to the user based on this data."
                )
                response = self.chat_expert.respond(followup_prompt, history, self.memory)
            else:
                response = f"**[Executed Tool: {tool_name}]**\n\n{tool_output}"
                
        else: # chat
            response = self.chat_expert.respond(refined_input, history, self.memory)
            
        # Add conversation turn to vector store for retrieval later
        if not user_input.startswith('/') and len(user_input.strip()) > 5:
            # Only save response if it does not contain error signatures
            resp_lower = response.lower()
            if "error code:" not in resp_lower and "chatexpert error" not in resp_lower and "invalid_api_key" not in resp_lower and "fileexpert error" not in resp_lower:
                self.memory.add_text(
                    text=f"User asked: {user_input}\nAgent answered: {response[:300]}...",
                    metadata={"source": "history", "input": user_input[:100]}
                )
            
        return response

    def determine_route(self, user_input: str) -> Dict[str, Any]:
        """
        Uses heuristics first (slash commands), and falls back to Groq API semantic routing.
        """
        input_stripped = user_input.strip()
        
        # 1. Heuristics & Commands
        if input_stripped.startswith('/'):
            parts = input_stripped.split(maxsplit=2)
            cmd = parts[0].lower()
            
            if cmd == '/read' and len(parts) > 1:
                return {"expert": "file", "action": "read", "file_path": parts[1]}
            elif cmd == '/summarize' and len(parts) > 1:
                instruction = parts[2] if len(parts) > 2 else "Summarize the file content"
                return {"expert": "file", "action": "summarize", "file_path": parts[1], "instruction": instruction}
            elif cmd == '/rewrite' and len(parts) > 2:
                # Format: /rewrite file_path output_path instruction OR /rewrite file_path instruction
                subparts = parts[1].split(maxsplit=1)
                file_path = subparts[0]
                rest = parts[2]
                
                # Check if output path is provided
                match = re.match(r'^(\S+)\s+(.+)$', rest)
                if match:
                    out_path = match.group(1)
                    inst = match.group(2)
                else:
                    out_path = None
                    inst = rest
                return {"expert": "file", "action": "rewrite", "file_path": file_path, "output_path": out_path, "instruction": inst}
            elif cmd == '/remember' and len(parts) > 1:
                text_to_remember = input_stripped.split(maxsplit=1)[1]
                return {"expert": "memory", "action": "remember", "text_to_remember": text_to_remember}
            elif cmd == '/recall' and len(parts) > 1:
                query_text = input_stripped.split(maxsplit=1)[1]
                return {"expert": "memory", "action": "recall", "query_text": query_text}
            elif cmd == '/search' and len(parts) > 1:
                search_query = input_stripped.split(maxsplit=1)[1]
                return {"expert": "mcp", "tool_name": "web_search", "tool_arguments": {"query": search_query}}
            elif cmd == '/sys':
                return {"expert": "mcp", "tool_name": "get_system_info", "tool_arguments": {}}
                
        # Heuristics based on keyword searches in query
        # File operations
        file_patterns = [
            r"(?:read|view|open|show|cat)\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)",
            r"(?:summarize|summary|sum\s+up)\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)"
        ]
        
        for pattern in file_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                file_path = match.group(1)
                action = "summarize" if "summarize" in user_input.lower() or "summary" in user_input.lower() else "read"
                return {"expert": "file", "action": action, "file_path": file_path}

        # Rewrite operations
        rewrite_match = re.search(r"(?:rewrite|modify|edit)\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)(?:\s+(?:and\s+save\s+to|to)\s+([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+))?", user_input, re.IGNORECASE)
        if rewrite_match:
            file_path = rewrite_match.group(1)
            output_path = rewrite_match.group(2)
            # Find instructions: everything after rewrite statement
            instruction = user_input
            return {"expert": "file", "action": "rewrite", "file_path": file_path, "output_path": output_path, "instruction": instruction}

        # Tool fallback queries (e.g. search web, check CPU)
        if any(w in user_input.lower() for w in ["search the web for", "google for", "look up online"]):
            q = re.sub(r"(search the web for|google for|look up online)", "", user_input, flags=re.IGNORECASE).strip()
            return {"expert": "mcp", "tool_name": "web_search", "tool_arguments": {"query": q}}
        
        if any(w in user_input.lower() for w in ["system usage", "cpu usage", "ram usage", "system info", "host statistics"]):
            return {"expert": "mcp", "tool_name": "get_system_info", "tool_arguments": {}}

        # 2. Semantic LLM Routing (Fallback if Groq API Key is present)
        if get_groq_key():
            try:
                tools_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in mcp_registry.get_tool_definitions()])
                
                router_prompt = (
                    "You are a routing orchestrator for a Mixture of Experts (MoE) system.\n"
                    "Determine which expert should handle the user input.\n\n"
                    "Available Experts:\n"
                    "- 'file': Used when reading, summarizing, or rewriting specific files.\n"
                    "- 'memory': Storing facts explicitly or recalling information from user memory.\n"
                    "- 'mcp': Using external tools. Available tools are:\n"
                    f"{tools_desc}\n"
                    "- 'chat': General conversation, writing code, answering questions, or when no other expert fits.\n\n"
                    "Output a raw JSON object with the following schema (and nothing else):\n"
                    "{\n"
                    '  "expert": "file" | "memory" | "mcp" | "chat",\n'
                    '  "action": "read" | "summarize" | "rewrite" | "remember" | "recall" | "run_tool" (optional),\n'
                    '  "file_path": "path/to/file" (only for file operations),\n'
                    '  "output_path": "path/to/output_file" (only for file rewrite, optional),\n'
                    '  "instruction": "instructions on how to rewrite or summarize the file" (only for file operations),\n'
                    '  "text_to_remember": "fact to store in vector store" (only for memory remember),\n'
                    '  "query_text": "search query" (only for memory recall),\n'
                    '  "tool_name": "name of mcp tool to run" (only for mcp expert),\n'
                    '  "tool_arguments": {"arg_name": "arg_value"} (only for mcp expert)\n'
                    "}\n"
                )
                
                client = get_groq_client()
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a routing system. You must output valid, pure JSON only."
                        },
                        {
                            "role": "user",
                            "content": f"{router_prompt}\nUser Input: '{user_input}'"
                        }
                    ],
                    model=GROQ_CHAT_MODEL_NAME,
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                
                clean_response = chat_completion.choices[0].message.content.strip()
                route_data = json.loads(clean_response)
                return route_data
            except Exception as e:
                # Silently fallback to chat if routing fails
                pass
                
        # Default fallback
        return {"expert": "chat"}
