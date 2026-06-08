import sys
import os
from dotenv import load_dotenv

# Load variables before any imports that depend on config
load_dotenv()

from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.theme import Theme

# Add local path to sys.path to ensure absolute imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.config import get_groq_key, GROQ_CHAT_MODEL_NAME
from agent.experts.orchestrator import Orchestrator

# Setup beautiful Rich terminal theme
custom_theme = Theme({
    "info": "cyan bold",
    "warning": "yellow bold",
    "error": "red bold",
    "success": "green bold",
    "user": "magenta bold",
    "agent": "blue bold",
    "system": "grey50 italic",
})

console = Console(theme=custom_theme)

# Load Key Bindings for prompt_toolkit
kb = KeyBindings()

@kb.add('escape', 'enter')
def _(event):
    """Submit prompt on Escape + Enter."""
    event.current_buffer.validate_and_handle()

def print_help():
    help_text = """
[bold info]Available Interactive Commands:[/bold info]
  [info]/help[/info]           - Show this help menu.
  [info]/exit[/info] or [info]/quit[/info]   - Close the chatbot.
  [info]/clear[/info]          - Clear the screen and reset session history.
  [info]/history[/info]        - View current session chat history.
  [info]/memory[/info]         - View all stored facts in vector memory.
  [info]/reset-memory[/info]   - Clear the vector database completely.
  [info]/sys[/info]            - Check host system status (CPU, RAM, OS).

[bold info]File & Memory Shortcuts (handled automatically by AI or via commands):[/bold info]
  [info]/read <file>[/info]                       - Read a file content safely.
  [info]/summarize <file> [instruction][/info]   - Summarize a file.
  [info]/rewrite <file> <out_file> <inst>[/info]  - Rewrite a file with instruction.
  [info]/remember <fact>[/info]                    - Save a specific fact to vector memory.
  [info]/recall <query>[/info]                    - Retrieve matching facts from memory.
  [info]/search <query>[/info]                    - Run a quick DuckDuckGo web search.

[bold warning]Multiline Input Instructions:[/bold warning]
  - Type your input normally. You can press [bold yellow]Enter[/bold yellow] to start a new line.
  - When you are done, press [bold green]Alt+Enter[/bold green] (or [bold green]Esc[/bold green] followed by [bold green]Enter[/bold green]) to submit your input.
"""
    console.print(Panel(help_text.strip(), title="AI Agent Help & Syntax Guide", border_style="cyan"))

def main():
    # Print Premium Boot Header
    header_text = Text()
    header_text.append("=== AI AGENT CHATBOT v1.0 ===\n", style="success")
    header_text.append("Structure: MoE Routing | MCP Tools | Vector Memory\n", style="info")
    header_text.append(f"Model: Groq - {GROQ_CHAT_MODEL_NAME}\n", style="system")
    
    console.print(Panel(header_text, border_style="blue", expand=False))
    
    api_key = get_groq_key()
    if not api_key:
        warn_panel = Panel(
            "No [bold red]GROQ_API_KEY[/bold red] was found in the environment.\n"
            "The system is currently running in [bold yellow]Offline Mock Mode[/bold yellow]. AI responses will use fallback mocks.\n\n"
            "To enable full Groq AI capability, create a [bold green].env[/bold green] file in the project directory:\n"
            "[bold cyan]GROQ_API_KEY=your_real_api_key[/bold cyan]",
            title="Warning: API Key Missing",
            border_style="yellow"
        )
        console.print(warn_panel)
    else:
        console.print(f"[success]Groq API active using model '{GROQ_CHAT_MODEL_NAME}'.[/success]\n")

    # Initialize the Orchestrator
    orchestrator = Orchestrator(vector_store_path="vector_store.json")
    
    # Store session history
    session_history: List[Dict[str, str]] = []
    
    print_help()
    
    while True:
        try:
            # Multi-line input prompt using prompt_toolkit
            # Pressing Alt+Enter or Esc+Enter submits
            user_input = prompt(
                "\n[You] (Press Alt+Enter to submit):\n> ", 
                multiline=True,
                key_bindings=kb
            ).strip()
            
            if not user_input:
                continue
                
            # Intercept CLI specific actions
            if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
                console.print("[info]Exiting chatbot. Goodbye![/info]")
                break
                
            elif user_input.lower() == '/clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                session_history.clear()
                console.print("[success]Screen cleared and session history reset.[/success]")
                continue
                
            elif user_input.lower() == '/help':
                print_help()
                continue
                
            elif user_input.lower() == '/history':
                if not session_history:
                    console.print("[system]No chat history in this session yet.[/system]")
                else:
                    console.print("\n[bold info]--- Current Session History ---[/bold info]")
                    for msg in session_history:
                        role = "[bold magenta]User:[/bold magenta]" if msg['role'] == 'user' else "[bold blue]Agent:[/bold blue]"
                        console.print(f"{role}\n{msg['content']}")
                continue
                
            elif user_input.lower() == '/memory':
                docs = orchestrator.memory.documents
                if not docs:
                    console.print("[system]Vector database is empty.[/system]")
                else:
                    console.print(f"\n[bold info]--- Vector Memory ({len(docs)} items) ---[/bold info]")
                    for i, doc in enumerate(docs, 1):
                        source = doc['metadata'].get('source', 'unknown')
                        text_preview = doc['text'][:150].replace('\n', ' ')
                        console.print(f" {i}. [[info]{source}[/info]] {text_preview}...")
                continue
                
            elif user_input.lower() == '/reset-memory':
                orchestrator.memory.clear()
                console.print("[success]Vector database cleared successfully.[/success]")
                continue
            
            # Append user message to local history
            session_history.append({"role": "user", "content": user_input})
            
            # Process with visual spinner
            with console.status("[bold green]Agent processing query...[/bold green]", spinner="dots"):
                response = orchestrator.route_and_process(user_input, session_history)
            
            # Save response to history
            session_history.append({"role": "assistant", "content": response})
            
            # Print response
            console.print("\n[agent]Agent Response:[/agent]")
            # Render response with Markdown support
            console.print(Markdown(response))
            console.print("-" * 50)
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            console.print("\n[warning]Session interrupted by user (Ctrl+C).[/warning]")
            break
        except Exception as e:
            console.print(f"\n[error]An unexpected error occurred: {str(e)}[/error]")

if __name__ == "__main__":
    main()
