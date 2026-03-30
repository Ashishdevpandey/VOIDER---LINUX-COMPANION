#!/usr/bin/env python3
"""
AI OS CLI - Terminal Interface
Interactive command-line tool for the Local AI Assistant
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import httpx
from colorama import Fore, Style, init
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# Initialize colorama
init(autoreset=True)

# Initialize rich console
console = Console()

# Default API URL
DEFAULT_API_URL = "http://localhost:8000"

# Command completer
COMMANDS = [
    "help",
    "chat",
    "exec",
    "run",
    "index",
    "search",
    "history",
    "clear",
    "status",
    "config",
    "exit",
    "quit",
]


class AIOSClient:
    """CLI client for AI OS API"""
    
    def __init__(self, api_url: str = DEFAULT_API_URL):
        self.api_url = api_url.rstrip("/")
        self.session_id: Optional[str] = None
        self.http_client = httpx.Client(timeout=60.0)
        
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = self.http_client.get(f"{self.api_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def chat(self, message: str, execute: bool = False, use_rag: bool = False) -> dict:
        """Send chat message"""
        response = self.http_client.post(
            f"{self.api_url}/chat/simple",
            json={
                "message": message,
                "session_id": self.session_id,
                "execute_command": execute,
                "use_rag": use_rag,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        # Save session ID
        if "session_id" in data:
            self.session_id = data["session_id"]
        
        return data
    
    def execute_command(self, command: str) -> dict:
        """Execute command directly"""
        response = self.http_client.post(
            f"{self.api_url}/command/execute",
            json={"command": command},
        )
        response.raise_for_status()
        return response.json()
    
    def check_command(self, command: str) -> dict:
        """Check command safety"""
        response = self.http_client.post(
            f"{self.api_url}/command/check",
            json={"command": command},
        )
        response.raise_for_status()
        return response.json()
    
    def get_history(self, limit: int = 20) -> list:
        """Get command history"""
        response = self.http_client.get(
            f"{self.api_url}/command/history",
            params={"limit": limit},
        )
        response.raise_for_status()
        return response.json().get("commands", [])
    
    def clear_history(self) -> dict:
        """Clear command history"""
        response = self.http_client.delete(f"{self.api_url}/command/history")
        response.raise_for_status()
        return response.json()
    
    def index_files(self, directory: str, recursive: bool = True) -> dict:
        """Index files for RAG"""
        response = self.http_client.post(
            f"{self.api_url}/rag/index",
            json={"directory": directory, "recursive": recursive},
        )
        response.raise_for_status()
        return response.json()
    
    def search_rag(self, query: str, top_k: int = 5) -> list:
        """Search RAG index"""
        response = self.http_client.post(
            f"{self.api_url}/rag/search",
            json={"query": query, "top_k": top_k},
        )
        response.raise_for_status()
        return response.json().get("results", [])
    
    def get_rag_stats(self) -> dict:
        """Get RAG statistics"""
        response = self.http_client.get(f"{self.api_url}/rag/stats")
        response.raise_for_status()
        return response.json()
    
    def get_config(self) -> dict:
        """Get API configuration"""
        response = self.http_client.get(f"{self.api_url}/config")
        response.raise_for_status()
        return response.json()


def print_banner():
    """Print welcome banner"""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🤖 {Fore.WHITE}AI OS{Fore.CYAN} - Local AI Assistant for Linux                    ║
║                                                              ║
║   Chat · Execute Commands · Search Files (RAG)              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
    """
    print(banner)


def print_help():
    """Print help information"""
    help_text = """
[bold cyan]AI OS CLI Commands:[/bold cyan]

  [green]chat <message>[/green]     - Chat with AI
  [green]run <request>[/green]      - Generate and execute a command
  [green]exec <command>[/green]     - Execute a command directly
  [green]index <directory>[/green]  - Index files for RAG
  [green]search <query>[/green]     - Search indexed files
  [green]history[/green]            - Show command history
  [green]clear[/green]              - Clear command history
  [green]status[/green]             - Check API status
  [green]config[/green]             - Show configuration
  [green]help[/green]               - Show this help
  [green]exit/quit[/green]          - Exit CLI

[bold cyan]Shortcuts:[/bold cyan]
  Just type a message to chat
  Use [yellow]![/yellow] prefix to execute commands (e.g., !ls -la)
  Use [yellow]?[/yellow] prefix to search RAG (e.g., ?python code)
    """
    console.print(Markdown(help_text))


def format_response(data: dict) -> str:
    """Format API response for display"""
    message = data.get("message", "")
    
    # Check for command result
    command_result = data.get("command_result")
    if command_result:
        if command_result.get("blocked"):
            return f"{Fore.RED}⚠️  Command blocked: {command_result.get('block_reason')}{Style.RESET_ALL}"
        
        returncode = command_result.get("returncode", -1)
        stdout = command_result.get("stdout", "")
        stderr = command_result.get("stderr", "")
        
        output = []
        if stdout:
            output.append(stdout)
        if stderr:
            output.append(f"[stderr]: {stderr}")
        
        result_text = "\n".join(output) if output else "(no output)"
        
        if returncode == 0:
            return f"{Fore.GREEN}✅{Style.RESET_ALL}\n{result_text}"
        else:
            return f"{Fore.RED}❌ Exit code: {returncode}{Style.RESET_ALL}\n{result_text}"
    
    return message


def interactive_mode(client: AIOSClient):
    """Run interactive CLI mode"""
    print_banner()
    
    # Check API health
    if not client.health_check():
        print(f"{Fore.RED}⚠️  Warning: API not available at {client.api_url}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Make sure the server is running: uvicorn backend.main:app{Style.RESET_ALL}\n")
    else:
        print(f"{Fore.GREEN}✅ Connected to API at {client.api_url}{Style.RESET_ALL}\n")
    
    print_help()
    
    # Setup prompt session
    history_file = Path.home() / ".aios_history"
    session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=WordCompleter(COMMANDS),
        style=PTStyle.from_dict({
            "prompt": "ansicyan bold",
        }),
    )
    
    while True:
        try:
            # Get input
            user_input = session.prompt("\n🤖 AI OS > ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ("exit", "quit"):
                print(f"{Fore.CYAN}Goodbye! 👋{Style.RESET_ALL}")
                break
            
            elif user_input.lower() == "help":
                print_help()
                continue
            
            elif user_input.lower() == "status":
                if client.health_check():
                    print(f"{Fore.GREEN}✅ API is healthy{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ API is not responding{Style.RESET_ALL}")
                continue
            
            elif user_input.lower() == "config":
                try:
                    config = client.get_config()
                    table = Table(title="AI OS Configuration")
                    table.add_column("Setting", style="cyan")
                    table.add_column("Value", style="green")
                    for key, value in config.items():
                        table.add_row(key, str(value))
                    console.print(table)
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                continue
            
            elif user_input.lower() == "history":
                try:
                    history = client.get_history()
                    if history:
                        table = Table(title="Command History")
                        table.add_column("ID", style="dim")
                        table.add_column("Command", style="cyan")
                        table.add_column("Status", style="green")
                        for item in history[:10]:
                            status = "✅" if item["result"]["returncode"] == 0 else "❌"
                            if item["result"].get("blocked"):
                                status = "🚫"
                            table.add_row(
                                item["id"],
                                item["command"][:50],
                                status,
                            )
                        console.print(table)
                    else:
                        print(f"{Fore.YELLOW}No history available{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                continue
            
            elif user_input.lower() == "clear":
                try:
                    result = client.clear_history()
                    print(f"{Fore.GREEN}✅ {result.get('message', 'History cleared')}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                continue
            
            elif user_input.lower().startswith("index "):
                directory = user_input[6:].strip()
                try:
                    print(f"{Fore.CYAN}Indexing {directory}...{Style.RESET_ALL}")
                    result = client.index_files(directory)
                    if result.get("success"):
                        print(f"{Fore.GREEN}✅ Indexed {result.get('files_indexed')} files, {result.get('chunks_created')} chunks{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}❌ Indexing failed: {result.get('errors', ['Unknown error'])}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                continue
            
            elif user_input.lower().startswith("search "):
                query = user_input[7:].strip()
                try:
                    results = client.search_rag(query)
                    if results:
                        print(f"{Fore.CYAN}Found {len(results)} results:{Style.RESET_ALL}\n")
                        for i, result in enumerate(results, 1):
                            panel = Panel(
                                result.get("content", "")[:500],
                                title=f"[cyan]Result {i} - {result.get('source', 'Unknown')} (score: {result.get('score', 0):.2f})[/cyan]",
                                border_style="green",
                            )
                            console.print(panel)
                    else:
                        print(f"{Fore.YELLOW}No results found{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                continue
            
            elif user_input.lower().startswith("run "):
                request = user_input[4:].strip()
                try:
                    with console.status("[cyan]Generating command..."):
                        data = client.chat(request, execute=True)
                    print(format_response(data))
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                continue
            
            elif user_input.lower().startswith("exec "):
                command = user_input[5:].strip()
                try:
                    # Check safety first
                    safety = client.check_command(command)
                    if not safety.get("is_safe"):
                        print(f"{Fore.RED}🚫 Command blocked: {safety.get('reason')}{Style.RESET_ALL}")
                        continue
                    
                    if safety.get("requires_confirmation"):
                        confirm = input(f"{Fore.YELLOW}⚠️  {safety.get('risk_level')} risk: {safety.get('reason')}\nExecute anyway? (y/N): {Style.RESET_ALL}")
                        if confirm.lower() != "y":
                            print(f"{Fore.CYAN}Cancelled{Style.RESET_ALL}")
                            continue
                    
                    with console.status("[cyan]Executing..."):
                        data = client.execute_command(command)
                    
                    returncode = data.get("returncode", -1)
                    stdout = data.get("stdout", "")
                    stderr = data.get("stderr", "")
                    
                    if stdout:
                        console.print(Syntax(stdout, "text", theme="monokai"))
                    if stderr:
                        console.print(Syntax(stderr, "text", theme="monokai"), style="red")
                    
                    status_color = "green" if returncode == 0 else "red"
                    console.print(f"\n[{status_color}]Exit code: {returncode}[/{status_color}]")
                    
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                continue
            
            elif user_input.lower().startswith("chat "):
                message = user_input[5:].strip()
                try:
                    with console.status("[cyan]Thinking..."):
                        data = client.chat(message)
                    print(f"\n{Fore.GREEN}🤖 AI:{Style.RESET_ALL}")
                    console.print(Markdown(data.get("message", "")))
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                continue
            
            # Handle shortcuts
            elif user_input.startswith("!"):
                # Direct command execution
                command = user_input[1:].strip()
                try:
                    data = client.execute_command(command)
                    print(data.get("stdout", ""))
                    if data.get("stderr"):
                        print(f"{Fore.RED}{data['stderr']}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                continue
            
            elif user_input.startswith("?"):
                # RAG search
                query = user_input[1:].strip()
                try:
                    with console.status("[cyan]Searching..."):
                        data = client.chat(query, use_rag=True)
                    print(f"\n{Fore.GREEN}🤖 AI:{Style.RESET_ALL}")
                    console.print(Markdown(data.get("message", "")))
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                continue
            
            # Default: chat mode
            else:
                try:
                    with console.status("[cyan]Thinking..."):
                        data = client.chat(user_input)
                    print(f"\n{Fore.GREEN}🤖 AI:{Style.RESET_ALL}")
                    console.print(Markdown(data.get("message", "")))
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        
        except KeyboardInterrupt:
            print(f"\n{Fore.CYAN}Use 'exit' to quit{Style.RESET_ALL}")
            continue
        except EOFError:
            break


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI OS CLI - Local AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aios --chat                    Start interactive chat mode
  aios "show disk usage" --run   Execute a command
  aios "python code" --rag       Search files using RAG
  aios --index ./docs            Index documents
  aios --history                 Show command history
        """,
    )
    
    parser.add_argument(
        "message",
        nargs="?",
        help="Message or command to send",
    )
    parser.add_argument(
        "--chat", "-c",
        action="store_true",
        help="Start interactive chat mode",
    )
    parser.add_argument(
        "--run", "-r",
        action="store_true",
        help="Generate and execute command from message",
    )
    parser.add_argument(
        "--exec", "-e",
        action="store_true",
        help="Execute command directly",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Use RAG to search files",
    )
    parser.add_argument(
        "--index",
        metavar="DIRECTORY",
        help="Index directory for RAG",
    )
    parser.add_argument(
        "--search",
        metavar="QUERY",
        help="Search indexed files",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show command history",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear command history",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check API status",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API URL (default: {DEFAULT_API_URL})",
    )
    
    args = parser.parse_args()
    
    # Create client
    client = AIOSClient(api_url=args.api_url)
    
    # Handle flags
    if args.chat:
        interactive_mode(client)
        return
    
    if args.status:
        if client.health_check():
            print(f"{Fore.GREEN}✅ API is healthy at {args.api_url}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ API is not responding at {args.api_url}{Style.RESET_ALL}")
        return
    
    if args.history:
        try:
            history = client.get_history()
            if history:
                for item in history[:20]:
                    status = "✅" if item["result"]["returncode"] == 0 else "❌"
                    print(f"{status} [{item['id']}] {item['command'][:60]}")
            else:
                print("No history available")
        except Exception as e:
            print(f"Error: {e}")
        return
    
    if args.clear:
        try:
            result = client.clear_history()
            print(result.get("message", "History cleared"))
        except Exception as e:
            print(f"Error: {e}")
        return
    
    if args.index:
        try:
            print(f"Indexing {args.index}...")
            result = client.index_files(args.index)
            if result.get("success"):
                print(f"✅ Indexed {result.get('files_indexed')} files, {result.get('chunks_created')} chunks")
            else:
                print(f"❌ Failed: {result.get('errors', ['Unknown error'])}")
        except Exception as e:
            print(f"Error: {e}")
        return
    
    if args.search:
        try:
            results = client.search_rag(args.search)
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n--- Result {i} ({result.get('source')}) ---")
                    print(result.get("content", "")[:500])
            else:
                print("No results found")
        except Exception as e:
            print(f"Error: {e}")
        return
    
    if args.message:
        try:
            if args.exec:
                # Direct execution
                data = client.execute_command(args.message)
                print(data.get("stdout", ""))
                if data.get("stderr"):
                    print(f"stderr: {data['stderr']}")
            elif args.run:
                # Generate and execute
                data = client.chat(args.message, execute=True)
                print(format_response(data))
            elif args.rag:
                # RAG search
                data = client.chat(args.message, use_rag=True)
                print(data.get("message", ""))
            else:
                # Simple chat
                data = client.chat(args.message)
                print(data.get("message", ""))
        except Exception as e:
            print(f"Error: {e}")
        return
    
    # Default: interactive mode
    interactive_mode(client)


if __name__ == "__main__":
    main()
