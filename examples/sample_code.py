#!/usr/bin/env python3
"""
Sample Python code for RAG testing.
This demonstrates the command executor functionality.
"""

import subprocess
from typing import Tuple, Optional

class SafeCommandExecutor:
    """Safely execute Linux commands with timeouts and validation."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.history = []
    
    def execute(self, command: str) -> Tuple[str, str, int]:
        """
        Execute a command safely.
        
        Args:
            command: The command to execute
            
        Returns:
            Tuple of (stdout, stderr, returncode)
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            self.history.append({
                'command': command,
                'returncode': result.returncode
            })
            
            return result.stdout, result.stderr, result.returncode
            
        except subprocess.TimeoutExpired:
            return '', f'Command timed out after {self.timeout} seconds', -1
        except Exception as e:
            return '', str(e), -1

if __name__ == '__main__':
    executor = SafeCommandExecutor()
    stdout, stderr, code = executor.execute('ls -la')
    print(f'Exit code: {code}')
    print(f'Output: {stdout}')
