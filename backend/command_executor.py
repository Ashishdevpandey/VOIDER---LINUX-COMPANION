"""
Command Executor for AI OS
Handles safe execution of Linux commands with timeouts and output capture
"""

import asyncio
import logging
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime

from backend.models import CommandResult, RiskLevel
from backend.safety import get_safety_checker, SafetyChecker

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for command execution"""
    timeout: int = 30
    shell: bool = True
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    capture_output: bool = True


class CommandExecutor:
    """Safe command executor with history tracking"""
    
    def __init__(
        self,
        safety_checker: Optional[SafetyChecker] = None,
        default_timeout: int = 30,
        max_history: int = 100,
    ):
        """
        Initialize command executor
        
        Args:
            safety_checker: Safety checker instance
            default_timeout: Default command timeout in seconds
            max_history: Maximum history items to keep
        """
        self.safety_checker = safety_checker or get_safety_checker()
        self.default_timeout = default_timeout
        self.max_history = max_history
        
        # Command history
        self._history: Dict[str, CommandResult] = {}
        self._history_order: List[str] = []
        
        logger.info("Command executor initialized")
    
    async def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        user_input: Optional[str] = None,
        skip_safety_check: bool = False,
    ) -> CommandResult:
        """
        Execute a command safely
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds (overrides default)
            user_input: Original user input that generated this command
            skip_safety_check: Skip safety check (for internal use)
            
        Returns:
            CommandResult with execution details
        """
        start_time = time.time()
        
        # Sanitize command
        command = self.safety_checker.sanitize_command(command)
        
        if not command:
            return CommandResult(
                command="",
                stdout="",
                stderr="Empty command",
                returncode=-1,
                duration_ms=0,
                blocked=True,
                block_reason="Empty command",
            )
        
        # Safety check
        if not skip_safety_check:
            safety_result = self.safety_checker.check_command(command)
            
            if not safety_result.is_safe:
                duration_ms = (time.time() - start_time) * 1000
                result = CommandResult(
                    command=command,
                    stdout="",
                    stderr=f"Command blocked: {safety_result.reason}",
                    returncode=-1,
                    duration_ms=duration_ms,
                    risk_level=safety_result.risk_level,
                    blocked=True,
                    block_reason=safety_result.reason,
                )
                self._add_to_history(result, user_input)
                return result
        
        # Determine risk level
        risk_level = self.safety_checker.get_risk_level(command)
        
        # Execute command
        try:
            result = await self._run_command(
                command=command,
                timeout=timeout or self.default_timeout,
            )
            result.risk_level = risk_level
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            result = CommandResult(
                command=command,
                stdout="",
                stderr=f"Command timed out after {timeout or self.default_timeout} seconds",
                returncode=-1,
                duration_ms=duration_ms,
                risk_level=risk_level,
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = CommandResult(
                command=command,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                returncode=-1,
                duration_ms=duration_ms,
                risk_level=risk_level,
            )
        
        # Add to history
        self._add_to_history(result, user_input)
        
        return result
    
    async def _run_command(
        self,
        command: str,
        timeout: int,
    ) -> CommandResult:
        """
        Run a command using subprocess
        
        Args:
            command: Command to run
            timeout: Timeout in seconds
            
        Returns:
            CommandResult
        """
        start_time = time.time()
        
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Decode output
            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
            
            return CommandResult(
                command=command,
                stdout=stdout_str,
                stderr=stderr_str,
                returncode=process.returncode or 0,
                duration_ms=duration_ms,
            )
            
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return CommandResult(
                command=command,
                stdout="",
                stderr=str(e),
                returncode=-1,
                duration_ms=duration_ms,
            )
    
    def execute_sync(
        self,
        command: str,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        """
        Execute command synchronously
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            
        Returns:
            CommandResult
        """
        return asyncio.run(self.execute(command, timeout))
    
    def _add_to_history(
        self,
        result: CommandResult,
        user_input: Optional[str] = None,
    ) -> None:
        """
        Add result to history
        
        Args:
            result: Command result to add
            user_input: Original user input
        """
        command_id = str(uuid.uuid4())[:8]
        self._history[command_id] = result
        self._history_order.append(command_id)
        
        # Trim history if needed
        while len(self._history_order) > self.max_history:
            old_id = self._history_order.pop(0)
            del self._history[old_id]
        
        # Log execution
        status = "BLOCKED" if result.blocked else ("SUCCESS" if result.returncode == 0 else "FAILED")
        logger.info(
            f"Command {status}: {result.command[:50]}... "
            f"(exit: {result.returncode}, duration: {result.duration_ms:.0f}ms)"
        )
    
    def get_history(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Get command history
        
        Args:
            limit: Maximum items to return
            offset: Offset for pagination
            
        Returns:
            List of history items
        """
        items = []
        for cmd_id in reversed(self._history_order):
            result = self._history[cmd_id]
            items.append({
                "id": cmd_id,
                "command": result.command,
                "returncode": result.returncode,
                "executed_at": result.executed_at.isoformat(),
                "duration_ms": result.duration_ms,
                "risk_level": result.risk_level.value,
                "blocked": result.blocked,
            })
        
        return items[offset:offset + limit]
    
    def get_history_item(self, command_id: str) -> Optional[CommandResult]:
        """
        Get specific history item
        
        Args:
            command_id: Command ID
            
        Returns:
            CommandResult if found
        """
        return self._history.get(command_id)
    
    def clear_history(self) -> int:
        """
        Clear command history
        
        Returns:
            Number of items cleared
        """
        count = len(self._history)
        self._history.clear()
        self._history_order.clear()
        logger.info(f"History cleared ({count} items)")
        return count
    
    def get_stats(self) -> Dict:
        """
        Get execution statistics
        
        Returns:
            Statistics dictionary
        """
        if not self._history:
            return {
                "total_commands": 0,
                "successful": 0,
                "failed": 0,
                "blocked": 0,
                "avg_duration_ms": 0,
            }
        
        total = len(self._history)
        successful = sum(1 for r in self._history.values() if r.returncode == 0 and not r.blocked)
        failed = sum(1 for r in self._history.values() if r.returncode != 0 and not r.blocked)
        blocked = sum(1 for r in self._history.values() if r.blocked)
        avg_duration = sum(r.duration_ms for r in self._history.values()) / total
        
        return {
            "total_commands": total,
            "successful": successful,
            "failed": failed,
            "blocked": blocked,
            "avg_duration_ms": round(avg_duration, 2),
        }


# Global executor instance
_executor: Optional[CommandExecutor] = None


def get_command_executor(
    safety_checker: Optional[SafetyChecker] = None,
    default_timeout: int = 30,
) -> CommandExecutor:
    """
    Get or create global command executor
    
    Args:
        safety_checker: Safety checker instance
        default_timeout: Default timeout
        
    Returns:
        CommandExecutor instance
    """
    global _executor
    if _executor is None:
        _executor = CommandExecutor(
            safety_checker=safety_checker,
            default_timeout=default_timeout,
        )
    return _executor
