"""
Safety module for AI OS
Handles command validation, risk assessment, and safety checks
"""

import logging
import re
import shlex
from typing import List, Optional, Tuple

from backend.models import RiskLevel, SafetyCheckResponse

logger = logging.getLogger(__name__)


# Default dangerous patterns that should always be blocked
DEFAULT_BLOCKED_PATTERNS = [
    # Filesystem destruction
    r"rm\s+-rf\s+/\s*$",
    r"rm\s+--no-preserve-root",
    r":\(\)\{\s*:\|:&\s*\};:",  # Fork bomb
    r"mkfs\.\w+",
    r"dd\s+if=.*of=/dev/[sh]d\w",
    r">\s*/dev/[sh]d\w+",
    r">\s*/dev/null\s+2>&1\s*&&\s*rm\s+-rf",
    
    # System shutdown
    r"^\s*shutdown",
    r"^\s*reboot",
    r"^\s*halt",
    r"^\s*poweroff",
    r"init\s+0",
    r"init\s+6",
    r"systemctl\s+(poweroff|reboot|halt|suspend|hibernate)",
    
    # Dangerous chmod/chown
    r"chmod\s+-R\s+777\s+/",
    r"chmod\s+777\s+/\s*$",
    r"chown\s+-R\s+\w+:\w+\s+/\s*$",
    
    # Dangerous moves
    r"mv\s+/\s+\S+",
    r"mv\s+/\w+\s+/\w+\s*/\s*$",
    
    # Remote code execution
    r"curl\s+.*\|\s*(sh|bash|zsh)",
    r"curl\s+.*\|\s*\w+\s*-\s*\|\s*(sh|bash|zsh)",
    r"wget\s+.*\|\s*(sh|bash|zsh)",
    r"wget\s+-q\s+-O-\s+.*\|\s*(sh|bash|zsh)",
    
    # Container escape attempts
    r"docker\s+run.*--privileged",
    r"docker\s+run.*-v\s+/:\s*/",
    r"docker\s+run.*--pid=host",
    r"docker\s+run.*--network=host",
    
    # Kernel/module manipulation
    r"insmod",
    r"rmmod",
    r"modprobe\s+-r",
    r"sysctl\s+-w",
]

# Patterns requiring confirmation (medium/high risk)
DEFAULT_CONFIRM_PATTERNS = [
    r"\bsudo\b",
    r"\bsu\s+-\s*\w+",
    r"rm\s+-rf\s+\S+",
    r"rm\s+-r\s+\S+",
    r"rm\s+-f\s+\S+",
    r"chmod\s+777\s+\S+",
    r"chmod\s+-R\s+\S+",
    r"chown\s+-R\s+\S+",
    r"kill\s+-9\s+\d+",
    r"pkill\s+-9",
    r"killall\s+-9",
    r"docker\s+system\s+prune",
    r"docker\s+volume\s+rm",
    r"docker\s+rm\s+-f",
    r"docker\s+rmi\s+-f",
    r"pip\s+uninstall",
    r"pip3\s+uninstall",
    r"apt\s+remove",
    r"apt\s+purge",
    r"apt-get\s+remove",
    r"apt-get\s+purge",
    r"yum\s+remove",
    r"yum\s+erase",
    r"dnf\s+remove",
    r"pacman\s+-R",
    r"pacman\s+-Rs",
    r"npm\s+uninstall\s+-g",
    r"yarn\s+global\s+remove",
]

# Safe commands (low risk)
DEFAULT_SAFE_COMMANDS = {
    "ls", "ll", "la", "pwd", "cd", "cat", "head", "tail", "less", "more",
    "grep", "find", "which", "whereis", "df", "du", "free", "top", "htop",
    "ps", "uptime", "whoami", "id", "uname", "hostname", "date", "cal",
    "echo", "printf", "man", "help", "clear", "history", "env", "printenv",
    "alias", "type", "file", "stat", "wc", "sort", "uniq", "cut", "awk",
    "sed", "tr", "tee", "xargs", "tar", "zip", "unzip", "gzip", "gunzip",
    "ping", "curl", "wget", "netstat", "ss", "lsof", "fuser", "pgrep",
    "pkill", "killall", "jobs", "fg", "bg", "disown", "nohup", "screen",
    "tmux", "git", "docker", "docker-compose", "kubectl", "helm", "python",
    "python3", "pip", "pip3", "node", "npm", "yarn", "go", "rustc", "cargo",
    "make", "cmake", "gcc", "g++", "javac", "java", "mvn", "gradle",
}


class SafetyChecker:
    """Command safety checker"""
    
    def __init__(
        self,
        blocked_patterns: Optional[List[str]] = None,
        confirm_patterns: Optional[List[str]] = None,
        safe_commands: Optional[set] = None,
    ):
        """
        Initialize safety checker
        
        Args:
            blocked_patterns: Regex patterns for blocked commands
            confirm_patterns: Regex patterns requiring confirmation
            safe_commands: Set of safe command names
        """
        self.blocked_patterns = blocked_patterns or DEFAULT_BLOCKED_PATTERNS
        self.confirm_patterns = confirm_patterns or DEFAULT_CONFIRM_PATTERNS
        self.safe_commands = safe_commands or DEFAULT_SAFE_COMMANDS
        
        # Compile regex patterns
        self._blocked_regex = [re.compile(p, re.IGNORECASE) for p in self.blocked_patterns]
        self._confirm_regex = [re.compile(p, re.IGNORECASE) for p in self.confirm_patterns]
        
        logger.info("Safety checker initialized")
    
    def is_safe_command(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a command is safe to execute
        
        Args:
            command: Command string to check
            
        Returns:
            Tuple of (is_safe, reason_if_blocked)
        """
        if not command or not command.strip():
            return False, "Empty command"
        
        command = command.strip()
        
        # Check blocked patterns
        for pattern in self._blocked_regex:
            if pattern.search(command):
                matched = pattern.search(command).group(0)
                reason = f"Command matches dangerous pattern: {matched}"
                logger.warning(f"Blocked command: {command[:50]}... - {reason}")
                return False, reason
        
        return True, None
    
    def get_risk_level(self, command: str) -> RiskLevel:
        """
        Get risk level for a command
        
        Args:
            command: Command string to check
            
        Returns:
            RiskLevel enum value
        """
        if not command or not command.strip():
            return RiskLevel.LOW
        
        command = command.strip()
        
        # Check if blocked (CRITICAL)
        is_safe, _ = self.is_safe_command(command)
        if not is_safe:
            return RiskLevel.CRITICAL
        
        # Check patterns requiring confirmation
        for pattern in self._confirm_regex:
            if pattern.search(command):
                # Check for high-risk patterns
                high_risk_patterns = [
                    r"rm\s+-rf\s+/",
                    r"sudo\s+rm",
                    r"docker\s+system\s+prune",
                ]
                for hr_pattern in high_risk_patterns:
                    if re.search(hr_pattern, command, re.IGNORECASE):
                        return RiskLevel.HIGH
                return RiskLevel.MEDIUM
        
        # Check if base command is in safe list
        try:
            base_cmd = shlex.split(command)[0]
            if base_cmd in self.safe_commands:
                return RiskLevel.LOW
        except ValueError:
            pass
        
        # Unknown commands are medium risk
        return RiskLevel.MEDIUM
    
    def requires_confirmation(self, command: str) -> bool:
        """
        Check if command requires user confirmation
        
        Args:
            command: Command string to check
            
        Returns:
            True if confirmation is required
        """
        risk_level = self.get_risk_level(command)
        return risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
    
    def check_command(self, command: str) -> SafetyCheckResponse:
        """
        Full safety check for a command
        
        Args:
            command: Command string to check
            
        Returns:
            SafetyCheckResponse with full details
        """
        is_safe, block_reason = self.is_safe_command(command)
        risk_level = self.get_risk_level(command)
        requires_confirm = self.requires_confirmation(command)
        
        # Build reason string
        if not is_safe:
            reason = block_reason
        elif risk_level == RiskLevel.MEDIUM:
            reason = "Command may modify system state"
        elif risk_level == RiskLevel.HIGH:
            reason = "Command can cause significant changes or data loss"
        elif risk_level == RiskLevel.CRITICAL:
            reason = "Command is critically dangerous"
        else:
            reason = None
        
        return SafetyCheckResponse(
            command=command,
            is_safe=is_safe,
            risk_level=risk_level,
            requires_confirmation=requires_confirm,
            reason=reason,
        )
    
    def sanitize_command(self, command: str) -> str:
        """
        Sanitize a command string
        
        Args:
            command: Command to sanitize
            
        Returns:
            Sanitized command
        """
        # Remove null bytes
        command = command.replace("\x00", "")
        
        # Remove control characters except newlines and tabs
        command = "".join(c for c in command if c == "\n" or c == "\t" or ord(c) >= 32)
        
        # Strip whitespace
        command = command.strip()
        
        return command
    
    def validate_command_chain(self, commands: str) -> List[SafetyCheckResponse]:
        """
        Validate a chain of commands (semicolon or pipe separated)
        
        Args:
            commands: Command chain string
            
        Returns:
            List of safety check responses
        """
        results = []
        
        # Split by common separators
        separators = r"[;|&]"
        individual_commands = re.split(separators, commands)
        
        for cmd in individual_commands:
            cmd = cmd.strip()
            if cmd:
                results.append(self.check_command(cmd))
        
        return results


# Global safety checker instance
_safety_checker: Optional[SafetyChecker] = None


def get_safety_checker() -> SafetyChecker:
    """Get or create global safety checker instance"""
    global _safety_checker
    if _safety_checker is None:
        _safety_checker = SafetyChecker()
    return _safety_checker


def is_safe_command(command: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a command is safe (convenience function)
    
    Args:
        command: Command string to check
        
    Returns:
        Tuple of (is_safe, reason_if_blocked)
    """
    return get_safety_checker().is_safe_command(command)


def get_command_risk_level(command: str) -> RiskLevel:
    """
    Get risk level for a command (convenience function)
    
    Args:
        command: Command string to check
        
    Returns:
        RiskLevel enum value
    """
    return get_safety_checker().get_risk_level(command)


def require_confirmation(command: str) -> bool:
    """
    Check if command requires confirmation (convenience function)
    
    Args:
        command: Command string to check
        
    Returns:
        True if confirmation is required
    """
    return get_safety_checker().requires_confirmation(command)
