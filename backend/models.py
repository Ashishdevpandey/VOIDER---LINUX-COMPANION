"""
Pydantic models for AI OS API
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    """Command risk levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MessageRole(str, Enum):
    """Message roles for chat"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Single chat message"""
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, description="User message")
    session_id: Optional[str] = Field(default=None, description="Session ID for context")
    execute_command: bool = Field(default=False, description="Whether to execute generated commands")
    use_rag: bool = Field(default=False, description="Whether to use RAG for context")
    context: Optional[List[ChatMessage]] = Field(default=None, description="Previous messages")

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        """Validate message is not empty"""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class CommandResult(BaseModel):
    """Command execution result"""
    command: str = Field(..., description="Executed command")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    returncode: int = Field(..., description="Exit code")
    executed_at: datetime = Field(default_factory=datetime.now)
    duration_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    blocked: bool = Field(default=False, description="Whether command was blocked")
    block_reason: Optional[str] = Field(default=None, description="Reason for blocking")


class ChatResponse(BaseModel):
    """Chat response model"""
    message: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session ID")
    command_result: Optional[CommandResult] = Field(default=None, description="Command execution result")
    rag_context: Optional[List[str]] = Field(default=None, description="RAG context used")
    timestamp: datetime = Field(default_factory=datetime.now)
    model: str = Field(default="llama3.2", description="Model used")
    tokens_used: Optional[int] = Field(default=None, description="Tokens used")


class IndexRequest(BaseModel):
    """File indexing request"""
    directory: str = Field(..., description="Directory to index")
    recursive: bool = Field(default=True, description="Index recursively")
    file_types: Optional[List[str]] = Field(default=None, description="File types to index")

    @field_validator("directory")
    @classmethod
    def directory_not_empty(cls, v: str) -> str:
        """Validate directory is not empty"""
        if not v.strip():
            raise ValueError("Directory cannot be empty")
        return v.strip()


class IndexResponse(BaseModel):
    """File indexing response"""
    success: bool = Field(..., description="Whether indexing was successful")
    files_indexed: int = Field(default=0, description="Number of files indexed")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    errors: List[str] = Field(default_factory=list, description="Indexing errors")
    duration_seconds: float = Field(default=0.0, description="Indexing duration")


class SearchRequest(BaseModel):
    """RAG search request"""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")


class SearchResult(BaseModel):
    """Single search result"""
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Source file")
    score: float = Field(..., description="Similarity score")
    page: Optional[int] = Field(default=None, description="Page number (for PDFs)")


class SearchResponse(BaseModel):
    """RAG search response"""
    query: str = Field(..., description="Search query")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(default=0, description="Total number of results")


class CommandHistoryItem(BaseModel):
    """Single command history item"""
    id: str = Field(..., description="Command ID")
    command: str = Field(..., description="Command string")
    result: CommandResult = Field(..., description="Command result")
    user_input: Optional[str] = Field(default=None, description="Original user input")


class HistoryResponse(BaseModel):
    """Command history response"""
    commands: List[CommandHistoryItem] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    per_page: int = Field(default=20)


class HealthStatus(BaseModel):
    """Health check status"""
    status: str = Field(..., description="Overall status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, Any] = Field(default_factory=dict, description="Service statuses")


class SafetyCheckRequest(BaseModel):
    """Safety check request"""
    command: str = Field(..., description="Command to check")


class SafetyCheckResponse(BaseModel):
    """Safety check response"""
    command: str = Field(..., description="Command checked")
    is_safe: bool = Field(..., description="Whether command is safe")
    risk_level: RiskLevel = Field(..., description="Risk level")
    requires_confirmation: bool = Field(..., description="Whether confirmation is required")
    reason: Optional[str] = Field(default=None, description="Reason for risk level")


class SessionInfo(BaseModel):
    """Session information"""
    session_id: str = Field(..., description="Session ID")
    created_at: datetime = Field(..., description="Session creation time")
    message_count: int = Field(default=0, description="Number of messages")
    last_activity: Optional[datetime] = Field(default=None)


class ConfigResponse(BaseModel):
    """Configuration response"""
    app_name: str = Field(...)
    version: str = Field(...)
    llm_model: str = Field(...)
    safety_enabled: bool = Field(...)
    rag_enabled: bool = Field(...)
    supported_extensions: List[str] = Field(default_factory=list)
