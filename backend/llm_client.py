"""
LLM Client for AI OS
Handles communication with Ollama for chat and command generation
"""

import logging
from typing import AsyncGenerator, Dict, List, Optional, Any
import asyncio

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.models import MessageRole, ChatMessage

logger = logging.getLogger(__name__)


# Default system prompts
DEFAULT_CHAT_PROMPT = """You are AI OS, a helpful Linux assistant. You can:
1. Answer general questions
2. Generate and execute Linux commands
3. Search and answer from user's local files (RAG)

Be concise, accurate, and helpful. If you don't know something, say so."""

DEFAULT_COMMAND_PROMPT = """You are a Linux command generator. Convert user requests to safe commands.
Rules:
- Output ONLY the command, no explanations
- Use safe flags (df -h, not df)
- If request is dangerous, output: "BLOCKED: [reason]"
- If unclear, output: "CLARIFY: [question]"

Examples:
"check disk" → df -h
"show memory" → free -h
"list python processes" → ps aux | grep python
"delete everything" → BLOCKED: Destructive command not allowed"""

DEFAULT_RAG_PROMPT = """You are an AI assistant answering questions based on user's files.
Context from files:
{context}

User question: {question}

Answer based ONLY on the context above.
If answer not in context, say "I don't have information about that in your files."
Be concise and accurate."""


class LLMClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 60,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize LLM client
        
        Args:
            model: Ollama model name
            base_url: Ollama server URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            system_prompt: Default system prompt
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt or DEFAULT_CHAT_PROMPT
        
        # Initialize Ollama client
        self._client = ollama.Client(host=base_url)
        
        logger.info(f"LLM client initialized with model: {model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def chat(
        self,
        message: str,
        context: Optional[List[ChatMessage]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a chat message and get response
        
        Args:
            message: User message
            context: Previous conversation context
            system_prompt: Override system prompt
            temperature: Override temperature
            
        Returns:
            AI response text
        """
        try:
            # Build messages
            messages = []
            
            # Add system prompt
            sys_prompt = system_prompt or self.system_prompt
            messages.append({"role": "system", "content": sys_prompt})
            
            # Add context
            if context:
                for msg in context:
                    messages.append({
                        "role": msg.role.value,
                        "content": msg.content,
                    })
            
            # Add user message
            messages.append({"role": "user", "content": message})
            
            # Make request
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature or self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            
            return response["message"]["content"]
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise
    
    async def stream_chat(
        self,
        message: str,
        context: Optional[List[ChatMessage]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response
        
        Args:
            message: User message
            context: Previous conversation context
            system_prompt: Override system prompt
            temperature: Override temperature
            
        Yields:
            Chunks of response text
        """
        try:
            # Build messages
            messages = []
            
            # Add system prompt
            sys_prompt = system_prompt or self.system_prompt
            messages.append({"role": "system", "content": sys_prompt})
            
            # Add context
            if context:
                for msg in context:
                    messages.append({
                        "role": msg.role.value,
                        "content": msg.content,
                    })
            
            # Add user message
            messages.append({"role": "user", "content": message})
            
            # Stream response
            stream = self._client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature or self.temperature,
                    "num_predict": self.max_tokens,
                },
                stream=True,
            )
            
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
                    
        except Exception as e:
            logger.error(f"Stream chat error: {e}")
            raise
    
    def generate_command(self, request: str) -> str:
        """
        Generate a Linux command from natural language
        
        Args:
            request: Natural language request
            
        Returns:
            Generated command or BLOCKED/CLARIFY message
        """
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": DEFAULT_COMMAND_PROMPT},
                    {"role": "user", "content": request},
                ],
                options={
                    "temperature": 0.1,  # Low temperature for consistency
                    "num_predict": 256,
                },
            )
            
            command = response["message"]["content"].strip()
            
            # Clean up the response
            # Remove code blocks if present
            if command.startswith("```"):
                lines = command.split("\n")
                if len(lines) > 2:
                    command = "\n".join(lines[1:-1]).strip()
                else:
                    command = command.replace("```", "").strip()
            
            # Remove bash/sh prefix if present
            command = command.replace("bash ", "").replace("sh ", "").strip()
            
            logger.info(f"Generated command for '{request[:30]}...': {command[:50]}")
            return command
            
        except Exception as e:
            logger.error(f"Command generation error: {e}")
            return f"ERROR: {str(e)}"
    
    def generate_rag_response(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate response using RAG context
        
        Args:
            question: User question
            context: Retrieved context from documents
            system_prompt: Override RAG prompt
            
        Returns:
            Generated response
        """
        try:
            # Format prompt with context
            rag_prompt = system_prompt or DEFAULT_RAG_PROMPT
            formatted_prompt = rag_prompt.format(
                context=context,
                question=question,
            )
            
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": formatted_prompt},
                    {"role": "user", "content": question},
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            
            return response["message"]["content"]
            
        except Exception as e:
            logger.error(f"RAG response error: {e}")
            raise
    
    def check_model_available(self) -> bool:
        """
        Check if the configured model is available
        
        Returns:
            True if model is available
        """
        try:
            models = self._client.list()
            available_models = [m["model"] for m in models.get("models", [])]
            
            # Check exact match or model name without tag
            for available in available_models:
                if self.model in available or available in self.model:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Model check error: {e}")
            return False
    
    def pull_model(self) -> bool:
        """
        Pull the configured model
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Pulling model: {self.model}")
            
            for progress in self._client.pull(self.model, stream=True):
                if "status" in progress:
                    logger.info(f"Pull progress: {progress['status']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model pull error: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """
        List available models
        
        Returns:
            List of model names
        """
        try:
            models = self._client.list()
            return [m["model"] for m in models.get("models", [])]
            
        except Exception as e:
            logger.error(f"List models error: {e}")
            return []


# Global LLM client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client(
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    **kwargs,
) -> LLMClient:
    """
    Get or create global LLM client instance
    
    Args:
        model: Model name
        base_url: Ollama server URL
        **kwargs: Additional client options
        
    Returns:
        LLMClient instance
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(model=model, base_url=base_url, **kwargs)
    return _llm_client
