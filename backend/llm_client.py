"""
LLM Client for VOIDER - Linux Companion
Universal LangChain-powered client supporting multiple AI providers:
  - Ollama (local, free)
  - Groq (fast, free tier)
  - OpenAI (GPT-4o etc.)
  - Google Gemini (free tier)
  - xAI / Grok (OpenAI-compatible)
"""

import logging
import os
from typing import AsyncGenerator, Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# Default system prompts
DEFAULT_CHAT_PROMPT = """You are VOIDER, a friendly {target_os} companion AI assistant. You can:
1. Answer general questions and have conversations
2. Generate and explain {target_os} commands
3. Search and answer from user's local files (RAG)

Be concise, accurate, and helpful. If you don't know something, say so."""

DEFAULT_COMMAND_PROMPT = """You are a {target_os} command generator. Convert user requests to safe commands.
Rules:
- Output ONLY the command, no explanations
- Use safe flags (e.g. df -h instead of df if on Linux)
- If request is dangerous, output: "BLOCKED: [reason]"
- If unclear, output: "CLARIFY: [question]"
- Command MUST be compatible with {target_os}

Examples (Linux):
"check disk" → df -h
"show memory" → free -h
"list python processes" → ps aux | grep python
"delete everything" → BLOCKED: Destructive command not allowed
Examples (Windows):
"check disk" → Get-CimInstance Win32_LogicalDisk
"show memory" → Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory"""

DEFAULT_RAG_PROMPT = """You are an AI assistant answering questions based on user's files.
Context from files:
{context}

User question: {question}

Answer based ONLY on the context above.
If answer not in context, say "I don't have information about that in your files."
Be concise and accurate."""


# Supported providers configuration
PROVIDERS = {
    "ollama": {
        "name": "Ollama (Local)",
        "needs_api_key": False,
        "default_model": "llama3.2",
        "models": ["llama3.2", "llama3.1", "mistral", "gemma2", "qwen2.5", "phi3"],
    },
    "groq": {
        "name": "Groq",
        "needs_api_key": True,
        "default_model": "llama-3.3-70b-versatile",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    },
    "openai": {
        "name": "OpenAI",
        "needs_api_key": True,
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "gemini": {
        "name": "Google Gemini",
        "needs_api_key": True,
        "default_model": "gemini-1.5-flash",
        "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
    },
    "xai": {
        "name": "xAI / Grok",
        "needs_api_key": True,
        "default_model": "grok-beta",
        "models": ["grok-beta", "grok-vision-beta"],
    },
}


def _build_langchain_model(provider: str, model: str, api_key: Optional[str] = None):
    """Build the appropriate LangChain model object for the given provider."""
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0.7)

    elif provider == "groq":
        from langchain_groq import ChatGroq
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        return ChatGroq(model=model, groq_api_key=key, temperature=0.7)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        return ChatOpenAI(model=model, openai_api_key=key, temperature=0.7)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0.7)

    elif provider == "xai":
        from langchain_openai import ChatOpenAI
        key = api_key or os.environ.get("XAI_API_KEY", "")
        return ChatOpenAI(
            model=model,
            openai_api_key=key,
            openai_api_base="https://api.x.ai/v1",
            temperature=0.7,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


class LLMClient:
    """Universal LangChain-powered LLM client for VOIDER."""

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        # Legacy Ollama params (kept for backward compat)
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 60,
        system_prompt: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model or PROVIDERS.get(provider, {}).get("default_model", "llama3.2")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or DEFAULT_CHAT_PROMPT

        self._lc_model = _build_langchain_model(provider, self.model, api_key)
        logger.info(f"LLM client initialized: provider={provider}, model={self.model}")

    def _invoke(self, messages: list) -> str:
        """Run synchronous LangChain invocation."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        lc_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        response = self._lc_model.invoke(lc_messages)
        return response.content

    def chat(
        self,
        message: str,
        context=None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        target_os: str = "Linux",
    ) -> str:
        """Send a chat message and get a response."""
        sys_prompt = system_prompt or self.system_prompt
        if "{target_os}" in sys_prompt:
            sys_prompt = sys_prompt.format(target_os=target_os)
            
        messages = [{"role": "system", "content": sys_prompt}]
        if context:
            for msg in context:
                messages.append({"role": msg.role.value, "content": msg.content})
        messages.append({"role": "user", "content": message})
        try:
            return self._invoke(messages)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise

    async def stream_chat(
        self,
        message: str,
        context=None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        target_os: str = "Linux",
    ) -> AsyncGenerator[str, None]:
        """Stream chat response using LangChain async streaming."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        lc_messages = []
        sys_prompt = system_prompt or self.system_prompt
        if "{target_os}" in sys_prompt:
            sys_prompt = sys_prompt.format(target_os=target_os)
            
        lc_messages.append(SystemMessage(content=sys_prompt))

        if context:
            for msg in context:
                role = msg.role.value if hasattr(msg.role, "value") else msg.role
                content = msg.content
                if role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

        lc_messages.append(HumanMessage(content=message))

        try:
            async for chunk in self._lc_model.astream(lc_messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Stream chat error: {e}")
            raise

    def generate_command(self, request: str, target_os: str = "Linux") -> str:
        """Generate a Linux command from natural language."""
        try:
            sys_prompt = DEFAULT_COMMAND_PROMPT
            if "{target_os}" in sys_prompt:
                sys_prompt = sys_prompt.format(target_os=target_os)
                
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": request},
            ]
            command = self._invoke(messages).strip()

            # Clean up code block wrappers if present
            if command.startswith("```"):
                lines = command.split("\n")
                if len(lines) > 2:
                    command = "\n".join(lines[1:-1]).strip()
                else:
                    command = command.replace("```", "").strip()

            command = command.replace("bash ", "").replace("sh ", "").strip()
            logger.info(f"Generated command: {command[:60]}")
            return command
        except Exception as e:
            logger.error(f"Command generation error: {e}")
            return f"ERROR: {str(e)}"

    def generate_rag_response(self, question: str, context: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using RAG context."""
        try:
            rag_prompt = system_prompt or DEFAULT_RAG_PROMPT
            formatted_prompt = rag_prompt.format(context=context, question=question)
            messages = [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": question},
            ]
            return self._invoke(messages)
        except Exception as e:
            logger.error(f"RAG response error: {e}")
            raise

    def check_model_available(self) -> bool:
        """Check if provider/model is reachable."""
        try:
            # Quick test ping
            self.chat("ping", system_prompt="Reply with one word: pong")
            return True
        except Exception as e:
            logger.warning(f"Model availability check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        """Return known models for current provider."""
        return PROVIDERS.get(self.provider, {}).get("models", [self.model])

    def pull_model(self) -> bool:
        """Only meaningful for Ollama provider."""
        if self.provider == "ollama":
            try:
                import ollama as ol
                for _ in ol.Client().pull(self.model, stream=True):
                    pass
                return True
            except Exception as e:
                logger.error(f"Model pull error: {e}")
                return False
        return True  # No-op for cloud providers


# Global LLM client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    # Legacy compat params
    base_url: str = "http://localhost:11434",
    **kwargs,
) -> LLMClient:
    """Get or create global LLM client instance."""
    global _llm_client
    _llm_client = LLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
    return _llm_client


def get_providers_info() -> Dict:
    """Return full providers config (safe, no API keys)."""
    return PROVIDERS
