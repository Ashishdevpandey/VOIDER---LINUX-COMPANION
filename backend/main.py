"""
FastAPI Application for AI OS
Main API server for the Local AI Assistant
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import json
from fastapi.staticfiles import StaticFiles

from backend.models import (
    ChatRequest,
    ChatResponse,
    CommandHistoryItem,
    CommandResult,
    ConfigResponse,
    HealthStatus,
    HistoryResponse,
    IndexRequest,
    IndexResponse,
    MessageRole,
    SafetyCheckRequest,
    SafetyCheckResponse,
    SearchRequest,
    SearchResponse,
)
from backend.llm_client import get_llm_client, LLMClient, DEFAULT_RAG_PROMPT, get_providers_info, PROVIDERS
from backend.command_executor import get_command_executor, CommandExecutor
from backend.rag_engine import get_rag_engine, RAGEngine
from backend.safety import get_safety_checker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
llm_client: Optional[LLMClient] = None
command_executor: Optional[CommandExecutor] = None
rag_engine: Optional[RAGEngine] = None
config: Dict = {}


def load_config() -> Dict:
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    # Default config
    return {
        "app": {"name": "AI OS", "version": "1.0.0"},
        "llm": {"model": "llama3.2", "base_url": "http://localhost:11434"},
        "server": {"host": "0.0.0.0", "port": 8000},
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global llm_client, command_executor, rag_engine, config
    
    # Startup
    logger.info("Starting AI OS server...")
    
    # Load config
    config = load_config()
    
    # Initialize components
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", None)
    api_key = llm_config.get("api_key", None) or os.environ.get(f"{provider.upper()}_API_KEY", None)
    llm_client = get_llm_client(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=llm_config.get("base_url", "http://localhost:11434"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 2048),
    )
    
    safety_config = config.get("safety", {})
    safety_checker = get_safety_checker()
    command_executor = get_command_executor(
        safety_checker=safety_checker,
        default_timeout=safety_config.get("default_timeout", 30),
    )
    
    rag_config = config.get("rag", {})
    if rag_config.get("enabled", True):
        rag_engine = get_rag_engine(
            embedding_model=rag_config.get("embedding_model", "all-MiniLM-L6-v2"),
            chunk_size=rag_config.get("chunk_size", 500),
            chunk_overlap=rag_config.get("chunk_overlap", 50),
            vector_store_path=rag_config.get("vector_store_path", "./data/vector_store"),
        )
    
    logger.info("AI OS server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI OS server...")


# Create FastAPI app
app = FastAPI(
    title="AI OS - Local AI Assistant",
    description="Linux AI Assistant with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for UI
ui_path = Path(__file__).parent.parent / "ui"
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint - API info"""
    return {
        "name": "AI OS",
        "version": "1.0.0",
        "description": "Local AI Assistant with RAG for Linux",
        "docs": "/docs",
        "ui": "/ui",
        "health": "/health",
    }


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    services = {
        "api": {"status": "healthy"},
        "llm": {"status": "unknown"},
        "rag": {"status": "unknown"},
    }
    
    # Check LLM
    if llm_client:
        try:
            if llm_client.check_model_available():
                services["llm"] = {"status": "healthy", "model": llm_client.model}
            else:
                services["llm"] = {"status": "unhealthy", "error": "Model not available"}
        except Exception as e:
            services["llm"] = {"status": "unhealthy", "error": str(e)}
    
    # Check RAG
    if rag_engine:
        stats = rag_engine.get_stats()
        services["rag"] = {
            "status": "healthy",
            "documents": stats["total_documents"],
            "index_built": stats["index_built"],
        }
    
    # Overall status
    overall_status = "healthy"
    for service in services.values():
        if service.get("status") != "healthy":
            overall_status = "degraded"
            break
    
    return HealthStatus(
        status=overall_status,
        version="1.0.0",
        services=services,
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get API configuration"""
    rag_config = config.get("rag", {})
    
    return ConfigResponse(
        app_name=config.get("app", {}).get("name", "AI OS"),
        version=config.get("app", {}).get("version", "1.0.0"),
        llm_model=config.get("llm", {}).get("model", "llama3.2"),
        safety_enabled=config.get("safety", {}).get("enabled", True),
        rag_enabled=rag_config.get("enabled", True),
        supported_extensions=rag_config.get("supported_extensions", []),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    
    Handles:
    - General chat
    - Command generation and execution
    - RAG-based Q&A
    """
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    
    session_id = request.session_id or str(uuid.uuid4())[:8]
    
    try:
        # Check if this is a command request
        user_message_lower = request.message.lower()
        is_command_request = any(
            keyword in user_message_lower
            for keyword in ["run", "execute", "command", "show", "check", "list", "find", "get"]
        )
        
        command_result = None
        rag_context = None
        
        # Handle RAG request
        if request.use_rag and rag_engine:
            context = rag_engine.get_context_string(request.message)
            response_text = llm_client.generate_rag_response(
                question=request.message,
                context=context,
            )
            rag_context = [context]
        
        # Handle command generation only (from UI)
        elif request.message.lower().startswith("generate command:"):
            actual_request = request.message[17:].strip()
            response_text = llm_client.generate_command(actual_request, target_os=request.target_os)
            
        # Handle command generation and execution
        elif is_command_request and request.execute_command:
            # Generate command
            generated_command = llm_client.generate_command(request.message, target_os=request.target_os)
            
            # Check if blocked by LLM
            if generated_command.startswith("BLOCKED:"):
                response_text = f"⚠️ {generated_command}"
            elif generated_command.startswith("CLARIFY:"):
                response_text = f"❓ {generated_command}"
            elif generated_command.startswith("ERROR:"):
                response_text = f"❌ {generated_command}"
            else:
                # Execute command
                result = await command_executor.execute(
                    command=generated_command,
                    user_input=request.message,
                )
                command_result = result
                
                # Build response
                if result.blocked:
                    response_text = f"⚠️ Command blocked: {result.block_reason}\n\nCommand: `{result.command}`"
                elif result.returncode == 0:
                    output = result.stdout or "(no output)"
                    response_text = f"✅ Command executed successfully\n\n```\n$ {result.command}\n{output}\n```"
                else:
                    error = result.stderr or "(unknown error)"
                    response_text = f"❌ Command failed (exit {result.returncode})\n\n```\n$ {result.command}\n{error}\n```"
        
        # Regular chat
        else:
            response_text = llm_client.chat(
                message=request.message,
                context=request.context,
                target_os=request.target_os,
            )
        
        return ChatResponse(
            message=response_text,
            session_id=session_id,
            command_result=command_result,
            rag_context=rag_context,
            model=llm_client.model,
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/simple")
async def chat_simple(request: Dict):
    """Simple chat endpoint for basic use"""
    chat_request = ChatRequest(
        message=request.get("message", ""),
        session_id=request.get("session_id"),
        execute_command=request.get("execute_command", False),
        use_rag=request.get("use_rag", False),
        target_os=request.get("target_os", "Linux"),
    )
    return await chat(chat_request)


@app.post("/chat/stream")
async def chat_stream(request: Dict):
    """Chat endpoint with SSE streaming"""
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    
    session_id = request.get("session_id") or str(uuid.uuid4())[:8]
    message = request.get("message", "")
    execute_command = request.get("execute_command", False)
    use_rag = request.get("use_rag", False)
    target_os = request.get("target_os", "Linux")
    
    async def generate_sse():
        try:
            init_data = json.dumps({'session_id': session_id})
            yield f"data: {init_data}\n\n"
            
            user_message_lower = message.lower()
            is_command_request = any(
                keyword in user_message_lower
                for keyword in ["run", "execute", "command", "show", "check", "list", "find", "get"]
            )
            
            # Only generate command (from UI Command Generator)
            if message.lower().startswith("generate command:"):
                actual_request = message[17:].strip()
                response_text = llm_client.generate_command(actual_request, target_os=target_os)
                chunk_data = json.dumps({'chunk': response_text})
                yield f"data: {chunk_data}\n\n"
            
            # Handle RAG request
            elif use_rag and rag_engine:
                context = rag_engine.get_context_string(message)
                rag_prompt = DEFAULT_RAG_PROMPT.format(context=context, question=message)
                async for chunk in llm_client.stream_chat(message, system_prompt=rag_prompt):
                    chunk_data = json.dumps({'chunk': chunk})
                    yield f"data: {chunk_data}\n\n"
            
            # Handle command generation and execution
            elif is_command_request and execute_command:
                chunk_data = json.dumps({'chunk': 'Generating command...\n'})
                yield f"data: {chunk_data}\n\n"
                generated_command = llm_client.generate_command(message, target_os=target_os)
                
                if generated_command.startswith("BLOCKED:") or generated_command.startswith("CLARIFY:") or generated_command.startswith("ERROR:"):
                    chunk_data = json.dumps({'chunk': generated_command})
                    yield f"data: {chunk_data}\n\n"
                else:
                    chunk_data = json.dumps({'chunk': f'Executing: `{generated_command}`...\n\n'})
                    yield f"data: {chunk_data}\n\n"
                    
                    result = await command_executor.execute(
                        command=generated_command,
                        user_input=message,
                    )
                    
                    if result.blocked:
                        text = f"⚠️ Command blocked: {result.block_reason}\n\nCommand: `{result.command}`"
                    elif result.returncode == 0:
                        output = result.stdout or "(no output)"
                        text = f"✅ Command executed successfully\n\n```\n$ {result.command}\n{output}\n```"
                    else:
                        error = result.stderr or "(unknown error)"
                        text = f"❌ Command failed (exit {result.returncode})\n\n```\n$ {result.command}\n{error}\n```"
                    chunk_data = json.dumps({'chunk': text})
                    yield f"data: {chunk_data}\n\n"
            
            # Regular chat
            else:
                async for chunk in llm_client.stream_chat(message, target_os=target_os):
                    chunk_data = json.dumps({'chunk': chunk})
                    yield f"data: {chunk_data}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_data = json.dumps({'error': str(e)})
            yield f"data: {error_data}\n\n"
            
    return StreamingResponse(generate_sse(), media_type="text/event-stream")


@app.post("/command/execute", response_model=CommandResult)
async def execute_command(request: Dict):
    """Execute a command directly"""
    if not command_executor:
        raise HTTPException(status_code=503, detail="Command executor not initialized")
    
    command = request.get("command", "").strip()
    if not command:
        raise HTTPException(status_code=400, detail="Command is required")
    
    try:
        result = await command_executor.execute(command)
        return result
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/command/check", response_model=SafetyCheckResponse)
async def check_command(request: SafetyCheckRequest):
    """Check if a command is safe"""
    safety_checker = get_safety_checker()
    return safety_checker.check_command(request.command)


@app.get("/command/history", response_model=HistoryResponse)
async def get_history(
    limit: int = 20,
    offset: int = 0,
):
    """Get command execution history"""
    if not command_executor:
        raise HTTPException(status_code=503, detail="Command executor not initialized")
    
    history = command_executor.get_history(limit=limit, offset=offset)
    return HistoryResponse(
        commands=[
            CommandHistoryItem(
                id=item["id"],
                command=item["command"],
                result=CommandResult(
                    command=item["command"],
                    returncode=item["returncode"],
                    executed_at=datetime.fromisoformat(item["executed_at"]),
                    duration_ms=item["duration_ms"],
                    risk_level=item["risk_level"],
                    blocked=item["blocked"],
                ),
            )
            for item in history
        ],
        total=len(history),
        page=offset // limit + 1,
        per_page=limit,
    )


@app.delete("/command/history")
async def clear_history():
    """Clear command history"""
    if not command_executor:
        raise HTTPException(status_code=503, detail="Command executor not initialized")
    
    count = command_executor.clear_history()
    return {"message": f"Cleared {count} history items"}


@app.post("/rag/index", response_model=IndexResponse)
async def index_files(request: IndexRequest):
    """Index files for RAG"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        result = rag_engine.index_directory(
            directory=request.directory,
            recursive=request.recursive,
            file_types=request.file_types,
        )
        return result
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/search", response_model=SearchResponse)
async def search_rag(request: SearchRequest):
    """Search indexed documents"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        results = rag_engine.search(request)
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/stats")
async def get_rag_stats():
    """Get RAG engine statistics"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return rag_engine.get_stats()


@app.post("/rag/save")
async def save_rag_index(name: str = "default"):
    """Save RAG index to disk"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    path = rag_engine.save_index(name)
    return {"message": f"Index saved to {path}"}


@app.post("/rag/load")
async def load_rag_index(name: str = "default"):
    """Load RAG index from disk"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    success = rag_engine.load_index(name)
    if success:
        return {"message": f"Index '{name}' loaded successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Index '{name}' not found")


@app.delete("/rag/index")
async def clear_rag_index():
    """Clear RAG index"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    rag_engine.clear_index()
    return {"message": "RAG index cleared"}


@app.get("/provider/current")
async def get_current_provider():
    """Get current active LLM provider info (no API key exposed)"""
    if not llm_client:
        return {"provider": "none", "model": "none", "configured": False}
    provider_info = PROVIDERS.get(llm_client.provider, {})
    return {
        "provider": llm_client.provider,
        "provider_name": provider_info.get("name", llm_client.provider),
        "model": llm_client.model,
        "configured": True,
        "needs_api_key": provider_info.get("needs_api_key", False),
    }


@app.post("/provider/set")
async def set_provider(request: Dict):
    """Switch to a different LLM provider dynamically"""
    global llm_client

    provider = request.get("provider", "ollama")
    api_key = request.get("api_key", None)
    model = request.get("model", None)

    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}. Supported: {list(PROVIDERS.keys())}")

    provider_info = PROVIDERS[provider]
    if provider_info.get("needs_api_key") and not api_key:
        raise HTTPException(status_code=400, detail=f"{provider_info['name']} requires an API key.")

    try:
        llm_client = get_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
        )
        logger.info(f"Switched to provider: {provider}, model: {llm_client.model}")
        return {
            "success": True,
            "provider": provider,
            "provider_name": provider_info.get("name"),
            "model": llm_client.model,
            "message": f"Connected to {provider_info['name']} ({llm_client.model})",
        }
    except Exception as e:
        logger.error(f"Provider switch error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to {provider}: {str(e)}")


@app.get("/provider/list")
async def list_providers():
    """List all supported providers and their models"""
    return get_providers_info()


@app.get("/llm/models")
async def list_models():
    """List available LLM models"""
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    
    try:
        models = llm_client.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"List models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/pull")
async def pull_model(model: str):
    """Pull a model from Ollama"""
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    
    try:
        # This is synchronous and may take a while
        # In production, use background tasks
        success = llm_client.pull_model()
        if success:
            return {"message": f"Model {model} pulled successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to pull model")
    except Exception as e:
        logger.error(f"Pull model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 8000)
    
    uvicorn.run(app, host=host, port=port)
