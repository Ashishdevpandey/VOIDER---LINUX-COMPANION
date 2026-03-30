#!/bin/bash
#
# AI OS Startup Script
# Starts the AI OS server with all dependencies
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR="venv"
OLLAMA_URL="http://localhost:11434"
API_PORT=8000
API_HOST="0.0.0.0"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║   🤖 AI OS - Local AI Assistant                              ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
check_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Virtual environment not found at $VENV_DIR"
        print_status "Please run the installation script first: ./install.sh"
        exit 1
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
}

# Check if Ollama is running
check_ollama() {
    print_status "Checking Ollama..."
    
    if ! curl -s "$OLLAMA_URL/api/tags" &> /dev/null; then
        print_warning "Ollama is not running at $OLLAMA_URL"
        
        # Check if Ollama is installed
        if command -v ollama &> /dev/null; then
            print_status "Starting Ollama..."
            ollama serve &
            OLLAMA_PID=$!
            
            # Wait for Ollama to start
            for i in {1..30}; do
                if curl -s "$OLLAMA_URL/api/tags" &> /dev/null; then
                    print_success "Ollama started successfully (PID: $OLLAMA_PID)"
                    return
                fi
                sleep 1
            done
            
            print_error "Ollama failed to start within 30 seconds"
            exit 1
        else
            print_error "Ollama is not installed"
            print_status "Please install Ollama:"
            echo "  curl -fsSL https://ollama.com/install.sh | sh"
            exit 1
        fi
    else
        print_success "Ollama is running"
    fi
}

# Check for required model
check_model() {
    print_status "Checking for llama3.2 model..."
    
    if ! curl -s "$OLLAMA_URL/api/tags" | grep -q "llama3.2"; then
        print_warning "llama3.2 model not found"
        print_status "Pulling llama3.2 model (this may take a while)..."
        
        if command -v ollama &> /dev/null; then
            ollama pull llama3.2
            print_success "llama3.2 model pulled successfully"
        else
            print_warning "Cannot pull model - ollama command not available"
        fi
    else
        print_success "llama3.2 model is available"
    fi
}

# Create necessary directories
setup_directories() {
    print_status "Setting up directories..."
    
    mkdir -p data/vector_store
    mkdir -p logs
    
    print_success "Directories ready"
}

# Start the API server
start_server() {
    print_status "Starting AI OS API server..."
    
    # Check if port is already in use
    if lsof -Pi :$API_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $API_PORT is already in use"
        print_status "Attempting to stop existing process..."
        lsof -Pi :$API_PORT -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    print_status "Starting server on http://$API_HOST:$API_PORT"
    
    # Export environment variables
    export PYTHONPATH="$(pwd)"
    export PYTHONUNBUFFERED=1
    
    # Start server in background
    uvicorn backend.main:app \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --reload \
        --log-level info &
    
    SERVER_PID=$!
    
    # Wait for server to start
    print_status "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s "http://localhost:$API_PORT/health" &> /dev/null; then
            print_success "Server started successfully (PID: $SERVER_PID)"
            return
        fi
        sleep 1
    done
    
    print_error "Server failed to start within 30 seconds"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
}

# Print access information
print_access_info() {
    echo
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                              ║${NC}"
    echo -e "${GREEN}║   ✅ AI OS is running!                                       ║${NC}"
    echo -e "${GREEN}║                                                              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo "Access Points:"
    echo -e "  🌐 Web UI:     ${CYAN}http://localhost:$API_PORT/ui${NC}"
    echo -e "  📚 API Docs:   ${CYAN}http://localhost:$API_PORT/docs${NC}"
    echo -e "  🔍 Health:     ${CYAN}http://localhost:$API_PORT/health${NC}"
    echo
    echo "CLI Usage:"
    echo -e "  ${CYAN}python cli/cli.py --chat${NC}           Start interactive chat"
    echo -e "  ${CYAN}python cli/cli.py --status${NC}         Check API status"
    echo -e "  ${CYAN}python cli/cli.py --help${NC}           Show all options"
    echo
    echo "Examples:"
    echo -e "  ${CYAN}curl http://localhost:$API_PORT/chat -d '{\"message\":\"hello\"}'${NC}"
    echo
    echo "Press Ctrl+C to stop the server"
    echo
}

# Cleanup function
cleanup() {
    echo
    print_status "Shutting down AI OS..."
    
    # Kill server
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        print_status "Server stopped"
    fi
    
    # Kill Ollama (only if we started it)
    if [ -n "$OLLAMA_PID" ]; then
        kill $OLLAMA_PID 2>/dev/null || true
        print_status "Ollama stopped"
    fi
    
    print_success "AI OS stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Main function
main() {
    print_status "Starting AI OS..."
    
    check_venv
    activate_venv
    check_ollama
    check_model
    setup_directories
    start_server
    print_access_info
    
    # Keep script running
    wait
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-ollama)
            SKIP_OLLAMA=1
            shift
            ;;
        --port)
            API_PORT="$2"
            shift 2
            ;;
        --host)
            API_HOST="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-ollama    Skip Ollama checks (use if running Ollama separately)"
            echo "  --port PORT    Set API port (default: 8000)"
            echo "  --host HOST    Set API host (default: 0.0.0.0)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"
