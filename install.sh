#!/bin/bash
#
# AI OS Installation Script
# Sets up the Local AI Assistant environment
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.10"
VENV_DIR="venv"
REQUIRED_PACKAGES=("python3" "python3-pip" "python3-venv" "curl")

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║   🤖 AI OS - Local AI Assistant Installation                 ║"
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

# Check if running on Linux
check_os() {
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_warning "This script is designed for Linux. Your OS: $OSTYPE"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Check for required packages
check_dependencies() {
    print_status "Checking dependencies..."
    
    missing_packages=()
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! command -v "$package" &> /dev/null; then
            # Special check for python3-venv
            if [ "$package" = "python3-venv" ]; then
                if ! python3 -m venv --help &> /dev/null; then
                    missing_packages+=("$package")
                fi
            elif [ "$package" = "python3-pip" ]; then
                if ! python3 -m pip --version &> /dev/null; then
                    missing_packages+=("$package")
                fi
            else
                missing_packages+=("$package")
            fi
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        print_error "Missing packages: ${missing_packages[*]}"
        print_status "Please install them using your package manager:"
        echo "  Ubuntu/Debian: sudo apt-get install ${missing_packages[*]}"
        echo "  CentOS/RHEL:   sudo yum install ${missing_packages[*]}"
        echo "  Arch:          sudo pacman -S ${missing_packages[*]}"
        exit 1
    fi
    
    print_success "All dependencies are installed"
}

# Check Python version
check_python_version() {
    print_status "Checking Python version..."
    
    PYTHON_CMD=$(command -v python3.10 || command -v python3 || command -v python)
    
    if [ -z "$PYTHON_CMD" ]; then
        print_error "Python not found. Please install Python $PYTHON_VERSION or higher."
        exit 1
    fi
    
    PYTHON_VERSION_FULL=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_VERSION_MAJOR=$(echo $PYTHON_VERSION_FULL | cut -d. -f1)
    PYTHON_VERSION_MINOR=$(echo $PYTHON_VERSION_FULL | cut -d. -f2)
    
    if [ "$PYTHON_VERSION_MAJOR" -lt 3 ] || ([ "$PYTHON_VERSION_MAJOR" -eq 3 ] && [ "$PYTHON_VERSION_MINOR" -lt 10 ]); then
        print_error "Python $PYTHON_VERSION_FULL is too old. Please install Python $PYTHON_VERSION or higher."
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION_FULL is compatible"
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            print_status "Using existing virtual environment"
            return
        fi
    fi
    
    $PYTHON_CMD -m venv "$VENV_DIR"
    print_success "Virtual environment created at $VENV_DIR"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    print_success "Dependencies installed successfully"
}

# Create directories
setup_directories() {
    print_status "Setting up directories..."
    
    mkdir -p data/vector_store
    mkdir -p logs
    mkdir -p examples
    
    print_success "Directories created"
}

# Create CLI symlink
create_symlink() {
    print_status "Creating CLI symlink..."
    
    CLI_PATH="$(pwd)/cli/cli.py"
    SYMLINK_PATH="$VENV_DIR/bin/aios"
    
    cat > "$SYMLINK_PATH" << EOF
#!/bin/bash
source "$(pwd)/$VENV_DIR/bin/activate"
python "$CLI_PATH" "\$@"
EOF
    
    chmod +x "$SYMLINK_PATH"
    
    print_success "CLI symlink created at $SYMLINK_PATH"
    print_status "You can use 'aios' command after activating the virtual environment"
}

# Check Ollama installation
check_ollama() {
    print_status "Checking Ollama installation..."
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama is installed"
        
        # Check if Ollama is running
        if curl -s http://localhost:11434/api/tags &> /dev/null; then
            print_success "Ollama is running"
            
            # Check for llama3.2 model
            if ollama list | grep -q "llama3.2"; then
                print_success "llama3.2 model is available"
            else
                print_warning "llama3.2 model not found"
                print_status "Pulling llama3.2 model..."
                ollama pull llama3.2 || print_warning "Failed to pull model. You can pull it later with: ollama pull llama3.2"
            fi
        else
            print_warning "Ollama is installed but not running"
            print_status "Start Ollama with: ollama serve"
        fi
    else
        print_warning "Ollama is not installed"
        print_status "Please install Ollama:"
        echo "  curl -fsSL https://ollama.com/install.sh | sh"
    fi
}

# Create example files
create_examples() {
    print_status "Creating example files..."
    
    # Sample document for RAG testing
    cat > examples/sample_document.md << 'EOF'
# AI OS Sample Document

This is a sample document for testing the RAG (Retrieval Augmented Generation) functionality.

## Features

AI OS provides the following features:

1. **Chat Interface** - Natural language conversation using local LLM
2. **Command Execution** - Convert natural language to Linux commands and execute safely
3. **Safety System** - Block dangerous commands, require confirmation for risky ones
4. **RAG (Retrieval Augmented Generation)** - Search and answer from local files

## Safety Features

The safety system includes:

- Block list for dangerous commands (rm -rf /, sudo, etc.)
- Risk level assessment (LOW, MEDIUM, HIGH, CRITICAL)
- Confirmation system for risky commands
- Timeout protection (30 seconds default)
- Command logging and history

## Usage Examples

### Chat Mode
```
aios "What is the current directory?"
```

### Command Execution
```
aios "show disk usage" --run
```

### RAG Search
```
aios "find information about safety" --rag
```

## Configuration

Configuration is stored in `config/config.yaml`. You can customize:

- LLM model and parameters
- Safety settings
- RAG embedding model
- Server configuration
EOF

    # Sample Python file for RAG testing
    cat > examples/sample_code.py << 'EOF'
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
EOF

    print_success "Example files created in examples/"
}

# Main installation function
main() {
    print_status "Starting AI OS installation..."
    
    check_os
    check_dependencies
    check_python_version
    create_venv
    install_dependencies
    setup_directories
    create_symlink
    check_ollama
    create_examples
    
    echo
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                              ║${NC}"
    echo -e "${GREEN}║   ✅ AI OS installed successfully!                           ║${NC}"
    echo -e "${GREEN}║                                                              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo "Next steps:"
    echo "  1. Activate virtual environment: source $VENV_DIR/bin/activate"
    echo "  2. Start the server: ./start.sh"
    echo "  3. Open web UI: http://localhost:8000/ui"
    echo "  4. Use CLI: python cli/cli.py --chat"
    echo
    echo "Documentation:"
    echo "  - API docs: http://localhost:8000/docs"
    echo "  - README: cat README.md"
    echo
}

# Run main function
main "$@"
