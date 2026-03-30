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
