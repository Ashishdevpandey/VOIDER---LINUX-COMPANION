# 🌌 VOIDER :- Linux Companion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-000000.svg)](https://ollama.com/)
[![UI: Glassmorphism](https://img.shields.io/badge/UI-macOS%20Glass-ec4899.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Your intelligent, beautifully-designed Linux companion. Chat, execute local system commands safely, and perform RAG-based file searches directly from a stunning macOS-inspired glassmorphic interface.

**VOIDER** replaces clunky terminal operations with a sleek, unified graphical interface running natively on your Linux system. Powered at its core by **Ollama**, **FastAPI**, and **FAISS**, it allows you to maintain full privacy with 100% local AI models while taking complete control of your desktop environment.

---

## ✨ How It Works (Architecture Details)

VOIDER operates via a decoupled architecture, joining a high-performance Python application on the backend with a lightweight, highly-polished HTML/CSS/JS frontend.

### 🧠 1. The Core AI Engine (Backend)
- **FastAPI Server**: Acts as the central nervous system, handling RESTful web requests and executing Python scripts `(http://localhost:8000)`.
- **Ollama Integration**: Talks locally to the Ollama daemon (defaults to `llama3.2`) to generate text dynamically. No cloud APIs, utter privacy.
- **RAG Engine**: Uses `FAISS` and lightweight sentence transformers to index designated folders on your filesystem. When you ask a question about your local documents, VOIDER retrieves the relevant text chunks to provide highly-accurate, contextualized answers.
- **Safe Command Executor**: Evaluates generated Bash/Linux commands against a multi-layered blocklist. High-risk commands (like disk formatting or `rm -rf /`) are permanently rejected or require manual verification.

### 🎨 2. The Glassmorphic Unibody UI (Frontend)
- **Aesthetic**: Borrowing the best from macOS layout philosophies, the frontend boasts a sweeping translucent "unibody" window with animated gradient background orbs and glowing neon accents.
- **Native Polish**: Traffic light window controls and floating panel designs ensure the app feels like a first-class premium citizen on your desktop.
- **Real-Time Interface**: Features Web API integrations to keep chat streams, command outputs, and RAG stats responsive without needing page reloads.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/voider.git
cd ai-os

# Run the automated installation script
chmod +x install.sh
./install.sh
```

### Boot Sequence

```bash
# Activate your virtual environment
source venv/bin/activate

# Ensure Ollama is running and has the llama3.2 model pulled
ollama serve &
ollama pull llama3.2

# Start the FastAPI Server (Port 8000)
./start.sh
```

**Access the UI**: Open http://localhost:8080/ (if deploying via standard python server) or navigate to the built-in HTML route provided by FastAPI.

---

## 🛠️ We Want YOU! (Calling All Developers)

**VOIDER** is designed to become the definitive AI-powered desktop companion for Linux power users, and we can't do it without the open-source community! 

Do you love Python, Bash optimizations, frontend styling, or AI engineering? **We want developers to contribute to this project.**

### Areas We Need Help With:
- 🖌️ **Advanced Frontend Frameworking**: Replacing our vanilla JS implementation with Tauri, Electron, or a Python QT/GTK architecture for a true standalone desktop application.
- 🐍 **Backend Optimization**: Introducing better asynchronous command streaming (`Server-Sent Events` upgrades) and smarter context-aware session memory that remembers your Linux configurations.
- 📦 **Packaging**: Helping us package VOIDER into a `.deb`, `RPM`, `AppImage`, or `Flatpak` for simple 1-click distributions on popular Linux distros like Ubuntu, Fedora, and Arch.
- 🧠 **Multi-Modal AI**: Expanding local integrations to run vision models (like LLaVA) for screen-understanding or image interactions.

### How to Contribute:
1. **Fork the repo** and throw us a Star ⭐!
2. **Create a feature branch**: `git checkout -b feature/your-awesome-idea`
3. **Commit your changes**: `git commit -m 'Added custom theming engine'`
4. **Push to the branch**: `git push origin feature/your-awesome-idea`
5. **Open a Pull Request**, and we'll review it ASAP.

Feel free to open an Issue if you have a bug report or a feature request. Let's build the ultimate Linux AI sidekick together!

---

<p align="center">
  Made with ❤️ by the VOIDER Open Source Community
</p>
