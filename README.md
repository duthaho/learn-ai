# AI in Action

A hands-on curriculum for developers learning AI engineering — from zero to production.

Each module has two parts:
- **README** — deep documentation explaining concepts, mechanics, and best practices
- **project/** — a real mini-project you build step by step

## Roadmap

### Phase 1: Foundations

| # | Module | Project |
|---|--------|---------|
| 01 | [How LLMs Work](01-how-llms-work/) | Token Budget Calculator |
| 02 | [Prompt Engineering](02-prompt-engineering/) | Prompt Testing Workbench |
| 03 | [Embeddings & Vector Search](03-embeddings-vector-search/) | Semantic Search Engine |
| 04 | [The AI API Layer](04-ai-api-layer/) | LLM API Explorer |
| 05 | [Streaming & Real-Time AI](05-streaming-realtime-ai/) | Streaming Chat |

### Phase 2: Core AI Engineering

| # | Module | Project |
|---|--------|---------|
| 06 | [Tool Use & Function Calling](06-tool-use-function-calling/) | AI Assistant with Tools |
| 07 | [RAG](07-rag/) | RAG Q&A |
| 08 | [Structured Output](08-structured-output/) | Data Extractor |
| 09 | [Conversational AI & Memory](09-conversational-ai-memory/) | Memory Chatbot |
| 10 | Image & Multimodal AI *(coming soon)* | |

### Phase 3: Agents & Autonomy

| # | Module | Project |
|---|--------|---------|
| 11 | Building AI Agents *(coming soon)* | |
| 12 | Multi-Agent Systems *(coming soon)* | |
| 13 | Workflows & Chains *(coming soon)* | |
| 14 | AI Code Generation *(coming soon)* | |
| 15 | Evaluation & Testing *(coming soon)* | |

### Phase 4: Production AI

| # | Module | Project |
|---|--------|---------|
| 16 | AI Safety & Guardrails *(coming soon)* | |
| 17 | Caching & Cost Optimization *(coming soon)* | |
| 18 | Observability & Monitoring *(coming soon)* | |
| 19 | Advanced RAG *(coming soon)* | |
| 20 | Deployment Patterns *(coming soon)* | |

### Phase 5: Frontier & Specialization

| # | Module | Project |
|---|--------|---------|
| 21 | AI for Frontend Developers *(coming soon)* | |
| 22 | Voice & Audio AI *(coming soon)* | |
| 23 | MCP — Model Context Protocol *(coming soon)* | |
| 24 | Building AI Products *(coming soon)* | |

## Getting Started

### Prerequisites

- Python 3.11+
- An API key for at least one LLM provider

### Setup

```bash
git clone <repo-url> ai-in-action
cd ai-in-action

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

cp .env.example .env       # add your API key
pip install -r requirements.txt
```

## Module Format

Each module follows this structure:

```
XX-module-name/
├── README.md              # Deep documentation (concepts, diagrams, best practices)
├── quiz.md                # Self-assessment quiz
└── project/
    ├── README.md          # Step-by-step build instructions
    ├── start.py           # Starter code with TODOs — you fill these in
    └── solution.py        # Completed reference implementation
```

Read the README to understand the concepts. Then open `project/` and build.
