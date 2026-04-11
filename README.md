# 🎓 Multi-Agent Generative AI Knowledge Assistant

> A production-structured multi-agent AI system built for student projects,
> portfolios, and interview showcases. Features RAG, Text-to-SQL, and a clean
> Streamlit chat UI — all wired together through an LLM-powered router.

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User / Chat UI                    │
│                  (Streamlit — ui/app.py)             │
└─────────────────────────┬───────────────────────────┘
                          │ user query
                          ▼
┌─────────────────────────────────────────────────────┐
│                   Router Agent                       │
│   Classifies intent → "rag" | "sql" | "general"    │
└──────────┬──────────────┬──────────────┬────────────┘
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │   RAG    │   │   SQL    │   │ General  │
    │  Agent   │   │  Agent   │   │  Agent   │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
    ┌────▼─────┐   ┌────▼─────┐       │
    │  FAISS   │   │  SQLite  │       │
    │ Vector DB│   │   DB     │       │
    └──────────┘   └──────────┘       │
           │              │              │
           └──────────────┴──────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  Response Agent                      │
│         Formats, cites sources, adds badges          │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
                   Final answer to user
```

## 🗂️ Project Structure

```
multi_agent_assistant/
│
├── agents/                     # All AI agents
│   ├── base_agent.py           # Abstract base class (ABC pattern)
│   ├── router_agent.py         # Intent classification + routing
│   ├── rag_agent.py            # Retrieval-Augmented Generation
│   ├── sql_agent.py            # Natural language → SQL → results
│   ├── general_agent.py        # Fallback LLM agent
│   └── response_agent.py       # Formats final output (no LLM call)
│
├── rag/                        # RAG pipeline components
│   ├── document_processor.py   # PDF/DOCX/TXT loading + chunking
│   └── vector_store.py         # FAISS wrapper (add / search / persist)
│
├── database/                   # Database layer
│   └── db_setup.py             # SQLite schema + sample data seeding
│
├── ui/                         # Frontend
│   └── app.py                  # Streamlit chat interface
│
├── utils/                      # Shared utilities
│   ├── config.py               # Centralised config from .env
│   ├── logger.py               # Structured logging (loguru)
│   ├── llm_client.py           # Provider-agnostic LLM factory
│   └── memory.py               # Cross-agent conversation memory
│
├── tests/                      # Unit tests
│   └── test_agents.py          # Mocked tests (no API calls needed)
│
├── sample_docs/                # Sample knowledge base documents
│   └── ai_knowledge_base.txt   # AI/ML reference document for RAG demo
│
├── logs/                       # Auto-created at runtime
├── main.py                     # Orchestrator + CLI demo entry point
├── requirements.txt            # All dependencies
└── .env.example                # Environment variable template
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd multi_agent_assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)
```

### 3. Seed the Database

```bash
python database/db_setup.py
```

### 4. Run the App

```bash
# Option A — Streamlit UI (recommended)
streamlit run ui/app.py

# Option B — CLI demo
python main.py
```

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 🤖 Agent Details

### RouterAgent
- Uses zero-shot LLM classification (temperature=0 for determinism)
- Returns one of: `rag`, `sql`, `general`
- Falls back to `general` for unexpected outputs

### RAGAgent
- Embeds user query using `sentence-transformers/all-MiniLM-L6-v2` (runs locally)
- Searches FAISS for top-K relevant chunks
- Injects chunks into a grounded prompt — model is instructed not to hallucinate
- Returns answer with source citations

### SQLAgent
- Injects the database schema into a prompt
- LLM generates a SQLite SELECT query (temperature=0)
- Validates SQL — blocks any non-SELECT operations
- Executes and returns results as a Markdown table

### GeneralAgent
- Plain conversational LLM with conversation history injected
- Handles greetings, general knowledge, follow-ups

### ResponseAgent
- Pure Python (no LLM call) — formats the raw `AgentResponse` object
- Adds emoji badges, source citations, SQL code blocks

---

## 💡 Example Queries

| Query | Agent | What happens |
|---|---|---|
| `Show students with marks above 80` | SQL | Generates & runs SQL JOIN query |
| `Average score per subject` | SQL | Aggregation query with GROUP BY |
| `Top 3 students overall` | SQL | ORDER BY + LIMIT |
| `What is RAG?` | RAG | Retrieves from uploaded docs |
| `Summarise the PDF` | RAG | Retrieves all top-K chunks |
| `What is machine learning?` | General | Plain LLM response |
| `Hello!` | General | Greeting response |

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | Anthropic Claude (or OpenAI) | Swappable via .env |
| Framework | LangChain | Agent/chain orchestration |
| Embeddings | sentence-transformers | Local, free, no API key |
| Vector DB | FAISS | Fast, in-memory, beginner-friendly |
| Database | SQLite | Zero-config, perfect for demos |
| UI | Streamlit | Rapid Python-native chat UI |
| Logging | loguru | Clean structured logs |
| Validation | Pydantic | Typed agent responses |
| Testing | pytest + unittest.mock | No API calls in tests |

---

## 🎯 Interview Talking Points

1. **Architecture Pattern**: Router → Specialist Agent → Formatter follows the
   Command pattern and SRP (Single Responsibility Principle).

2. **Provider Abstraction**: `llm_client.py` is a factory — the rest of the system
   is LLM-provider agnostic. Swapping Claude for GPT-4o is one env-var change.

3. **Security in Text-to-SQL**: The SQL agent validates that every generated query
   is a SELECT before execution. In production, add a read-only DB user.

4. **RAG vs Fine-tuning**: RAG is preferred for dynamic/private knowledge because
   it doesn't require retraining and sources can be cited and updated.

5. **Memory Design**: Shared `ConversationMemory` singleton with a rolling window
   prevents unbounded context growth while preserving continuity.

6. **Testing Strategy**: All agent tests mock the LLM layer — tests run in
   milliseconds with zero API costs, enabling fast CI/CD.

---

## 🚀 Production Enhancements (for discussion)

- Replace FAISS with **Pinecone** or **Weaviate** for persistence + scale
- Add **authentication** (Streamlit Auth, Auth0)
- Replace SQLite with **PostgreSQL** + read-only user
- Add **streaming** responses with `st.write_stream()`
- Add **tool calling** so the LLM can decide to use multiple agents per query
- Add **evaluation** metrics (RAGAS for RAG quality, exact-match for SQL)
- Deploy on **AWS/GCP** using Docker + a Streamlit Cloud / FastAPI backend

---

## 📄 License

MIT License — free to use for educational and portfolio purposes.
