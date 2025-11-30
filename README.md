# GraphIQ

DS 246: Agentic &amp; Generative AI Project

GraphIQ is an agentic AI system designed to extract knowledge from documents, represent it as a graph, and enable context-aware question answering using both vector search and structured graph reasoning.

At a high level, the project ingests PDFs or text documents, breaks them into meaningful chunks, summarizes them, extracts entities and relationships, stores everything in Neo4j, and exposes APIs for exploration and chat-based interaction.

## API KEYS

Make a `.env` in root and put the following values in it.

```bash
NEO4J_URI=<Value>
NEO4J_USERNAME=<Value>
NEO4J_PASSWORD=<Value>
NEO4J_DATABASE=<Value>
AURA_INSTANCEID=<Value>
AURA_INSTANCENAME=<Value>
OPENROUTER_API_KEY=<Value>
GEMINI_API_KEY=<Value>
```

## How to use Openrouter and Gemini

1. Go to [https://openrouter.ai/models?max_price=0](OpenRouter) and pick whichever model is required, copy it's 'model_id'.
2. Check `openrouter.ipynb` on how to run chat completetion.
3. For embedding code blocks, don't change `model` parameter.
4. You can check documentation for [https://openrouter.ai/docs/quickstart#using-the-openai-sdk](Openrouter) and [https://ai.google.dev/gemini-api/docs/embeddings](Gemini Embedding).
5. You can use Gemini for generation as well, just make sure the code produces appropriate output.

## How to use Neo4j

1. Check `neo4j.ipynb` for some examples
2. They are all taken from [https://neo4j.com/docs/python-manual/current/query-simple/](Neo4j Documentation)
3. Use `with GraphDatabase.driver(URI, auth=AUTH) as driver:` always before running any query.

## How to run backend

It is recommended to make a virtual environment.

```cmd
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## How to run frontend

Make sure backend is running as frontend takes some information from backend.

```cmd
cd frontend
pnpm install
pnpm dev
```

## Backend File structure

```pgsql
GraphIQ/
├── backend/
│   │
│   ├── main.py                 # Fastapi endpoints and logic are defined here
│   ├── services.py             # VectorDB, GraphDB classes and 
│   │                             clients for Openrouter and Gemini are initialized here
│   ├── db.py                   # VectorDB and GraphDB class definitions
│   ├── utils.py                # Extra functions required for chunking, context formatting
│   ├── schema.py               # Pydantic schemas for some agents
│   ├── prompts.py              # All system prompts and user prompts are defined here
│   ├── agents.py               # All core agentic workflows are defined here
│
├── frontend/
│   ├── client
│   │   ├── components
│   │   │   ├── ChatPane.tsx    # Logic and structure of chatbot
│   │   │   ├── GraphPane.tsx   # Logic and structure of how the graph is displayed
│
├── test/                       # Some test files used to verify agent logic
├── .gitignore
├── neo4j.ipynb                 # Some sample code for neo4j graph SB
├── openrouter.ipynb            # Some sample code for openrouter and gemini embedding model
├── README.md
├── requirements.txt            # Python library requirements
├── start.bat                   # Can be used to run both frontend and backend directly, 
│                                 given that node_modules are installed and venv is active
├── .env                        # Needs to be created. Will be used for all keys
```
