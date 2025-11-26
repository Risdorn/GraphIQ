# GraphIQ

DS 246: Agentic &amp; Generative AI Project

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

## How to run frontend

Make sure backend is running as frontend takes some information from backend.

```cmd
cd frontend
pnpm install
pnpm dev
```

## Running both backend and frontend

1. Make sure virtual environment is activated
2. Run `start.bat`
