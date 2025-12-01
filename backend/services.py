import os
import dotenv
if dotenv.load_dotenv():
    print("Environment Variables Loaded Successfully")
else:
    print("Environment Variables Failed to Load")
    raise ValueError("No env variabled found")

import json
from openai import OpenAI
from typing import Optional
from pydantic import ValidationError

from db import VectorStore, Neo4jGraphDB
from prompts import ingestion_system_prompt, entity_extraction_system_prompt, relation_extraction_system_prompt, retrieval_system_prompt, reasoning_system_prompt
from prompts import baseline_system_prompt

reset = False # DO NOT CHANGE THIS (Making it True, will reset both vectorDB and graphDB)

os.makedirs("storage", exist_ok=True)
vectordb = VectorStore(reset=reset)
node_vectordb = VectorStore(metadata_path="storage/node.json", index_path="storage/node.faiss", reset=reset)
graphdb = Neo4jGraphDB(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    db_name=os.getenv("NEO4J_DATABASE"),
    reset=reset
)

llm_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

def create_llm_agent(system_prompt: Optional[str] = None):
    """
    Creates a LLM agent with a system prompt
    """
    if system_prompt and len(system_prompt) > 0:
        system_message = {"role": "system", "content": system_prompt}
    def call_llm(prompt: str, output_model = None, retries: int = 3):
        # Set system prompt
        if system_prompt and len(system_prompt) > 0:
            messages = [system_message]
        else: messages = []
        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )
        
        for attempt in range(retries):
            # Send messages
            completion = llm_client.chat.completions.create(
                # model="openai/gpt-oss-20b:free",
                # model="google/gemma-3-4b-it:free",
                model="x-ai/grok-4.1-fast:free",
                messages=messages
            )
            
            raw = completion.choices[0].message.content
            
            # If no model validation needed, just return raw
            if output_model is None:
                return raw
            
            # Validate output
            try:
                data = json.loads(raw)
                validated = output_model.model_validate(data)
                return validated
            
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Validation error (attempt {attempt+1}): {e}")
                continue  # retry
        raise ValueError("LLM failed to return valid output after retries.")
    return call_llm


# Create agents
ingestion_agent = create_llm_agent(system_prompt=ingestion_system_prompt)
entity_agent = create_llm_agent(system_prompt=entity_extraction_system_prompt)
relation_agent = create_llm_agent(system_prompt=relation_extraction_system_prompt)
retrieval_agent = create_llm_agent(system_prompt=retrieval_system_prompt)
reasoning_agent = create_llm_agent(system_prompt=reasoning_system_prompt)
baseline_agent = create_llm_agent(system_prompt=baseline_system_prompt)