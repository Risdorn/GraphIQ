import json
from pydantic import ValidationError

from typing import Optional
from prompts import entity_relation_system_prompt

from services import llm_client, vectordb

def create_llm_agent(system_prompt: Optional[str] = None):
    """
    Creates a LLM agent with a system prompt
    """
    if system_prompt:
        system_message = {"role": "system", "content": system_prompt}
    def call_llm(prompt: str, output_model = None, retries: int = 3):
        # Set system prompt
        if system_prompt:
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
entity_relation_agent = create_llm_agent(system_prompt=entity_relation_system_prompt)


# Test, DO NOT UNCOMMENT
# from agents import extract_entities_relationship
# chunk_metadata = {
#     "chunk_id": 0,
#     "text": "Gradient Descent is used to optimize neural networks...",
#     "summary": "Gradient Descent and Neural Networks"
# }
# chunk_metadata["chunk_id"] = vectordb.add(chunk_metadata["text"], chunk_metadata["summary"])

# extract_entities_relationship(chunk_metadata, entity_relation_agent)