import os
import dotenv
if dotenv.load_dotenv():
    print("Environment Variables Loaded Successfully")
else:
    print("Environment Variables Failed to Load")
    raise ValueError("No env variabled found")

from db import VectorStore, Neo4jGraphDB
from openai import OpenAI

vectordb = VectorStore(reset=True)
graphdb = Neo4jGraphDB(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    db_name=os.getenv("NEO4J_DATABASE")
)
# if not graphdb.driver.verify_connectivity():
#     print("Graph DB not Connected, check on console")
#     raise ValueError("GraphDB connection failed")

llm_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)