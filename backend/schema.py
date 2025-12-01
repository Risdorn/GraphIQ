from pydantic import BaseModel
from typing import List, Dict

class EntitiesOutput(BaseModel):
    entities: Dict[str, List[str]]

class Relation(BaseModel):
    source: str
    relation: str
    target: str

class RetrievalEntitiesOutput(BaseModel):
    entities: List[str]

class RelationsOutput(BaseModel):
    relations: List[Relation]

class SummariesOutput(BaseModel):
    summaries: List[str]
