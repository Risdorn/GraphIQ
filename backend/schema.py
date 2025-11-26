from pydantic import BaseModel
from typing import List

class EntitiesOutput(BaseModel):
    entities: List[str]

class Relation(BaseModel):
    source: str
    relation: str
    target: str

class RelationsOutput(BaseModel):
    relations: List[Relation]
