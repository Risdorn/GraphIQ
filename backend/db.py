import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase

from google import genai


class VectorStore:
    """
    Append-only vector store backed by FAISS + metadata list.
    Uses the FAISS index position as the global ID.
    """

    def __init__(self, metadata_path: str = "storage/metadata.json", index_path: str = "storage/index.faiss",
                 output_dim: int = 3072, reset: bool = False):
        self.embedding_model = genai.Client()
        self.output_dim = output_dim
        self.metadata_path = metadata_path
        self.index_path = index_path

        # Create or load FAISS index
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

        self._load_or_initialize(reset)

    def _load_or_initialize(self, reset):
        """
        Loads FAISS index + metadata if they exist, else initializes new ones.
        """
        metadata_exists = os.path.exists(self.metadata_path)
        index_exists = os.path.exists(self.index_path)

        if metadata_exists and index_exists and not reset:
            # Load metadata
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            # Load FAISS index
            self.index = faiss.read_index(self.index_path)

            print(f"[VectorStore] Loaded existing store with {len(self.metadata)} chunks.")

        else:
            # Create new FAISS index
            self.index = faiss.IndexFlatL2(self.output_dim)
            self.metadata = []

            self._save()  # create empty files
            print("[VectorStore] Initializing new vector store.")

    def add(self, text: str, summary: str, node: Optional[List[str]] = None) -> int:
        """
        Add a new chunk to the vector store.
        Returns the FAISS index ID (used as unique chunk_id).
        """
        # 1. embed text
        result = self.embedding_model.models.embed_content(
            model="gemini-embedding-001",
            contents=text)
        embedding = result.embeddings[0].values
        embedding = np.array(embedding).astype("float32").reshape((1,-1))
        embedding = embedding / np.sqrt((embedding**2).sum(axis=1, keepdims=True))
        embedding = embedding.astype("float32")

        # 2. add to faiss
        self.index.add(embedding)

        # 3. add metadata
        chunk_metadata = {
            "chunk_id": len(self.metadata), # FAISS index position
            "text": text,
            "summary": summary,
            "node": node,   # None or a List of string like ["Transformer"]
        }
        self.metadata.append(chunk_metadata)

        # 4. save
        self._save()

        return chunk_metadata["chunk_id"]

    def search(self, query: str, top_k: int = 5):
        """
        Search FAISS using a query string.
        Returns (distances, indices, metadata_objects)
        """

        if len(self.metadata) == 0:
            return [], [], []

        # embed query
        result = self.embedding_model.models.embed_content(
            model="gemini-embedding-001",
            contents=query)
        query_emb = result.embeddings[0].values
        query_emb = np.array(query_emb).astype("float32").reshape((1,-1))
        query_emb = query_emb / np.sqrt((query_emb**2).sum(axis=1, keepdims=True))
        query_emb = query_emb.astype("float32")

        # search in faiss
        distances, indices = self.index.search(query_emb, top_k)

        # fetch metadata for results
        results_metadata = [self.metadata[i] for i in indices[0]]

        return distances[0].tolist(), indices[0].tolist(), results_metadata

    def get(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve metadata by FAISS index.
        """
        return self.metadata[idx]
    
    def update(self, idx: int, node: List[str]) -> None:
        """
        Update the node associated with the chunk
        """
        if self.metadata[idx]["node"] is not None:
            self.metadata[idx]["node"].extend(node)
            self.metadata[idx]["node"] = list(set(self.metadata[idx]["node"]))
        else:
            self.metadata[idx]["node"] = node
        self._save()
        return "New nodes assigned"

    def _save(self):
        """
        Saves FAISS index + metadata list.
        """

        # Save metadata
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)

    def __len__(self):
        return len(self.metadata)


class Neo4jGraphDB:
    def __init__(self, uri: str, user: str, password: str, db_name: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.db_name = db_name

    # Runner
    def _run(self, query: str, params: dict = None, single: bool = True):
        with self.driver.session(database=self.db_name) as session:
            result = session.run(query, params or {})
            if single:
                return result.single()
            else:
                return list(result)

    # Operations
    def create_node(self, name: str, node_type: str, chunk_ids: List[int] = None) -> None:
        """
        Create a graph node with a name, type, and list of chunk_ids.
        """
        chunk_ids = chunk_ids or []
        query = """
        MERGE (n:Entity {name: $name})
        SET n.type = $node_type,
            n.chunk_ids = coalesce(n.chunk_ids, []) + $chunk_ids
        """
        self._run(query, {"name": name, "node_type": node_type, "chunk_ids": chunk_ids})

    def node_exists(self, name: str) -> bool:
        query = """
        MATCH (n:Entity {name: $name})
        RETURN n LIMIT 1
        """
        result = self._run(query, {"name": name})
        return result is not None

    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Return node properties as dict or None.
        """
        query = """
        MATCH (n:Entity {name: $name})
        RETURN n
        """
        record = self._run(query, {"name": name})
        return dict(record["n"]) if record else None

    def add_chunks_to_node(self, name: str, chunk_ids: List[int]) -> None:
        """
        Append chunk_ids to node's chunk list.
        """
        query = """
        MATCH (n:Entity {name: $name})
        SET n.chunk_ids = coalesce(n.chunk_ids, []) + $chunk_ids
        """
        self._run(query, {"name": name, "chunk_ids": chunk_ids})

    def search_nodes(self, search: str) -> List[str]:
        """
        Fuzzy search (case-insensitive contains).
        """
        query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($search)
        RETURN n.name AS name
        """
        result = self._run(query, {"search": search}, single=False)
        return [record["name"] for record in result]

    def create_relation(self, src: str, rel:str, dst: str):
        """
        Create a relationship: (src)-[rel]->(dst)
        """
        rel = rel.upper().replace(" ", "_")  # sanitize
        query = f"""
        MATCH (a:Entity {{name: $src}})
        MATCH (b:Entity {{name: $dst}})
        MERGE (a)-[r:{rel}]->(b)
        """
        self._run(query, {"src": src, "dst": dst})
    
    def any_relation_exists(self, src: str, dst: str):
        """
        Check if relation already exists
        """
        query = """
        MATCH (a:Entity {name: $src})-[r]->(b:Entity {name: $dst})
        RETURN r LIMIT 1
        """
        result = self._run(query, {"src": src, "dst": dst})
        return result is not None
    
    def create_relation_safe(self, src: str, rel:str, dst: str):
        """
        Create relation only when both nodes are not connected
        """
        if self.any_relation_exists(src=src, dst=dst):
            return False # Nothing new created
        self.create_relation(src=src, rel=rel, dst=dst)
        return True

    def get_chunks_for_node(self, name: str) -> List[int]:
        query = """
        MATCH (n:Entity {name: $name})
        RETURN n.chunk_ids AS chunk_ids
        """
        record = self._run(query, {"name": name})
        return record["chunk_ids"] if record else []
    
    def get_one_hop(self, name: str):
        query = """
        MATCH (n:Entity {name: $name})-[r]-> (out)
        RETURN type(r) AS rel, out.name AS out_node
        """
        records = self._run(query, {"name": name}, single=False)
        return [[r["rel"], r["out_node"]] for r in records]

    def __del__(self):
        self.driver.close()