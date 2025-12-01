import re
import time

from services import vectordb, graphdb, node_vectordb
from services import ingestion_agent, entity_agent, relation_agent, retrieval_agent, reasoning_agent, baseline_agent

from prompts import entity_extraction_prompt, relation_extraction_prompt, ingestion_prompt, retrieval_prompt

from schema import SummariesOutput, EntitiesOutput, RelationsOutput, RetrievalEntitiesOutput

from utils import paragraph_chunking
from utils import format_vector_context, format_graph_context

def ingest_file(text, batch_size=10):
    # 1. Normalize unicode, remove non-printables, fix spacing
    text = text.replace('\u00A0', ' ')
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\t+', ' ', text)
    # Remove repeated blank lines
    text = re.sub(r"\n{3,}", '\n\n', text)
    # Trim
    text = text.strip()

    # 2. chunk
    chunks = paragraph_chunking(text=text, max_tokens=512)
    print(f"Got {len(chunks)} Chunks")

    # 3. summarize and embed
    chunks_metadata = []
    total_chunks = len(chunks)
    start = time.time()
    for i in range(0, total_chunks, batch_size):
        # use the chunk index as the chunk_id
        print(f"Running {i}-{i+batch_size} chunks")
        chunk_texts = chunks[i:i+batch_size]
        prompt = ingestion_prompt + f"\n\n{chunk_texts}"
        summaries = ingestion_agent(prompt=prompt, output_model=SummariesOutput)
        summaries = summaries.summaries
        chunk_ids = vectordb.batch_add(chunk_texts, summaries)
        docs = [{
            'chunk_id': chunk_id,
            'text': c,
            'summary': summary,
        } for chunk_id, c, summary in zip(chunk_ids, chunk_texts, summaries)]
        chunks_metadata.extend(docs)
    print(f"Chunk extraction took {time.time() - start:.2f}s")
    print("Added to vector DB")
    return chunks_metadata

def extract_entities_relationship(chunk_metadata):
    """
    Perform Entity and Relationship extraction
    
    input of the form:
    {
        "chunk_id": int,
        "text": str,
        "summary": str
    }
    """
    # 0. Batch chunks
    start = time.time()
    text = "Chunk Texts:\n\n"
    for metadata in chunk_metadata:
        text += f"Chunk {metadata['chunk_id']}: {metadata['text']}\n"
        
    # 1. Extract entities
    print("Extracting Entities")
    prompt = f"{entity_extraction_prompt}\n\n{text}"
    entities = entity_agent(prompt=prompt, output_model=EntitiesOutput)
    entities_nodes = entities.entities
    entities = list(entities_nodes.keys())
    
    # 2. Extract Relationships
    print("Extracting Relations")
    existing_nodes = []
    for ent in entities:
        exist = node_vectordb.search(ent, top_k=5)
        exist = exist[2]
        exist = [e["text"] for e in exist]
        existing_nodes.extend(ent)
        
    prompt = f"{relation_extraction_prompt}\n\n Existing Entities: {existing_nodes}\nText Entities: {entities}\n{text}"
    relations = relation_agent(prompt=prompt, output_model=RelationsOutput)
    relations = relations.relations
    
    # 3. Deduplicate and remove similar entities
    mapping = {}
    for ent in entities:
        matches = graphdb.search_nodes(ent)
        if len(matches) > 0:
            mapping[ent] = matches[0]
        else:
            mapping[ent] = ent

    canon_entities = list(set(mapping.values()))
    # reverse mapping to properly get which chunk id belongs to which canon entity
    reverse_mapping = {k: [] for k in canon_entities}
    for k, v in mapping.items():
        reverse_mapping[v].append(k)
    
    # 4. Store nodes in Graph DB
    new_nodes, old_nodes = 0, 0
    chunk_id_to_entity = {} # Will be required for vectordb update
    for node in canon_entities:
        chunk_ids = []
        # Get all chunks associated with this node
        for ent in reverse_mapping[node]:
            chunk_ids.extend(entities_nodes[ent])
        chunk_ids = [int(x) for x in chunk_ids]
        # Map this node to all chunks
        for cid in chunk_ids:
            if cid not in chunk_id_to_entity: chunk_id_to_entity[cid]= []
            chunk_id_to_entity[cid].append(node)
        # Add in GraphDB
        if not graphdb.node_exists(node):
            new_nodes += 1
            graphdb.create_node(name=node, node_type="concept", chunk_ids=chunk_ids)
            node_vectordb.add(node, summary="")
        else:
            old_nodes += 1
            graphdb.add_chunks_to_node(name=node, chunk_ids=chunk_ids)
    
    # 5. Store Relationship in GraphDB
    new_relationship_created = 0
    for rel in relations:
        created = graphdb.create_relation_safe(
            mapping[rel.source],
            rel.relation,
            mapping[rel.target]
        )
        if created: new_relationship_created += 1
    
    # 6. Add nodes to vector DB metadata
    for cid, nodes in chunk_id_to_entity.items():
        vectordb.update(cid, nodes)
    
    all_ids = ""
    for metadata in chunk_metadata:
        all_ids += f", {metadata['chunk_id']}"
    print(f"""
=======================================================
For chunks {all_ids}:
New Nodes Created: {new_nodes},
Old Nodes Updated: {old_nodes},
New Relations added: {new_relationship_created}
Extraction took {time.time()-start:.2f}s
=======================================================
          """)

def retrieval(query, top_k = 5, top_nodes = 3):
    vectordb_data = []
    graphdb_data = []
    # Get most appropriate entities
    prompt = retrieval_prompt + "\n\n" + query
    entities = retrieval_agent(prompt=prompt, output_model=RetrievalEntitiesOutput)
    entities = entities.entities
    # Get most relevant node_hits
    seed_nodes = []
    seed_node_names = set()
    for name in entities:
        node_hits = node_vectordb.search(query=name, top_k=top_k)
        for distance, metadata in zip(node_hits[0], node_hits[2]):
            node_name = metadata["text"]
            if node_name in seed_node_names: continue
            seed_node_names.add(node_name)
            # Check if the node is directly present in user query
            match_score = 1.0 if node_name in entities else 0.0
            # Check how well connected the node is
            graph_centrality_score = graphdb.degree_in_score(node_name)
            seed_nodes.append(
                {
                    "node": metadata["text"],
                    "score": 0.55 * (1-distance) + 0.35 * match_score + 0.1 * graph_centrality_score
                }
            )
    
    # Sort and pick top 3
    selected = sorted(seed_nodes, key=lambda x: x["score"], reverse=True)
    core_nodes = selected[:top_nodes]
    
    # Neighbor nodes
    text_chunks = []
    summary_chunks = []
    neighbor_nodes = set()
    neighbor_node_names = set()
    for node_dict in core_nodes:
        # Add text chunk ids 
        text_chunks.extend(graphdb.get_chunks_for_node(node_dict["node"]))
        # One hop away
        nodes = graphdb.get_one_hop(node_dict["node"])
        for node in nodes:
            text = node_dict["node"] + "->" + node[0] + "->" + node[1]
            if text in neighbor_nodes: continue # Check if START->REL->END is the exact same
            if node[1] not in neighbor_node_names: # Check if we have already added chunk_ids
                neighbor_node_names.add(node[1])
                summary_chunks.extend(graphdb.get_chunks_for_node(node[1]))
            neighbor_node_names.add(node[1])
            neighbor_nodes.add(text)
    
    # Get data from vectordb
    text_chunks = list(set(text_chunks))
    summary_chunks = list(set(summary_chunks))
    # Direct match from vectordb
    vector_hits = vectordb.search(query=query, top_k=top_k)
    for metadata in vector_hits[2]:
        vectordb_data.append({
            "chunk_id": metadata["chunk_id"],
            "text": metadata["text"],
        })
    # All text
    for chunk_id in text_chunks:
        metadata = vectordb.get(chunk_id)
        vectordb_data.append({
            "chunk_id": metadata["chunk_id"],
            "text": metadata["text"]
        })
    # Only summary
    for chunk_id in summary_chunks:
        metadata = vectordb.get(chunk_id)
        vectordb_data.append({
            "chunk_id": metadata["chunk_id"],
            "summary": metadata["summary"]
        })
    # Graphdb data, just the START->REL->END data
    graphdb_data = list(neighbor_nodes)
    return {"query": query, "vector_context": vectordb_data, "graph_context": graphdb_data}

def reasoning_insights(query, vector_context, graph_context):
    vector_ctx = format_vector_context(vector_context)
    graph_ctx = format_graph_context(graph_context)
    
    context = f"""
[VECTOR STORE RESULTS]
{vector_ctx}

[GRAPH DB RESULTS]
{graph_ctx}
"""
    context = context.strip()
    prompt = f"""You are given the following retrieved context and a user question.\n\n"
=== RETRIEVED CONTEXT START ===\n
{context}\n
=== RETRIEVED CONTEXT END ===\n\n
User question: {query}"""

    answer = reasoning_agent(prompt=prompt)
    return {"query": query, "answer": answer}

def baseline(query, top_k=5):
    hits = vectordb.search(query=query, top_k=top_k)
    context = ""
    for i, metadata in enumerate(hits[2]):
        context += f"Result {i+1}: {metadata['text']}"
    prompt = f"""You are given the following retrieved context and a user question.\n\n"
=== RETRIEVED CONTEXT START ===\n
{context}\n
=== RETRIEVED CONTEXT END ===\n\n
User question: {query}"""
    answer = baseline_agent(prompt=prompt)
    return {"query": query, "answer": answer}