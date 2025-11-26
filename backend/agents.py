from services import vectordb, graphdb

from prompts import entity_extraction_prompt, relation_extraction_prompt

from schema import EntitiesOutput, RelationsOutput

def extract_entities_relationship(chunk_metadata, entity_relation_agent):
    """
    Perform Entity and Relationship extraction
    
    input of the form:
    {
        "chunk_id": int,
        "text": str,
        "summary": str
    }
    """
    # 1. Extract entities
    prompt = f"{entity_extraction_prompt}\n\n {chunk_metadata['text']}"
    entities = entity_relation_agent(prompt=prompt, output_model=EntitiesOutput)
    entities = entities.entities
    
    # 2. Extract Relationships
    prompt = f"{relation_extraction_prompt}\n\n Entities: {entities}\nText: {chunk_metadata['text']}"
    relations = entity_relation_agent(prompt=prompt, output_model=RelationsOutput)
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
    
    # 4. Store nodes in Graph DB
    new_nodes, old_nodes = 0, 0
    for node in canon_entities:
        if not graphdb.node_exists(node):
            new_nodes += 1
            graphdb.create_node(name=node, node_type="concept", chunk_ids=[chunk_metadata["chunk_id"]])
        else:
            old_nodes += 1
            graphdb.add_chunks_to_node(name=node, chunk_ids=[chunk_metadata["chunk_id"]])
    
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
    vectordb.update(chunk_metadata["chunk_id"], canon_entities)
    
    print(f"""
=======================================================
For chunk {chunk_metadata["chunk_id"]}:
New Nodes Created: {new_nodes},
Old Nodes Updated: {old_nodes},
New Relations added: {new_relationship_created}
=======================================================
          """)
