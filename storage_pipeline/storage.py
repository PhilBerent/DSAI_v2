#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles Step 7: Data Storage and Indexing.

Persists chunks, embeddings, and graph data to their respective databases.
Assumes Neo4j and Pinecone connections are established and index exists.
Document store aspect is simplified here - chunk text/metadata stored with embeddings.
"""

import logging
from typing import List, Dict, Any
from pinecone import Index as PineconeIndex, Vector
from neo4j import Driver as Neo4jDriver
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Adjust path to import from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import required global modules
try:
    from globals import *
    from UtilityFunctions import *
    from DSAIParams import *
except ImportError as e:
    print(f"Error importing core modules (globals, UtilityFunctions, DSAIParams): {e}")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pinecone Storage ---
PINECONE_UPSERT_BATCH_SIZE = 100 # Match Pinecone limits/recommendations
MAX_PINECONE_WORKERS = 10 # Parallel uploads

def store_embeddings_pinecone(
    pinecone_index: PineconeIndex,
    embeddings: Dict[str, List[float]],
    chunks: List[Dict[str, Any]] # Pass chunks to get metadata
):
    """Stores text chunks and their embeddings in Pinecone.

    Args:
        pinecone_index: The initialized Pinecone index object.
        embeddings: Dictionary mapping chunk_id to embedding vector.
        chunks: List of chunk dictionaries to extract metadata.
    """
    logging.info(f"Starting storage of {len(embeddings)} embeddings/chunks to Pinecone index: {pinecone_index.name}")
    vectors_to_upsert = []
    chunks_dict = {chunk['chunk_id']: chunk for chunk in chunks}

    for chunk_id, vector in embeddings.items():
        if chunk_id not in chunks_dict:
            logging.warning(f"Chunk ID {chunk_id} found in embeddings but not in chunk list. Skipping.")
            continue

        chunk_data = chunks_dict[chunk_id]
        # Prepare metadata for Pinecone (must be JSON serializable, limited size)
        pinecone_metadata = {
            "text": chunk_data['text'][:20000], # Store truncated text (Pinecone limit)
            "document_id": chunk_data['metadata'].get('document_id', 'Unknown'),
            "sequence": chunk_data['metadata']['source_location'].get('sequence', -1),
            "structure_ref": chunk_data['metadata']['source_location'].get('structure_ref', 'Unknown')
            # Add other simple metadata if needed and within size limits
            # Complex analysis results are better queried from the graph DB
        }

        # Clean metadata: ensure values are simple types
        for key, value in pinecone_metadata.items():
            if isinstance(value, (dict, list)): # Basic check
                 try:
                      pinecone_metadata[key] = json.dumps(value)
                 except TypeError:
                      pinecone_metadata[key] = str(value) # Fallback to string

        vectors_to_upsert.append(Vector(id=chunk_id, values=vector, metadata=pinecone_metadata))

    # Upsert in batches using ThreadPoolExecutor for potential speedup
    total_upserted = 0
    with ThreadPoolExecutor(max_workers=MAX_PINECONE_WORKERS) as executor:
        futures = []
        for i in range(0, len(vectors_to_upsert), PINECONE_UPSERT_BATCH_SIZE):
            batch = vectors_to_upsert[i : i + PINECONE_UPSERT_BATCH_SIZE]
            futures.append(executor.submit(pinecone_index.upsert, vectors=batch))

        for future in as_completed(futures):
            try:
                result = future.result()
                total_upserted += result.upserted_count
                logging.info(f"Upserted batch result: {result}")
            except Exception as e:
                logging.error(f"Error during Pinecone batch upsert: {e}", exc_info=True)

    logging.info(f"Finished storing embeddings. Total upserted count reported by Pinecone: {total_upserted}")

# --- Neo4j Storage ---
NEO4J_BATCH_SIZE = 500 # Adjust based on transaction size limits/performance
MAX_NEO4J_WORKERS = 5

def execute_neo4j_write_batch(driver: Neo4jDriver, query: str, batch_data: List[Dict[str, Any]]):
    """Executes a batch of write operations in a single Neo4j transaction."""
    try:
        with driver.session() as session:
            # Use bookmarks for causal consistency if needed
            result = session.run(query, batch=batch_data)
            summary = result.consume()
            logging.debug(f"Neo4j batch write summary: {summary.counters}")
    except Exception as e:
        logging.error(f"Failed Neo4j batch write: {e}. Query: {query[:200]}... Data Count: {len(batch_data)}", exc_info=True)
        # Consider raising the error or implementing more robust error handling/retries
        # raise

def store_graph_data_neo4j(
    neo4j_driver: Neo4jDriver,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]]
):
    """Stores nodes and edges in Neo4j.

    Uses UNWIND for batching node and edge creation.
    Creates constraints for uniqueness if they don't exist.
    """
    logging.info(f"Starting storage of {len(nodes)} nodes and {len(edges)} edges to Neo4j.")

    # --- Ensure Constraints (Run once or check existence) ---
    # Important for preventing duplicate nodes and for MERGE performance
    constraint_queries = [
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
        "CREATE CONSTRAINT structure_id IF NOT EXISTS FOR (s:Structure) REQUIRE s.structure_id IS UNIQUE", # Generic Structure
        # Add constraints for specific structure types if used e.g., Chapter
        "CREATE CONSTRAINT character_name IF NOT EXISTS FOR (c:Character) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
        "CREATE CONSTRAINT organization_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE"
        # Add others as needed
    ]
    try:
        with neo4j_driver.session() as session:
            for query in constraint_queries:
                try:
                    session.run(query)
                    logging.info(f"Applied or verified constraint: {query.split(' FOR ')[0]}")
                except Exception as e:
                    # Ignore errors if constraint already exists, log others
                    if "already exists" not in str(e).lower():
                         logging.error(f"Failed to apply constraint: {query}. Error: {e}", exc_info=True)
                    else:
                         logging.debug(f"Constraint likely already exists: {query.split(' FOR ')[0]}")
    except Exception as e:
        logging.error(f"Could not connect to Neo4j to apply constraints: {e}", exc_info=True)
        raise

    # --- Batch Node Creation --- #
    # Using MERGE to avoid creating duplicate nodes if the pipeline is re-run
    # Requires constraints to be set on the identifying property (e.g., name, document_id)
    node_query = """
    UNWIND $batch AS node_data
    // Determine the merge key based on label
    CALL {
        WITH node_data
        WITH node_data,
             CASE node_data.labels[0]
                 WHEN 'Document' THEN {document_id: node_data.properties.document_id}
                 WHEN 'Structure' THEN {structure_id: node_data.properties.structure_id}
                 WHEN 'Chapter' THEN {structure_id: node_data.properties.structure_id} // Example
                 WHEN 'Character' THEN {name: node_data.properties.name}
                 WHEN 'Location' THEN {name: node_data.properties.name}
                 WHEN 'Organization' THEN {name: node_data.properties.name}
                 ELSE {id: node_data.id} // Fallback, less ideal
             END AS merge_props
        // MERGE using dynamic labels and properties
        MERGE (n) WHERE n[keys(merge_props)[0]] = merge_props[keys(merge_props)[0]]
        // Set labels dynamically (can be slow, alternative is multiple queries per label)
        CALL apoc.create.addLabels(n, node_data.labels) YIELD node
        // Set properties
        SET node += node_data.properties
        RETURN count(*) AS dummy // Return something to make CALL work
    }
    RETURN count(*) AS nodes_processed
    """
    # Note: The above node query uses APOC for dynamic labels which might be slow.
    # A faster alternative is separate MERGE queries for each Node Label type.
    # Example for Character:
    # node_query_char = """
    # UNWIND $batch AS node_data
    # MERGE (c:Character {name: node_data.properties.name})
    # SET c += node_data.properties
    # """

    with ThreadPoolExecutor(max_workers=MAX_NEO4J_WORKERS) as executor:
        futures = []
        for i in range(0, len(nodes), NEO4J_BATCH_SIZE):
            batch = nodes[i : i + NEO4J_BATCH_SIZE]
            # If using separate queries per label, filter batch here
            futures.append(executor.submit(execute_neo4j_write_batch, neo4j_driver, node_query, batch))
        for future in as_completed(futures):
            try:
                future.result() # Wait for completion and check for exceptions
            except Exception as e:
                 logging.error(f"Neo4j node write future failed: {e}", exc_info=True)

    logging.info("Finished processing node storage batches.")

    # --- Batch Edge Creation --- #
    # Using MERGE on relationships requires matching both start/end nodes and properties
    # It's often simpler to use CREATE if sure duplicates won't be made, or MATCH+CREATE
    edge_query = """
    UNWIND $batch AS edge_data
    // Find start and end nodes using their unique IDs/properties
    CALL {
        WITH edge_data
        MATCH (start) WHERE start.document_id = edge_data.start_node_id OR start.structure_id = edge_data.start_node_id OR start.name = edge_data.start_node_id // Adapt matching logic based on ID format
        MATCH (end) WHERE end.document_id = edge_data.end_node_id OR end.structure_id = edge_data.end_node_id OR end.name = edge_data.end_node_id // Adapt matching logic based on ID format
        RETURN start, end, edge_data
    }
    // Use CREATE for simplicity, assuming pipeline ensures logical uniqueness
    // For MERGE: MERGE (start)-[r:TYPE {prop: val}]->(end)
    CALL apoc.create.relationship(start, edge_data.type, edge_data.properties, end) YIELD rel
    RETURN count(rel) AS relationships_created
    """
    # Adapt the MATCH clauses based on how node IDs were constructed in graph_builder.py
    # The current MATCH is very basic and might be inefficient/incorrect.
    # It assumes start/end_node_id might be document_id, structure_id, or name.
    # Better: Pass the actual unique property used for merging nodes (e.g., name for Character)

    # Revised edge_query using passed identifiers directly (assuming they are correct)
    # This requires graph_builder to pass the correct unique identifiers used for merging
    edge_query_revised = """
    UNWIND $batch AS edge_data
    // Match nodes based on the primary ID used in MERGE
    CALL {
        WITH edge_data
        // Match start node (adapt based on type)
        CALL { 
             WITH edge_data 
             MATCH (start {document_id: edge_data.start_node_id}) RETURN start 
             UNION ALL WITH edge_data 
             MATCH (start {structure_id: edge_data.start_node_id}) RETURN start 
             UNION ALL WITH edge_data 
             MATCH (start {name: edge_data.start_node_id}) RETURN start // Assumes name is unique key for entities
        } 
        WITH start, edge_data
        // Match end node (adapt based on type)
        CALL { 
            WITH edge_data 
            MATCH (end {document_id: edge_data.end_node_id}) RETURN end 
            UNION ALL WITH edge_data 
            MATCH (end {structure_id: edge_data.end_node_id}) RETURN end 
            UNION ALL WITH edge_data 
            MATCH (end {name: edge_data.end_node_id}) RETURN end // Assumes name is unique key for entities
        }
        RETURN start, end, edge_data WHERE start IS NOT NULL AND end IS NOT NULL // Ensure nodes were found
    }
    // Use CREATE or MERGE as appropriate. CREATE is simpler if pipeline handles duplicates.
    CREATE (start)-[r:PLACEHOLDER_REL_TYPE]->(end)
    // Set type and properties dynamically
    CALL apoc.create.setRelType(r, edge_data.type)
    CALL apoc.create.setRelProperties(r, edge_data.properties)
    RETURN count(r) AS relationships_created
    """
    # NOTE: The revised edge query also uses APOC and complex matching logic.
    # It might be necessary to create edges in separate batches per relationship type
    # with specific MATCH clauses for the start/end node labels and properties.
    # Example for INTERACTED_WITH between Characters:
    # edge_query_interacted = """
    # UNWIND $batch AS edge_data
    # MATCH (start:Character {name: edge_data.start_node_id})
    # MATCH (end:Character {name: edge_data.end_node_id})
    # CREATE (start)-[r:INTERACTED_WITH]->(end)
    # SET r = edge_data.properties
    # """

    with ThreadPoolExecutor(max_workers=MAX_NEO4J_WORKERS) as executor:
        futures = []
        for i in range(0, len(edges), NEO4J_BATCH_SIZE):
            batch = edges[i : i + NEO4J_BATCH_SIZE]
            # If using separate queries per rel type, filter batch here
            futures.append(executor.submit(execute_neo4j_write_batch, neo4j_driver, edge_query_revised, batch))
        for future in as_completed(futures):
             try:
                future.result()
             except Exception as e:
                 logging.error(f"Neo4j edge write future failed: {e}", exc_info=True)

    logging.info("Finished processing edge storage batches.")
    logging.info("Neo4j graph storage complete.")

# --- Document Store (Simplified) ---
# In this setup, the primary text and basic metadata are stored in Pinecone's metadata.
# A dedicated Document Store (like Elasticsearch or MongoDB) could be added here
# to store the full chunk text and ALL extracted metadata if Pinecone's limits
# are too restrictive or richer metadata querying is needed outside the graph.

def store_chunk_metadata_docstore(chunks_with_analysis: List[Dict[str, Any]]):
    """Placeholder for storing detailed chunk metadata in a dedicated document store."""
    logging.warning("Document Store functionality is currently simplified; core text/metadata stored in Pinecone.")
    # Example: Iterate chunks_with_analysis and insert into Elasticsearch/MongoDB
    # for chunk in chunks_with_analysis:
    #     doc_store_client.index(index="chunks", id=chunk['chunk_id'], document=chunk)
    pass 