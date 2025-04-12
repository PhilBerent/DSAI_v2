#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions for debugging the stored data."""

import logging
import json
from typing import Optional, List, Dict, Any
from pinecone import Index as PineconeIndex
from neo4j import Driver as Neo4jDriver
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

# Assumes db_connections can be imported to get handles
# Adjust path if necessary or pass connections as arguments
# Need to adjust path relative to this file's location
# pipeline_dir = os.path.join(parent_dir, 'storage_pipeline')
# sys.path.insert(0, pipeline_dir)
# try:
#     from db_connections import get_pinecone_index, get_neo4j_driver
# except ImportError as e:
#     print(f"Error importing db_connections from sibling directory: {e}")
#     # Fallback if run differently?
#     try:
#          from storage_pipeline.db_connections import get_pinecone_index, get_neo4j_driver
#     except ImportError:
#          logging.error("Could not import db_connections for debugging utilities.")
#          # Define dummy functions or raise error
#          def get_pinecone_index(): raise NotImplementedError("DB connection failed")
#          def get_neo4j_driver(): raise NotImplementedError("DB connection failed")

# Use absolute import
try:
    from storage_pipeline.db_connections import get_pinecone_index, get_neo4j_driver
except ImportError as e:
    logging.error(f"Could not import db_connections using absolute path: {e}. Debugging utilities might fail.")
    # Define dummy functions or raise error if essential
    def get_pinecone_index(): raise NotImplementedError(f"DB connection failed due to import error: {e}")
    def get_neo4j_driver(): raise NotImplementedError(f"DB connection failed due to import error: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pinecone Debugging ---

def get_chunk_from_pinecone(chunk_id: str, include_vector: bool = False) -> Optional[str]:
    """Retrieves and formats data for a specific chunk ID from Pinecone."""
    try:
        pinecone_index = get_pinecone_index()
        fetch_response = pinecone_index.fetch(ids=[chunk_id])

        if not fetch_response or not fetch_response.vectors:
            logging.warning(f"Chunk ID {chunk_id} not found in Pinecone index {pinecone_index.name}.")
            return None

        vector_data = fetch_response.vectors.get(chunk_id)
        if not vector_data:
            logging.warning(f"Data for Chunk ID {chunk_id} not found in Pinecone fetch response.")
            return None

        output = f"--- Pinecone Data for Chunk ID: {chunk_id} ---\n"
        output += "Metadata:\n"
        if vector_data.metadata:
            # Pretty print metadata
            metadata_str = json.dumps(vector_data.metadata, indent=2)
            output += f"{metadata_str}\n"
        else:
            output += "  (No metadata found)\n"

        if include_vector:
            output += "\nVector (first 10 dimensions):\n"
            if vector_data.values:
                output += f"  {vector_data.values[:10]}... (Total dimensions: {len(vector_data.values)})\n"
            else:
                output += "  (No vector values found)\n"

        return output

    except Exception as e:
        logging.error(f"Error retrieving chunk {chunk_id} from Pinecone: {e}")
        return f"Error retrieving chunk {chunk_id} from Pinecone: {e}"

# --- Neo4j Debugging ---

def get_node_by_id(node_id: str) -> Optional[str]:
    """Retrieves a node and its properties from Neo4j by its assumed unique ID.
       Note: Assumes node_id is unique across types or matches specific properties.
    """
    query = """
    MATCH (n)
    WHERE n.document_id = $node_id OR n.structure_id = $node_id OR n.name = $node_id OR n.id = $node_id
    RETURN n
    LIMIT 1
    """
    # This query is generic and might be slow. Specific queries per type are better.
    # Example specific query for Character:
    # query = "MATCH (n:Character {name: $node_id}) RETURN n LIMIT 1"

    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            if record and record["n"]:
                node_data = record["n"]
                output = f"--- Neo4j Node Data for ID: {node_id} ---\n"
                output += f"Labels: {list(node_data.labels)}\n"
                output += "Properties:\n"
                props_str = json.dumps(dict(node_data.items()), indent=2)
                output += f"{props_str}\n"
                return output
            else:
                logging.warning(f"Node with ID matching '{node_id}' not found in Neo4j.")
                return None
    except Exception as e:
        logging.error(f"Error retrieving node {node_id} from Neo4j: {e}")
        return f"Error retrieving node {node_id} from Neo4j: {e}"
    finally:
        if 'driver' in locals() and driver: driver.close()

def get_node_relationships(node_id: str, limit: int = 10) -> Optional[str]:
    """Retrieves relationships connected to a node."""
    query = """
    MATCH (n)-[r]-(m)
    WHERE n.document_id = $node_id OR n.structure_id = $node_id OR n.name = $node_id OR n.id = $node_id
    RETURN n, r, m
    LIMIT $limit
    """
    # Again, specific MATCH is better, e.g., MATCH (n:Character {name: $node_id})-[r]-(m)

    try:
        driver = get_neo4j_driver()
        output = f"--- Neo4j Relationships for Node ID: {node_id} (Limit {limit}) ---\n"
        records_found = False
        with driver.session() as session:
            result = session.run(query, node_id=node_id, limit=limit)
            for record in result:
                records_found = True
                start_node = record['n']
                rel = record['r']
                end_node = record['m']

                start_desc = start_node.get('name') or start_node.get('title') or start_node.get('document_id') or list(start_node.labels)[0]
                end_desc = end_node.get('name') or end_node.get('title') or end_node.get('document_id') or list(end_node.labels)[0]
                rel_props = json.dumps(dict(rel.items()))

                output += f"({start_desc}:{list(start_node.labels)}) -[{type(rel).__name__} {rel_props}]-> ({end_desc}:{list(end_node.labels)})\n"

        if not records_found:
             output += "(No relationships found)\n"
             logging.warning(f"No relationships found for node ID matching '{node_id}'.")

        return output

    except Exception as e:
        logging.error(f"Error retrieving relationships for node {node_id} from Neo4j: {e}")
        return f"Error retrieving relationships for node {node_id} from Neo4j: {e}"
    finally:
        if 'driver' in locals() and driver: driver.close()

# Example Usage (add under if __name__ == "__main__")
# if __name__ == "__main__":
#     test_chunk_id = "some-uuid-from-your-data" # Replace with a real chunk ID
#     test_node_id = "Pride and PrejudiceDingo" # Replace with a real node name/ID
#
#     print("--- Testing Pinecone Retrieval ---")
#     pinecone_data = get_chunk_from_pinecone(test_chunk_id, include_vector=False)
#     if pinecone_data:
#         print(pinecone_data)
#
#     print("\n--- Testing Neo4j Node Retrieval ---")
#     node_data_str = get_node_by_id(test_node_id)
#     if node_data_str:
#         print(node_data_str)
#
#     print("\n--- Testing Neo4j Relationship Retrieval ---")
#     rels_data_str = get_node_relationships(test_node_id, limit=5)
#     if rels_data_str:
#         print(rels_data_str) 