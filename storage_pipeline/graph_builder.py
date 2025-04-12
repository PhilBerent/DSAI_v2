#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles Step 6: Graph Data Construction.

Translates extracted metadata from chunks and document analysis into a structured graph
format (nodes and edges) suitable for loading into Neo4j.
"""

import logging
from typing import List, Dict, Any, Set, Tuple
import sys
import os
import re # Added import for re.sub

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

def build_graph_data(document_id: str, doc_analysis: Dict[str, Any], chunks_with_analysis: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Constructs lists of nodes and edges for the graph database.

    Args:
        document_id: A unique identifier for the document being processed.
        doc_analysis: The high-level analysis result for the document.
        chunks_with_analysis: List of chunk dictionaries, each containing its
                              metadata including the analysis result under 'analysis'.

    Returns:
        A tuple containing:
            - nodes: A list of node dictionaries for Neo4j.
            - edges: A list of edge dictionaries for Neo4j.
    """
    logging.info(f"Starting graph data construction for document: {document_id}")

    nodes = []
    edges = []
    entity_nodes_created: Set[Tuple[str, str]] = set() # Store (entity_type, entity_name) to avoid duplicates

    # --- 1. Create Document Node --- #
    doc_node = {
        "id": document_id,
        "labels": ["Document"],
        "properties": {
            "document_id": document_id,
            "title": document_id, # Use ID as title for now, consider adding proper title
            "summary": doc_analysis.get("overall_summary", ""),
            "type": doc_analysis.get("document_type", "Unknown")
            # Add more document-level props if needed
        }
    }
    nodes.append(doc_node)

    # --- 2. Create Chapter/Section Nodes (Optional but recommended) --- #
    structure_nodes = {}
    if doc_analysis.get("structure"):
        for i, struct in enumerate(doc_analysis["structure"]):
            struct_id = f"{document_id}_struct_{i+1}"
            struct_title = struct.get('title', f"Unit {struct.get('number', i+1)}")
            struct_node = {
                "id": struct_id,
                "labels": [struct.get("type", "Structure").replace("/","_or_")], # Ensure valid label
                "properties": {
                    "structure_id": struct_id,
                    "title": struct_title,
                    "number": struct.get("number", i+1),
                    "document_id": document_id
                }
            }
            nodes.append(struct_node)
            structure_nodes[struct_title] = struct_id # Map title to node ID
            # Link structure node to document node
            edges.append({
                "start_node_id": document_id,
                "end_node_id": struct_id,
                "type": "HAS_STRUCTURE",
                "properties": {"order": i}
            })

    # --- 3. Process Chunks for Entities and Relationships --- #
    # Consolidate entity references first
    all_entities: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for chunk in chunks_with_analysis:
        chunk_id = chunk['chunk_id']
        chunk_analysis = chunk.get('analysis')
        if not chunk_analysis:
            logging.warning(f"Chunk {chunk_id} missing analysis data. Skipping for graph.")
            continue

        chunk_sequence = chunk['metadata']['source_location']['sequence']
        chunk_struct_ref = chunk['metadata']['source_location'].get('structure_ref', 'Unknown')

        # Add entities to the consolidated list
        chunk_entities = chunk_analysis.get('entities', {})
        for entity_type, entities_list in chunk_entities.items():
            if not isinstance(entities_list, list): continue
            label = entity_type.capitalize() # e.g., "Characters" -> "Character"
            if label.endswith('s'): label = label[:-1]

            for entity_data in entities_list:
                if not isinstance(entity_data, dict) or 'name' not in entity_data:
                    continue
                entity_name = entity_data['name'].strip()
                if not entity_name: continue

                entity_key = (label, entity_name)
                if entity_key not in all_entities:
                    all_entities[entity_key] = {
                        "labels": [label],
                        "properties": {"name": entity_name},
                        "mentioned_in_chunks": set(),
                        "present_in_chunks": set()
                    }
                # Track presence/mention
                all_entities[entity_key]["mentioned_in_chunks"].add(chunk_id)
                if entity_data.get("mentioned_or_present") == "present":
                    all_entities[entity_key]["present_in_chunks"].add(chunk_id)

    # Create unique entity nodes
    entity_name_to_id: Dict[Tuple[str, str], str] = {}
    for (label, name), data in all_entities.items():
        entity_id = f"{label.lower()}_{name.replace(' ', '_').lower()}" # Simple ID generation
        entity_id = re.sub(r'\W+', '', entity_id) # Clean ID
        if (label, name) not in entity_nodes_created:
            node_data = {
                "id": entity_id,
                "labels": data["labels"],
                "properties": data["properties"]
            }
            nodes.append(node_data)
            entity_nodes_created.add((label, name))
        entity_name_to_id[(label, name)] = entity_id

    # Process relationships from chunks
    processed_interactions = set() # Avoid duplicating interactions based on exact participants/chunk
    for chunk in chunks_with_analysis:
        chunk_id = chunk['chunk_id']
        chunk_analysis = chunk.get('analysis')
        if not chunk_analysis: continue
        chunk_sequence = chunk['metadata']['source_location']['sequence']

        # Link entities mentioned/present in this chunk
        chunk_entities = chunk_analysis.get('entities', {})
        for entity_type, entities_list in chunk_entities.items():
             if not isinstance(entities_list, list): continue
             label = entity_type.capitalize()
             if label.endswith('s'): label = label[:-1]
             for entity_data in entities_list:
                 if not isinstance(entity_data, dict) or 'name' not in entity_data: continue
                 entity_name = entity_data['name'].strip()
                 if not entity_name: continue
                 entity_key = (label, entity_name)
                 if entity_key in entity_name_to_id:
                     entity_id = entity_name_to_id[entity_key]
                     rel_type = "PRESENT_IN" if entity_data.get("mentioned_or_present") == "present" else "MENTIONED_IN"
                     # Link entity to chunk (if chunks are nodes) or store chunk ID on edge
                     # For simplicity, we store chunk_id on interaction edges below
                     # Could add edges like: (entity)-[:MENTIONED_IN]->(ChunkNode)

        # Process interactions
        interactions = chunk_analysis.get('relationships_interactions', [])
        if not isinstance(interactions, list): continue

        for interaction in interactions:
            if not isinstance(interaction, dict): continue
            participants = interaction.get('participants', [])
            if not participants or len(participants) < 1:
                 continue

            # Normalize participant names and find their node IDs
            participant_ids = []
            valid_interaction = True
            for name in participants:
                name = name.strip()
                key = ("Character", name) # Assume interactions involve Characters for now
                if key in entity_name_to_id:
                    participant_ids.append(entity_name_to_id[key])
                else:
                    logging.warning(f"Interaction participant '{name}' in chunk {chunk_id} not found as a known Character node. Skipping relationship part.")
                    valid_interaction = False
                    break
            if not valid_interaction or len(participant_ids) < 1:
                continue

            # Create interaction edges (e.g., between first two participants)
            # More complex handling needed for group interactions
            if len(participant_ids) >= 2:
                p1_id = participant_ids[0]
                p2_id = participant_ids[1]
                # Ensure consistent edge direction (e.g., sort IDs) to avoid duplicates
                start_node, end_node = sorted([p1_id, p2_id])

                interaction_key = (start_node, end_node, chunk_id, interaction.get('summary', ''))
                if interaction_key in processed_interactions:
                    continue # Already processed this specific interaction in this chunk

                location_name = interaction.get('location')
                location_id = None
                if location_name:
                    loc_key = ("Location", location_name.strip())
                    if loc_key in entity_name_to_id:
                        location_id = entity_name_to_id[loc_key]
                    else:
                        logging.warning(f"Interaction location '{location_name}' in chunk {chunk_id} not found as Location node.")

                edge_props = {
                    "source_chunk_id": chunk_id,
                    "sequence_marker": chunk_sequence,
                    "summary": interaction.get('summary', ''),
                    "topic": interaction.get('topic'),
                    "interaction_type": interaction.get('type', 'interaction'),
                    "location_id": location_id # Store location ID if found
                    # Add list of all participant IDs if needed for group interactions
                }
                edges.append({
                    "start_node_id": start_node,
                    "end_node_id": end_node,
                    "type": "INTERACTED_WITH",
                    "properties": {k: v for k, v in edge_props.items() if v is not None} # Clean None props
                })
                processed_interactions.add(interaction_key)

                # Link participants to location for this interaction
                if location_id:
                    for p_id in participant_ids:
                         edges.append({
                             "start_node_id": p_id,
                             "end_node_id": location_id,
                             "type": "PRESENT_AT_INTERACTION",
                             "properties": {"source_chunk_id": chunk_id, "sequence_marker": chunk_sequence}
                         })

    logging.info(f"Graph data construction complete. Generated {len(nodes)} nodes and {len(edges)} edges.")
    return nodes, edges 