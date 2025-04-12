#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles Steps 2 & 4: LLM-based document and chunk analysis."""

import logging
import json
from typing import List, Dict, Any, Tuple, Optional
import sys
import os
import concurrent.futures

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

from .db_connections import client # OpenAI client
from .config_loader import CHAT_MODEL_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define expected JSON structures for robustness
# These act as schemas for the LLM output
DOCUMENT_ANALYSIS_SCHEMA = {
    "document_type": "Novel | Non-Fiction (Non-Technical) | Other",
    "structure": [{"type": "Chapter/Section", "title": "string", "number": "int/string"}],
    "overall_summary": "string",
    "preliminary_key_entities": {
        "characters": ["string"],
        "locations": ["string"],
        "organizations": ["string"]
    }
}

CHUNK_ANALYSIS_SCHEMA = {
    "entities": {
        "characters": [{"name": "string", "mentioned_or_present": "mentioned | present"}],
        "locations": [{"name": "string"}],
        "organizations": [{"name": "string"}]
    },
    "relationships_interactions": [
        {
            "type": "interaction | mention",
            "participants": ["character_name"],
            "location": "location_name | None",
            "summary": "string",
            "topic": "string | None"
        }
    ],
    "events": ["string"],
    "keywords_topics": ["string"]
}

def _call_openai_json_mode(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to call OpenAI API in JSON mode with a specific schema."""
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            # The 'schema' key is not a valid parameter for response_format
            # response_format={"type": "json_object", "schema": schema},
            # Only specify the type
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert literary analyst. Analyze the provided text and extract information strictly according to the provided JSON schema. Only output JSON."},
                {"role": "user", "content": prompt} # The prompt still contains the schema description
            ]
        )
        # Ensure response content is not None
        if response.choices[0].message.content is None:
             raise ValueError("OpenAI response content is None")
        
        result = json.loads(response.choices[0].message.content)
        # TODO: Add validation against the schema here if needed
        return result
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        # Consider adding retry logic here
        raise

# --- NEW: Step 3.1 - Map Phase Helper ---
def _analyze_large_block(block_info: Dict[str, Any], block_index: int) -> Optional[Dict[str, Any]]:
    """Analyzes a single large text block using LLM."""
    block_text = block_info['text']
    block_ref = block_info['ref']
    logging.info(f"Analyzing large block {block_index + 1}: {block_ref} (Length: {len(block_text)} chars)")

    # Define schema for block-level analysis
    BLOCK_ANALYSIS_SCHEMA = {
        "block_summary": "string (concise summary of this block)",
        "key_entities_in_block": {
            "characters": ["string"],
            "locations": ["string"],
            "organizations": ["string"]
        },
        "structural_marker_found": "string | None (e.g., 'Chapter X Title', 'Part Y Start')"
    }

    prompt = f"""
    Analyze the following large text block from a document. Extract a concise summary, key entities primarily featured *in this block*, and identify any structural marker (like Chapter/Part title) found near the beginning of this block. Adhere strictly to the provided JSON schema.

    JSON Schema:
    {json.dumps(BLOCK_ANALYSIS_SCHEMA, indent=2)}

    Text Block (Ref: {block_ref}):
    --- START BLOCK ---
    {block_text[:80000]}
    --- END BLOCK ---
    (Note: Block might be truncated for analysis if excessively long)

    Provide the analysis ONLY in the specified JSON format.
    """
    # Note: Added truncation [:80000] as a safety measure for the prompt itself. Adjust if needed.

    try:
        # Use the same helper function for the API call
        block_analysis_result = _call_openai_json_mode(prompt, BLOCK_ANALYSIS_SCHEMA)
        # Add block reference back for context in reduce step
        block_analysis_result['block_ref'] = block_ref
        block_analysis_result['block_index'] = block_index
        return block_analysis_result
    except Exception as e:
        logging.error(f"Failed to analyze large block {block_index + 1} ({block_ref}): {e}")
        return None # Return None on failure for this block


# --- REVISED: Step 3 - Iterative Document Analysis ---
def analyze_document_iteratively(large_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs iterative analysis (Step 3) using a Map-Reduce approach.

    Args:
        large_blocks: List of coarse chunks/blocks from Step 2.

    Returns:
        The synthesized document analysis result matching DOCUMENT_ANALYSIS_SCHEMA.
    """
    logging.info(f"Starting Step 3: Iterative document analysis on {len(large_blocks)} large blocks...")

    # --- 3.1 Map Phase ---
    map_results = []
    # Use ThreadPoolExecutor for parallel analysis of blocks
    # Adjust max_workers based on API rate limits and desired speed
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_block = {
            executor.submit(_analyze_large_block, block, i): i
            for i, block in enumerate(large_blocks)
        }
        for future in concurrent.futures.as_completed(future_to_block):
            block_index = future_to_block[future]
            try:
                result = future.result()
                if result:
                    map_results.append(result)
            except Exception as exc:
                logging.error(f"Block analysis task {block_index + 1} generated an exception: {exc}")

    # Sort results by original block index to maintain order
    map_results.sort(key=lambda x: x.get('block_index', -1))

    if not map_results:
        logging.error("Map phase (block analysis) failed for all blocks.")
        # Return a default/error structure or raise an exception
        return {"error": "Failed to analyze any document blocks."}

    logging.info(f"Map phase complete. Successfully analyzed {len(map_results)} blocks.")

    # --- 3.2 Reduce Phase ---
    logging.info("Starting Reduce phase: Synthesizing document overview...")

    # Prepare input for the reduction prompt
    synthesis_input = ""
    structure_list_from_map = []
    consolidated_entities = {"characters": set(), "locations": set(), "organizations": set()}

    for i, result in enumerate(map_results):
        # Extract values before using them in the f-string
        block_ref_val = result.get('block_ref', 'N/A')
        block_summary_val = result.get('block_summary', 'N/A')
        # Use the simpler variables in the f-string
        synthesis_input += f"Block {i+1} Summary (Ref: {block_ref_val}): {block_summary_val}\n"
        if result.get('structural_marker_found'):
             structure_list_from_map.append({
                 "type": "Detected Marker", # Placeholder type
                 "title": result['structural_marker_found'],
                 "number": i + 1 # Use order as number for now
             })
        # Consolidate entities
        entities = result.get('key_entities_in_block', {})
        consolidated_entities["characters"].update(entities.get("characters", []))
        consolidated_entities["locations"].update(entities.get("locations", []))
        consolidated_entities["organizations"].update(entities.get("organizations", []))

    # Convert sets back to lists for the final structure
    final_entities = {k: list(v) for k, v in consolidated_entities.items()}

    # Build the reduction prompt
    # Use the same schema definition as the original analyze_document_structure
    reduce_prompt = f"""
    Based on the following summaries and key entities extracted from consecutive blocks of a large document, synthesize an overall analysis. Provide the overall document type, a concise overall summary, a consolidated list of preliminary key entities (most important ones overall), and refine the list of structural markers into a coherent document structure. Adhere strictly to the provided JSON schema.

    JSON Schema:
    {json.dumps(DOCUMENT_ANALYSIS_SCHEMA, indent=2)}

    Summaries & Entities from Blocks:
    --- START BLOCK DATA ---
    {synthesis_input}
    --- END BLOCK DATA ---

    Provide the synthesized analysis ONLY in the specified JSON format. Ensure the 'structure' list is ordered correctly.
    """

    # Call LLM for reduction
    try:
        # Use the same helper function
        final_analysis = _call_openai_json_mode(reduce_prompt, DOCUMENT_ANALYSIS_SCHEMA)
        logging.info("Reduce phase complete. Synthesized document analysis.")
        # Optional: Refine the generated structure list if needed based on structure_list_from_map
        # e.g., try to match titles, assign more specific types like Chapter/Part if possible
    except Exception as e:
        logging.error(f"Reduce phase failed: {e}")
        # Fallback: return partial data or error
        final_analysis = {
            "document_type": "Unknown (Synthesis Failed)",
            "structure": structure_list_from_map, # Best guess from map phase
            "overall_summary": "Synthesis failed, see block summaries.",
            "preliminary_key_entities": final_entities, # From consolidation
            "error": f"Reduce phase failed: {e}"
        }

    return final_analysis

def analyze_chunk_details(chunk_text: str, chunk_id: str, doc_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Performs detailed chunk-level analysis (Step 4)."""
    logging.info(f"Starting detailed analysis for chunk: {chunk_id}...")

    # Provide some document context if available (e.g., main characters, setting)
    context_str = ""
    if doc_context and doc_context.get('preliminary_key_entities'):
        context_str = f"Document Context: Key characters might include {doc_context['preliminary_key_entities'].get('characters', [])}. "
        context_str += f"Primary locations might include {doc_context['preliminary_key_entities'].get('locations', [])}."

    prompt = f"""
    Analyze the following text chunk from a larger document. Extract detailed entities, relationships/interactions, events, and keywords/topics. Use the provided JSON schema. {context_str}

    JSON Schema:
    {json.dumps(CHUNK_ANALYSIS_SCHEMA, indent=2)}

    Text Chunk (ID: {chunk_id}):
    --- START CHUNK ---
    {chunk_text}
    --- END CHUNK ---

    Provide the analysis ONLY in the specified JSON format. Ensure character names are consistent. Specify if characters are directly present or just mentioned.
    """

    chunk_analysis_result = _call_openai_json_mode(prompt, CHUNK_ANALYSIS_SCHEMA)
    logging.info(f"Detailed analysis complete for chunk: {chunk_id}")
    return chunk_analysis_result

# Note: Batch processing chunks might be more efficient for API calls
# This would involve grouping chunks and modifying the prompt structure. 