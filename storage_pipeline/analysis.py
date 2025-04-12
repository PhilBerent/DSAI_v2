#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles Steps 2 & 4: LLM-based document and chunk analysis."""

import logging
import json
from typing import List, Dict, Any, Tuple
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
            response_format={"type": "json_object", "schema": schema},
            messages=[
                {"role": "system", "content": "You are an expert literary analyst. Analyze the provided text and extract information strictly according to the provided JSON schema. Only output JSON."},
                {"role": "user", "content": prompt}
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

def analyze_document_structure(full_text: str) -> Dict[str, Any]:
    """Performs high-level analysis (Step 2) on the entire document text."""
    logging.info("Starting high-level document analysis...")

    # Truncate text if excessively long to avoid overwhelming the model (optional)
    # max_chars = 50000 # Example limit
    # truncated_text = full_text[:max_chars] + ("... [truncated]" if len(full_text) > max_chars else "")
    truncated_text = full_text # Use full text for now, adjust if needed

    prompt = f"""
    Analyze the following document text to determine its structure, type, summary, and preliminary key entities. Adhere strictly to the provided JSON schema.

    JSON Schema:
    {json.dumps(DOCUMENT_ANALYSIS_SCHEMA, indent=2)}

    Document Text:
    --- START TEXT ---
    {truncated_text}
    --- END TEXT ---

    Provide the analysis ONLY in the specified JSON format.
    """

    analysis_result = _call_openai_json_mode(prompt, DOCUMENT_ANALYSIS_SCHEMA)
    logging.info("High-level document analysis complete.")
    return analysis_result

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