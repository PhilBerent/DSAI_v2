#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Stores prompts, schemas, and prompt generation logic for LLM calls."""

import json
from typing import Dict, Any

# --- Schemas ---

# Schema for block-level analysis in the map phase
BLOCK_ANALYSIS_SCHEMA = {
    "block_summary": "string (concise summary of this block)",
    "key_entities_in_block": {
        "characters": ["string"],
        "locations": ["string"],
        "organizations": ["string"]
    },
    "structural_marker_found": "string | None (e.g., 'Chapter X Title', 'Part Y Start')"
}

# Schema for detailed chunk analysis
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

# Schema for overall document analysis in the reduce phase
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


# --- System Messages ---

system_msg_for_large_block_anal = "You are an expert literary analyst. Analyze the provided text block and extract information strictly according to the provided JSON schema. Only output JSON."

chunk_system_message = "You are a detailed text analyst specializing in extracting entities, relationships, and events from text chunks within a larger document context. Output only valid JSON matching the schema."

reduce_system_message = "You are an expert synthesizer of document analysis. Based on block summaries and entity lists, perform the requested analysis and output ONLY valid JSON according to the schema."


# --- Prompt Generation Functions ---

def get_anal_large_block_prompt(block_info: Dict[str, Any]) -> str:
    """Generates the user prompt for analyzing a large text block."""
    block_text = block_info.get('text', '')
    block_ref = block_info.get('ref', 'Unknown Reference')

    # Use the globally defined schema
    schema_string = json.dumps(BLOCK_ANALYSIS_SCHEMA, indent=2)

    # Truncate block text if necessary (using a reasonable limit)
    # TODO: Consider making the truncation limit configurable
    max_chars = 80000
    truncated_text = block_text[:max_chars]
    truncation_note = "(Note: Block might be truncated for analysis if excessively long)" if len(block_text) > max_chars else ""

    prompt = f"""
Analyze the following large text block from a document. Extract a concise summary, key entities primarily featured *in this block*, and identify any structural marker (like Chapter/Part title) found near the beginning of this block. Adhere strictly to the provided JSON schema.

JSON Schema:
{schema_string}

Text Block (Ref: {block_ref}):
--- START BLOCK ---
{truncated_text}
--- END BLOCK ---
{truncation_note}

Provide the analysis ONLY in the specified JSON format.
"""
    return prompt

# TODO: Add prompt generation functions for other tasks like reduce phase, chunk analysis if needed.

# --- Specific Instructions for Reduce Phase (Example - Keep adding others) ---
# These might stay here or move to a more complex prompt management system later

def getNovelReducePrompt(NumBlocks, BlocksPerStructUnit=1, WordsForFinalAnalysisSummary=200):
    if BlocksPerStructUnit > 1:
        numStrucElements = NumBlocks // BlocksPerStructUnit
        # calc remainder
        remainder = NumBlocks % BlocksPerStructUnit
        lastElementLength = BlocksPerStructUnit + remainder
        remainderText = f" apart from the last element which will cover the last {lastElementLength} Blocks. " if remainder > 0 else "."
        strucElementsText = f"For the structure output create one entry for each {numStrucElements} Block (which are chapters or other distinct elements of the novel). As there are {NumBlocks} Blocks in total, create{numStrucElements} entries in total each one covering {BlocksPerStructUnit} Blocks{remainderText} "
    else :
       strucElementsText = f"For the structure output create one entry for each Block (which are chapters or other distinct elements of the novel). As there are {NumBlocks} Blocks in total, create {NumBlocks} entries in total. "
    strucElementsText += """ For each entry, assign a brief, descriptive title reflecting the main event or topic of that block summary (e.g., "Introduction of Character X", "The Climax at Location Y", "A Turning Point"). Use the block number for the number field."""
        
    ret = f"""
Overall Summary: Provide a detailed analysis of the novel (approx. {WordsForFinalAnalysisSummary}) for the overall_summary output. This should cover its main themes, key characters, and plot structure based on the provided Block Data summaries.

Structure Output: {strucElementsText}

Preliminary Key Entities (Characters, Locations, Organizations) Output: 
Your primary goal here is to identify the unique entities represented in the ENTITY DATA list and output a clean, deduplicated list for preliminary_key_entities.
Source: Use the ENTITY DATA as your sole source list for potential entities. The BLOCK DATA summaries should only be used as context to help you understand if different names in the ENTITY DATA refer to the same entity.
Task: Examine the lists of characters, locations, and organizations within ENTITY DATA. Identify all entries that are variations referring to the same underlying entity. For example, "Philip Jones", "Mr. Jones", "Jones", and "Philip" might all refer to one character.
Output Rule: For each unique entity identified, select only one canonical name to include in the final output lists (characters, locations, organizations).
Choosing the Canonical Name: Prefer the most complete or formal version of the name found in the ENTITY DATA for that entity (e.g., select "Professor Albus Dumbledore" if "Dumbledore" and "Albus" also appear for the same person). If multiple equally complete forms exist, choose the one that appears first or seems most representative based on the context from BLOCK DATA summaries.
No Variations: The final lists in preliminary_key_entities must contain only these single, canonical names. Do not include multiple variations of the same entity's name. Every unique entity concept represented by one or more names in the ENTITY DATA should appear exactly once in the output under its chosen canonical name.
"""
    
    return ret
    
def getBiographyReducePrompt(NumBlocks, BlocksPerStructUnit=1, WordsForFinalAnalysisSummary=200):
    ret = "Please provide a detailed analysis of the biography, including its main themes and the life of the subject."
    return ret

def getJournalArticleReducePrompt(NumBlocks, BlocksPerStructUnit=1, WordsForFinalAnalysisSummary=200):
    ret = "Please provide a detailed analysis of the journal article, including its main arguments and contributions to the field."
    return ret
