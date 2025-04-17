#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Stores prompts, schemas, and prompt generation logic for LLM calls."""

import json
from typing import Dict, Any
import sys
import os
from globals import *
from UtilityFunctions import *
from DSAIParams import *
from enums_and_constants import *


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
    "document_type": "Novel | Non-Fiction (Non-Technical) | Other", # Note: This might need alignment with DocumentTypeList
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


# --- Specific Instructions for Reduce Phase (Used by getReducePrompt) ---

def _getNovelReduceInstructions(num_blocks: int) -> str:
    # Keeping original wording, assuming NumBlocks = num_blocks
    # Note: The original prompt text from the user diff was slightly different than the one
    # generated by the previous version of the code I created earlier. Reverting to the user's last version.
    # Assume WordsForFinalAnalysisSummary=200, BlocksPerStructUnit=1 based on previous context
    WordsForFinalAnalysisSummary=200
    BlocksPerStructUnit=1

    if BlocksPerStructUnit > 1:
        numStrucElements = num_blocks // BlocksPerStructUnit
        remainder = num_blocks % BlocksPerStructUnit
        lastElementLength = BlocksPerStructUnit + remainder
        remainderText = f" apart from the last element which will cover the last {lastElementLength} Blocks. " if remainder > 0 else "."
        strucElementsText = f"For the structure output create one entry for each {numStrucElements} Block (which are chapters or other distinct elements of the novel). As there are {num_blocks} Blocks in total, create{numStrucElements} entries in total each one covering {BlocksPerStructUnit} Blocks{remainderText} "
    else :
       strucElementsText = f"For the structure output create one entry for each Block (which are chapters or other distinct elements of the novel). As there are {num_blocks} Blocks in total, create {num_blocks} entries in total. "
    strucElementsText += """ For each entry, assign a brief, descriptive title reflecting the main event or topic of that block summary (e.g., "Introduction of Character X", "The Climax at Location Y", "A Turning Point"). Use the block number for the number field."""

    ret = f"""
Overall Summary: Provide a detailed analysis of the novel (approx. {WordsForFinalAnalysisSummary}) for the overall_summary output. This should cover its main themes, key characters, and plot structure based on the provided Block Data summaries.

Structure Output: {strucElementsText}

Preliminary Key Entities (Characters, Locations, Organizations) Output:
Your primary goal here is to identify the unique entities represented in the ENTITY DATA list and output a clean, deduplicated list for preliminary_key_entities.
Source: Use the ENTITY DATA as your sole source list for potential entities. The BLOCK DATA summaries should only be used as context to help you understand if different names in the ENTITY DATA refer to the same entity.
Task: Examine the lists of characters, locations, and organizations within ENTITY DATA. Identify all entries that are variations referring to the same underlying entity. For example, "Philip Jones", "Mr. Jones", "Jones", and "Philip" might all refer to one character. In this example multiple names refere to the same entity. In this case you must return only one of the names so that the entity is represented only once in the final output list. The same applies to locations and organizations. ABDOLUTELY DO NOT return multiple names for the same entity in the final output list. For example you would not return both "Mr Jones" and "Jones" if they refer to the same person. The same applies to locations and organizations.
Output Rule: For each unique entity identified, select only one canonical name to include in the final output lists (characters, locations, organizations).
Choosing the Canonical Name: Prefer the most complete or formal version of the name found in the ENTITY DATA for that entity (e.g., select "Professor Albus Dumbledore" if "Dumbledore" and "Albus" also appear for the same person). If multiple equally complete forms exist, choose the one that appears first or seems most representative based on the context from BLOCK DATA summaries.
No Variations: The final lists in preliminary_key_entities must contain only these single, canonical names. Do not include multiple variations of the same entity's name. Every unique entity concept represented by one or more names in the ENTITY DATA should appear exactly once in the output under its chosen canonical name.
"""
    return ret

def _getBiographyReduceInstructions(num_blocks: int) -> str:
    # Placeholder - using original minimal text
    return "Please provide a detailed analysis of the biography, including its main themes and the life of the subject."

def _getJournalArticleReduceInstructions(num_blocks: int) -> str:
    # Placeholder - using original minimal text
    return "Please provide a detailed analysis of the journal article, including its main arguments and contributions to the field."


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

def getReducePrompt(
    num_blocks: int,
    formatted_entities_str: str,
    synthesis_input: str
) -> str:
    """Generates the user prompt for the Reduce phase analysis."""

    # Get specific instructions based on document type (assuming default Novel for now)
    # This uses the internal helper functions defined above
    novel_instructions = _getNovelReduceInstructions(num_blocks)
    biography_instructions = _getBiographyReduceInstructions(num_blocks)
    journal_instructions = _getJournalArticleReduceInstructions(num_blocks)

    # Prepare allowed types string using the imported list
    allowed_types_str = ", ".join([f"'{t}'" for t in DocumentTypeList])

    # Prepare schema string
    schema_string = json.dumps(DOCUMENT_ANALYSIS_SCHEMA, indent=2)

    # Construct the final prompt using an f-string, mirroring the original structure
    reduce_prompt = f"""
You will analyze summaries extracted from consecutive blocks of a large document. Follow these steps carefully:

1.  **Determine Document Type:** Based on the content of the summaries, determine the overall document type. Choose ONLY ONE type from the following list: [{allowed_types_str}].

2.  **Apply Specific Instructions:** Based *only* on the Document Type you determined in Step 1, follow the corresponding specific instructions below to guide your analysis:

    --- Instructions for '{DocumentType.NOVEL.value}' ---
    {novel_instructions}
    --- End Instructions for '{DocumentType.NOVEL.value}' ---

    --- Instructions for '{DocumentType.BIOGRAPHY.value}' ---
    {biography_instructions}
    --- End Instructions for '{DocumentType.BIOGRAPHY.value}' ---

    --- Instructions for '{DocumentType.JOURNAL_ARTICLE.value}' ---
    {journal_instructions}
    --- End Instructions for '{DocumentType.JOURNAL_ARTICLE.value}' ---

3.  **Generate Output:** Using insights from the summaries, the 'Raw Consolidated Entities' list (if applicable based on instructions), and the specific instructions you followed, generate the final analysis. Adhere strictly to the provided JSON Schema. Ensure the 'structure' list reflects the instructions for the determined document type. Ensure 'preliminary_key_entities' reflects the requested consolidation and deduplication.

JSON Schema:
{schema_string}

Raw Consolidated Entities (Potential Duplicates Exist):
--- START ENTITY DATA ---
{formatted_entities_str}
--- END ENTITY DATA ---

Summaries from Blocks:
--- START BLOCK DATA ---
{synthesis_input}
--- END BLOCK DATA ---

(Note: Summaries and entity lists may be truncated if excessively long. Prioritize analysis based on available data.)

Provide the complete synthesized analysis ONLY in the specified JSON format, including the determined 'document_type'.
"""
    return reduce_prompt

# Add other prompts as needed...
