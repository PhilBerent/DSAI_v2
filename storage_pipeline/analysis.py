#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles Steps 2 & 4: LLM-based document and chunk analysis."""

import logging
import json
from typing import List, Dict, Any, Tuple, Optional
import sys
import os
import concurrent.futures
import tiktoken # Add tiktoken import
import time # Added for rate limit sleep
import openai # Added for RateLimitError
from enum import Enum
from prompts import *

# Adjust path to import from parent directory AND sibling directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir) # Add parent DSAI_v2_Scripts

# Import required global modules
try:
    from globals import *
    from UtilityFunctions import *
    from DSAIParams import *
    # Import from the new constants file
    from enums_and_constants import *
    # Import the new centralized LLM call function
    from llm_calls import call_llm_json_mode
except ImportError as e:
    print(f"Error importing core modules or enums_and_constants or llm_calls: {e}")
    raise

# Remove OpenAI client import from here, it's handled in llm_calls
# from .db_connections import client
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

    # Define system message for this specific task
    system_msg = "You are an expert literary analyst. Analyze the provided text block and extract information strictly according to the provided JSON schema. Only output JSON."

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

    try:
        # Use the new centralized function for the API call in JSON mode
        # Pass the specific system message and the constructed prompt
        block_analysis_result = call_llm_json_mode(system_message=system_msg, prompt=prompt)

        # Add block reference back for context in reduce step
        block_analysis_result['block_ref'] = block_ref
        block_analysis_result['block_index'] = block_index
        return block_analysis_result
    except Exception as e:
        logging.error(f"Failed to analyze large block {block_index + 1} ({block_ref}): {e}")
        return None # Return None on failure for this block


# --- REVISED: Step 3 - Map Phase: Analyze Blocks in Parallel ---
def perform_map_block_analysis(large_blocks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Performs the Map phase of Step 3: Analyzes large blocks in parallel.
    Estimates token usage locally and processes all blocks with basic rate limit handling.

    Args:
        large_blocks: List of coarse chunks/blocks from Step 2.

    Returns:
        A tuple containing:
        - map_results: A list of analysis results from each successfully processed block.
        - final_entities: A dictionary of consolidated entities found across all blocks.
    """
    logging.info(f"Starting Map Phase: Analyzing {len(large_blocks)} large blocks...")

    # --- Dynamic Worker Calculation Logic (Copied from original function) --- #
    sample_size = min(ANALYSIS_SAMPLE_SIZE, len(large_blocks))
    estimated_total_tokens_per_call = None
    encoding = None

    try:
        encoding = tiktoken.encoding_for_model(CHAT_MODEL_NAME)
        logging.info(f"Using tiktoken encoding for {CHAT_MODEL_NAME}.")
    except Exception:
        logging.warning(f"tiktoken encoding for {CHAT_MODEL_NAME} not found. Using rough character count for token estimation.")

    if sample_size > 0:
        logging.info(f"Estimating input tokens locally based on {sample_size} sample blocks...")
        total_input_tokens_sampled = 0
        successful_samples = 0

        for i in range(sample_size):
            block_info = large_blocks[i]
            try:
                logging.debug(f"Estimating tokens for sample block {i+1}/{sample_size}...")
                BLOCK_ANALYSIS_SCHEMA_FOR_SAMPLING = {
                    "block_summary": "string (concise summary of this block)",
                    "key_entities_in_block": {
                        "characters": ["string"],
                        "locations": ["string"],
                        "organizations": ["string"]
                    },
                    "structural_marker_found": "string | None (e.g., 'Chapter X Title', 'Part Y Start')"
                }
                schema_str = json.dumps(BLOCK_ANALYSIS_SCHEMA_FOR_SAMPLING, indent=2)
                truncated_block = block_info['text'][:80000]
                system_msg = "You are an expert literary analyst. Analyze the provided text and extract information strictly according to the provided JSON schema. Only output JSON."
                block_ref_for_prompt = block_info['ref']
                user_msg = f""""Analyze the following large text block...JSON Schema:
{schema_str}...Text Block (Ref: {block_ref_for_prompt})...{truncated_block}...Provide the analysis ONLY..."""
                prompt_for_token_count = system_msg + user_msg

                if encoding:
                    input_tokens = len(encoding.encode(prompt_for_token_count))
                else:
                    input_tokens = len(prompt_for_token_count) // 4

                total_input_tokens_sampled += input_tokens
                successful_samples += 1
                logging.debug(f"Sample {i+1}: Estimated Input Tokens = {input_tokens}")

            except Exception as sample_exc:
                logging.warning(f"Error estimating tokens for sample block {i+1}: {sample_exc}")

        if successful_samples > 0:
            average_input_tokens = total_input_tokens_sampled / successful_samples
            estimated_output_tokens = average_input_tokens * ESTIMATED_OUTPUT_TOKEN_FRACTION
            estimated_total_tokens_per_call = average_input_tokens + estimated_output_tokens
            logging.info(f"Dynamic estimate: Average Input Tokens = {average_input_tokens:.0f}, "
                         f"Estimated Output Tokens = {estimated_output_tokens:.0f} (Fraction: {ESTIMATED_OUTPUT_TOKEN_FRACTION}), "
                         f"Estimated Total Tokens/Call = {estimated_total_tokens_per_call:.0f}")
        else:
            logging.warning("Failed to estimate tokens for any sample blocks. Cannot dynamically estimate.")

    else:
        logging.warning("No blocks available for sampling.")

    if estimated_total_tokens_per_call is None:
        dynamic_max_workers = MAX_WORKERS_FALLBACK
        logging.warning(f"Using fallback max_workers: {dynamic_max_workers}")
    else:
        calls_per_minute_tpm = GPT4O_TPM / estimated_total_tokens_per_call if estimated_total_tokens_per_call > 0 else 0
        concurrent_limit_rpm = GPT4O_RPM / WORKER_RATE_LIMIT_DIVISOR
        calculated_max_workers = min(calls_per_minute_tpm, concurrent_limit_rpm)
        dynamic_max_workers = max(1, int(calculated_max_workers * WORKER_SAFETY_FACTOR))
        dynamic_max_workers = min(dynamic_max_workers, len(large_blocks)) # Cannot have more workers than blocks
        logging.info(f"Calculated dynamic max_workers: {dynamic_max_workers} "
                     f"(Based on TPM={GPT4O_TPM}, RPM={GPT4O_RPM}, "
                     f"EstTotalTokens={estimated_total_tokens_per_call:.0f}, Factor={WORKER_SAFETY_FACTOR}, "
                     f"RPM Divisor={WORKER_RATE_LIMIT_DIVISOR})")
    # --- End Dynamic Worker Calculation Logic --- #

    map_results = []
    logging.info(f"Processing all {len(large_blocks)} blocks in parallel with {dynamic_max_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=dynamic_max_workers) as executor:
        future_to_block_index = {
            executor.submit(_analyze_large_block, block, i): i
            for i, block in enumerate(large_blocks)
        }

        for future in concurrent.futures.as_completed(future_to_block_index):
            original_block_index = future_to_block_index[future]
            block_ref = large_blocks[original_block_index].get('ref', f'Index {original_block_index}')
            try:
                result = future.result()
                if result:
                    map_results.append(result)
            except openai.RateLimitError as rle:
                logging.warning(f"Rate limit hit processing block {original_block_index} ({block_ref}). Sleeping for {RATE_LIMIT_SLEEP_SECONDS}s. Error: {rle}")
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)
            except Exception as exc:
                logging.error(f"Block analysis task for index {original_block_index} ({block_ref}) generated an exception: {exc}")

    # Sort results by original block index (important for Reduce phase)
    map_results.sort(key=lambda x: x.get('block_index', -1))

    if not map_results:
        logging.error("Map phase failed for all blocks or too many rate limit errors. Cannot proceed to Reduce phase.")
        # Return empty results, let caller handle this case
        return [], {}

    successful_count = len(map_results)
    total_count = len(large_blocks)
    failed_count = total_count - successful_count
    logging.info(f"Map phase complete. Successfully analyzed {successful_count}/{total_count} blocks. ({failed_count} failures/skips)")

    # --- Consolidate Entities from Map Results --- #
    consolidated_entities = {"characters": set(), "locations": set(), "organizations": set()}
    for i, result in enumerate(map_results):
        if not isinstance(result, dict):
             logging.warning(f"Skipping invalid map result at index {i}: {result}")
             continue
        block_ref_val = result.get('block_ref', f'Index {result.get("block_index", "Unknown")}')
        entities = result.get('key_entities_in_block', {})
        if isinstance(entities, dict):
            consolidated_entities["characters"].update(entities.get("characters", []))
            consolidated_entities["locations"].update(entities.get("locations", []))
            consolidated_entities["organizations"].update(entities.get("organizations", []))
        else:
             logging.warning(f"Unexpected entity format in block {block_ref_val}: {entities}")

    # Convert sets back to lists for the final structure
    final_entities = {k: sorted(list(v)) for k, v in consolidated_entities.items()} # Sort for consistency

    return map_results, final_entities


# --- REVISED: Step 3 - Reduce Phase: Synthesize Document Overview ---
def perform_reduce_document_analysis(
    map_results: List[Dict[str, Any]],
    final_entities: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Performs the Reduce phase of Step 3: Synthesizes the overall document analysis.

    Args:
        map_results: The list of analysis results from the Map phase.
        final_entities: The dictionary of consolidated entities from the Map phase.

    Returns:
        The synthesized document analysis result matching DOCUMENT_ANALYSIS_SCHEMA,
        or an error dictionary if reduction fails.
    """
    logging.info("Starting Reduce phase: Synthesizing document overview with type-specific instructions...")

    if not map_results:
        logging.error("Cannot perform Reduce phase: No valid results from Map phase.")
        return {
            "error": "No valid results from Map phase to synthesize.",
            "document_type": "Analysis Failed",
            "structure": [],
            "overall_summary": "",
            "preliminary_key_entities": {}
        }

    # Prepare input for the reduction prompt
    synthesis_input = ""
    structure_list_from_map = [] # Keep this if needed, though not used in current prompt logic?
    num_blocks = len(map_results)
    for i, result in enumerate(map_results):
        # Extract values safely using .get()
        block_ref_val = result.get('block_ref', f'Index {result.get("block_index", "Unknown")}')
        block_summary_val = result.get('block_summary', 'Summary Unavailable')
        synthesis_input += f"Block {i+1} Summary (Ref: {block_ref_val}): {block_summary_val}\\n"

        structural_marker = result.get('structural_marker_found')
        if structural_marker:
             structure_list_from_map.append({
                 "type": "Detected Marker", # Placeholder type
                 "title": structural_marker,
                 "number": result.get('block_index', i + 1) # Use block index if available
             })
        # NOTE: structure_list_from_map is constructed but not explicitly used in the reduce_prompt below.
        # It might be useful for future prompt refinements.

    # --- Format final_entities for the prompt --- #
    try:
        formatted_entities_str = json.dumps(final_entities, indent=2)
    except Exception as json_err:
        logging.warning(f"Could not format final_entities for prompt: {json_err}")
        formatted_entities_str = "Error formatting entities." # Fallback

    # --- Build the Reduce Prompt (Copied and adapted from original function) --- #
    allowed_types_str = ", ".join([f"'{t}'" for t in DocumentTypeList]) # Treat t as string
    novel_instructions = getNovelReducePrompt(num_blocks)
    biography_instructions = getBiographyReducePrompt(num_blocks)
    journal_instructions = getJournalArticleReducePrompt(num_blocks)
    # Define system message for the reduction task
    reduce_system_message = "You are an expert synthesizer of document analysis. Based on block summaries and entity lists, perform the requested analysis and output ONLY valid JSON according to the schema."

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
    {json.dumps(DOCUMENT_ANALYSIS_SCHEMA, indent=2)}

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

    # Call LLM for reduction using the centralized function
    try:
        final_analysis = call_llm_json_mode(
            system_message=reduce_system_message,
            prompt=reduce_prompt
            # Temperature uses default from llm_calls/DSAIParams
        )
        logging.info("Reduce phase complete. Document overview synthesized.")
        return final_analysis
    except Exception as e:
        logging.error(f"Failed to synthesize document overview (Reduce phase): {e}")
        # Return an error structure if the final call fails
        return {
            "error": f"Failed during final synthesis: {e}",
            "document_type": "Analysis Failed", "structure": [], "overall_summary": "", "preliminary_key_entities": {}
        }

def analyze_chunk_details(chunk_text: str, chunk_id: str, doc_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyzes a single fine-grained chunk for entities, relationships, etc."""
    logging.info(f"Analyzing details for chunk {chunk_id}...")

    # Provide document context in the prompt if available
    context_summary = "No broader document context provided."
    if doc_context:
        doc_type = doc_context.get("document_type", "Unknown Type")
        doc_summary = doc_context.get("overall_summary", "Summary Unavailable")
        context_summary = f"Document Context: Type={doc_type}. Overall Summary: {doc_summary[:500]}..."

    # Define system message for chunk analysis
    chunk_system_message = "You are a detailed text analyst specializing in extracting entities, relationships, and events from text chunks within a larger document context. Output only valid JSON matching the schema."

    prompt = f"""
    Analyze the following text chunk meticulously. Extract entities (characters, locations, organizations), relationships/interactions between characters, key events, and relevant keywords/topics. Consider the provided document context.

    {context_summary}

    Output Format: Adhere strictly to this JSON schema:
    {json.dumps(CHUNK_ANALYSIS_SCHEMA, indent=2)}

    Text Chunk (ID: {chunk_id}):
    --- START CHUNK ---
    {chunk_text}
    --- END CHUNK ---

    Provide the analysis ONLY in the specified JSON format.
    """

    try:
        # Use the centralized JSON mode caller
        chunk_analysis_result = call_llm_json_mode(
            system_message=chunk_system_message,
            prompt=prompt
            # Temperature uses default
        )
        return chunk_analysis_result
    except Exception as e:
        logging.error(f"Failed to analyze chunk details for {chunk_id}: {e}")
        # Return error structure or raise? For now, return error dict
        return {
            "error": f"Analysis failed for chunk {chunk_id}: {e}",
            "entities": {},
            "relationships_interactions": [],
            "events": [],
            "keywords_topics": []
        }

# Note: Batch processing chunks might be more efficient for API calls
# This would involve grouping chunks and modifying the prompt structure. 