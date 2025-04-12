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

    # --- Dynamic Worker Calculation Logic --- #
    sample_size = min(ANALYSIS_SAMPLE_SIZE, len(large_blocks))
    estimated_tokens_per_call = None # Initialize
    sample_results_collected = [None] * sample_size # Pre-allocate list for sample results
    encoding = None

    try:
        encoding = tiktoken.encoding_for_model(CHAT_MODEL_NAME)
        logging.info(f"Using tiktoken encoding for {CHAT_MODEL_NAME}.")
    except Exception:
        logging.warning(f"tiktoken encoding for {CHAT_MODEL_NAME} not found. Using rough character count for token estimation.")
        # encoding remains None

    if sample_size > 0:
        logging.info(f"Estimating tokens per call based on {sample_size} sample blocks...")
        total_tokens_sampled = 0
        successful_samples = 0

        # --- Sampling Loop --- #
        for i in range(sample_size):
            block_info = large_blocks[i]
            try:
                logging.debug(f"Processing sample block {i+1}/{sample_size}...")
                # Perform the analysis for the sample block
                sample_result = _analyze_large_block(block_info, i)

                if sample_result:
                    sample_results_collected[i] = sample_result # Store successful result

                    # --- Estimate Tokens --- #
                    # Reconstruct the prompt sent to _analyze_large_block for input token estimation
                    # (We need the schema definition from within _analyze_large_block's scope)
                    # Define schema for block-level analysis (copy from _analyze_large_block)
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
                    truncated_block = block_info['text'][:80000] # Match truncation
                    system_msg = "You are an expert literary analyst. Analyze the provided text and extract information strictly according to the provided JSON schema. Only output JSON."
                    # Replicate the prompt structure from _analyze_large_block closely
                    block_ref_for_prompt = block_info['ref'] # Assign to temporary variable
                    user_msg = f"""
Analyze the following large text block from a document. Extract a concise summary, key entities primarily featured *in this block*, and identify any structural marker (like Chapter/Part title) found near the beginning of this block. Adhere strictly to the provided JSON schema.

JSON Schema:
{schema_str}

Text Block (Ref: {block_ref_for_prompt}):
--- START BLOCK ---
{truncated_block}
--- END BLOCK ---
(Note: Block might be truncated for analysis if excessively long)

Provide the analysis ONLY in the specified JSON format.
                    """
                    prompt_for_token_count = system_msg + user_msg

                    if encoding:
                        input_tokens = len(encoding.encode(prompt_for_token_count))
                        output_tokens = len(encoding.encode(json.dumps(sample_result)))
                    else:
                        # Rough estimate if tiktoken failed
                        input_tokens = len(prompt_for_token_count) // 4
                        output_tokens = len(json.dumps(sample_result)) // 4
                    
                    # Calculate total tokens before logging
                    total_block_tokens = input_tokens + output_tokens
                    total_tokens_sampled += total_block_tokens
                    successful_samples += 1
                    # Use the pre-calculated total in the f-string
                    logging.debug(f"Sample {i+1}: InputTokens={input_tokens}, OutputTokens={output_tokens}, Total={total_block_tokens}")
                else:
                    logging.warning(f"Sample block {i+1} analysis failed, excluding from average.")
                    # sample_results_collected[i] remains None

            except Exception as sample_exc:
                logging.warning(f"Error processing sample block {i+1}: {sample_exc}")
                sample_results_collected[i] = None # Ensure it's None if analysis fails

        # --- Calculate Average Tokens --- #
        if successful_samples > 0:
            estimated_tokens_per_call = total_tokens_sampled / successful_samples
            logging.info(f"Dynamic estimate: Average tokens per call = {estimated_tokens_per_call:.0f}")
        else:
            logging.warning("Failed to analyze any sample blocks. Cannot dynamically estimate tokens.")
            # estimated_tokens_per_call remains None
    else:
        logging.warning("No blocks available for sampling.")

    # --- Determine Max Workers --- #
    # Use fallback if estimation failed
    if estimated_tokens_per_call is None:
        dynamic_max_workers = MAX_WORKERS_FALLBACK
        logging.warning(f"Using fallback max_workers: {dynamic_max_workers}")
    else:
        # Calculate based on TPM
        calls_per_minute_tpm = GPT4O_TPM / estimated_tokens_per_call if estimated_tokens_per_call > 0 else 0
        # Calculate based on RPM (Requests Per Minute)
        # Divide RPM by a factor to get sustainable concurrent requests (e.g., 6 implies ~10s per request)
        concurrent_limit_rpm = GPT4O_RPM / WORKER_RATE_LIMIT_DIVISOR

        # Use the minimum of the two limits, apply safety factor
        calculated_max_workers = min(calls_per_minute_tpm, concurrent_limit_rpm)
        dynamic_max_workers = max(1, int(calculated_max_workers * WORKER_SAFETY_FACTOR))

        logging.info(f"Calculated dynamic max_workers: {dynamic_max_workers} "
                     f"(Based on TPM={GPT4O_TPM}, RPM={GPT4O_RPM}, "
                     f"EstTokens={estimated_tokens_per_call:.0f}, Factor={WORKER_SAFETY_FACTOR}, "
                     f"RPM Divisor={WORKER_RATE_LIMIT_DIVISOR})")
        # Optional: Add an absolute max cap if desired
        # dynamic_max_workers = min(dynamic_max_workers, ABSOLUTE_MAX_WORKERS_CAP)

    # --- End Dynamic Worker Calculation Logic --- #


    # --- 3.1 Map Phase --- #
    map_results = []

    # Add successful sample results (already analyzed)
    map_results.extend([res for res in sample_results_collected if res is not None])
    logging.info(f"Reusing results from {len(map_results)} successful sample blocks.")

    # Process remaining blocks in parallel
    remaining_blocks = large_blocks[sample_size:]
    if remaining_blocks:
        logging.info(f"Processing remaining {len(remaining_blocks)} blocks in parallel with {dynamic_max_workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=dynamic_max_workers) as executor:
            future_to_block_index = {
                # Pass the correct block index (i + sample_size)
                executor.submit(_analyze_large_block, block, i + sample_size): i + sample_size
                for i, block in enumerate(remaining_blocks)
            }

            for future in concurrent.futures.as_completed(future_to_block_index):
                original_block_index = future_to_block_index[future]
                try:
                    result = future.result()
                    if result:
                        map_results.append(result)
                except Exception as exc:
                    logging.error(f"Block analysis task for index {original_block_index} generated an exception: {exc}")

    else:
        logging.info("No remaining blocks to process after sampling.")

    # Sort results by original block index to maintain order before reduction
    map_results.sort(key=lambda x: x.get('block_index', -1))

    if not map_results:
        logging.error("Map phase failed for all blocks. Cannot proceed to Reduce phase.")
        # Return a meaningful error structure
        return {
            "error": "Failed to analyze any document blocks.",
            "document_type": "Analysis Failed",
            "structure": [],
            "overall_summary": "",
            "preliminary_key_entities": {}
        }

    logging.info(f"Map phase complete. Successfully analyzed {len(map_results)} blocks.")

    # --- 3.2 Reduce Phase --- #
    logging.info("Starting Reduce phase: Synthesizing document overview...")

    # Prepare input for the reduction prompt
    synthesis_input = ""
    structure_list_from_map = []
    consolidated_entities = {"characters": set(), "locations": set(), "organizations": set()}

    for i, result in enumerate(map_results):
        if not isinstance(result, dict):
             logging.warning(f"Skipping invalid map result at index {i}: {result}")
             continue
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

        # Consolidate entities
        entities = result.get('key_entities_in_block', {})
        if isinstance(entities, dict):
            consolidated_entities["characters"].update(entities.get("characters", []))
            consolidated_entities["locations"].update(entities.get("locations", []))
            consolidated_entities["organizations"].update(entities.get("organizations", []))
        else:
             logging.warning(f"Unexpected entity format in block {block_ref_val}: {entities}")


    # Convert sets back to lists for the final structure
    final_entities = {k: sorted(list(v)) for k, v in consolidated_entities.items()} # Sort for consistency

    # Build the reduction prompt
    reduce_prompt = f"""
    Based on the following summaries and key entities extracted from consecutive blocks of a large document, synthesize an overall analysis. Provide the overall document type, a concise overall summary, a consolidated list of preliminary key entities (most important ones overall), and refine the list of structural markers into a coherent document structure. Adhere strictly to the provided JSON schema.

    JSON Schema:
    {json.dumps(DOCUMENT_ANALYSIS_SCHEMA, indent=2)}

    Summaries & Entities from Blocks:
    --- START BLOCK DATA ---
    {synthesis_input[:100000]}
    --- END BLOCK DATA ---
    (Note: Block data may be truncated if excessively long)

    Provide the synthesized analysis ONLY in the specified JSON format. Ensure the 'structure' list is ordered correctly based on block appearance.
    """

    # Call LLM for reduction
    try:
        final_analysis = _call_openai_json_mode(reduce_prompt, DOCUMENT_ANALYSIS_SCHEMA)
        logging.info("Reduce phase complete. Synthesized document analysis.")
        # Optional: Post-process final_analysis['structure'] using structure_list_from_map
        # For now, rely on the LLM's synthesis based on the prompt.
    except Exception as e:
        logging.error(f"Reduce phase failed: {e}")
        # Fallback: return partial data or error
        final_analysis = {
            "document_type": "Unknown (Synthesis Failed)",
            "structure": structure_list_from_map, # Best guess from map phase markers
            "overall_summary": "Synthesis failed, see block summaries for details.",
            "preliminary_key_entities": final_entities, # From consolidation
            "error": f"Reduce phase failed: {str(e)}"
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