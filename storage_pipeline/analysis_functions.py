#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles Steps 3 & 4: LLM-based document and chunk analysis."""

import logging
import json
from typing import List, Dict, Any, Tuple, Optional
import sys
import os
# Removed concurrent.futures, tiktoken, time, openai imports as they are handled elsewhere
from enum import Enum

# import prompts # Import the whole module

# Adjust path to import from parent directory AND sibling directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir) # Add parent DSAI_v2_Scripts

from globals import *
from UtilityFunctions import *
from DSAIParams import *
from enums_and_constants import *
from llm_calls import *
from prompts import *
from DSAIUtilities import *
# Import required global modules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Schemas moved to prompts.py

# --- REVISED: Step 3.1 - Map Phase Helper ---
def _analyze_large_block(block_info: Dict[str, Any], block_index: int, additional_data: Any=None) -> Optional[Dict[str, Any]]:
    """Analyzes a single large text block using LLM. (Now simpler)"""
    block_ref = block_info.get('ref', f'Index {block_index}')
    logging.info(f"Analyzing large block {block_index + 1}: {block_ref} (Length: {len(block_info.get('text',''))} chars)")

    # Get the user prompt using the dedicated function
    try:
        prompt = get_anal_large_block_prompt(block_info)
    except Exception as prompt_err:
        logging.error(f"Failed to generate prompt for block {block_index + 1} ({block_ref}): {prompt_err}")
        return None

    try:
        # Use the centralized function for the API call in JSON mode
        # Pass the specific system message and the generated prompt
        block_analysis_result = call_llm_json_mode(
            system_message=system_msg_for_large_block_anal,
            prompt=prompt
        )

        # Add block reference back for context in reduce step
        block_analysis_result['block_ref'] = block_ref
        block_analysis_result['block_index'] = block_index
        return block_analysis_result
    except Exception as e:
        # The parallel caller will log this exception
        logging.error(f"LLM call failed for block {block_index + 1} ({block_ref}): {e}")
        raise # Re-raise exception to be caught by parallel_llm_calls

# --- REVISED: Step 3 - Map Phase: Analyze Blocks in Parallel (Now Orchestration) ---
def perform_map_block_analysis(large_blocks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Performs the Map phase of Step 3: Orchestrates parallel analysis of large blocks.

    Args:
        large_blocks: List of coarse chunks/blocks from Step 2.

    Returns:
        A tuple containing:
        - map_results: A list of analysis results from each successfully processed block.
        - final_entities: A dictionary of consolidated entities found across all blocks.
    """
    logging.info(f"Starting Map Phase: Orchestrating analysis for {len(large_blocks)} large blocks...")

    if not large_blocks:
        logging.warning("No large blocks provided to analyze.")
        return [], {}

    # --- Step 3.0: Estimate Tokens and Calculate Workers ---
    logging.info("Estimating tokens per call for worker calculation...")

    estimated_tokens = calc_est_tokens_per_call(
        data_list=large_blocks,
        num_blocks_for_sample=NumSampleBlocksForLBA,
        estimated_output_token_fraction=EstOutputTokenFractionForLBA,
        system_message=system_msg_for_large_block_anal,
        prompt_generator_func=get_anal_large_block_prompt,
    )

    if estimated_tokens is None:
        logging.warning("Token estimation failed. Using fallback worker count.")
        num_workers = MAX_WORKERS_FALLBACK
    else:
        num_workers = calc_num_instances(estimated_tokens)

    # Limit workers by the number of blocks
    num_workers = min(num_workers, len(large_blocks))

    # --- Step 3.1: Run Parallel Analysis --- # 
    map_results_raw = parallel_llm_calls(
        function_to_run=_analyze_large_block,
        num_instances=num_workers,
        input_data_list=large_blocks,
        platform=AIPlatform,
        rate_limit_sleep=RATE_LIMIT_SLEEP_SECONDS
    )

    # --- Process Results --- #
    # Filter out None results (failures)
    map_results = [r for r in map_results_raw if r is not None]

    if not map_results:
        logging.error("Map phase failed for all blocks. Cannot proceed to Reduce phase.")
        return [], {}

    # Sort results by original block index (important for Reduce phase)
    # The analysis function already added 'block_index'
    map_results.sort(key=lambda x: x.get('block_index', -1))

    # --- Consolidate Entities from Map Results --- #
    logging.info("Consolidating entities from successful map results...")
    consolidated_entities = {"characters": set(), "locations": set(), "organizations": set()}
    for i, result in enumerate(map_results):
        # Basic type check, should be dict if successful
        if not isinstance(result, dict):
             logging.warning(f"Skipping invalid map result at index {i} after filtering: {result}")
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

# --- REVISED: Step 3 - Reduce Phase: Synthesize Document Overview (Uses getReducePrompt) ---
def perform_reduce_document_analysis(
    map_results: List[Dict[str, Any]],
    final_entities: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Performs the Reduce phase of Step 3: Synthesizes the overall document analysis.
    Uses prompts defined in prompts.py.
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

    # Prepare inputs needed for the prompt generator
    synthesis_input = ""
    num_blocks = len(map_results)
    for i, result in enumerate(map_results):
        block_ref_val = result.get('block_ref', f'Index {result.get("block_index", "Unknown")}')
        block_summary_val = result.get('block_summary', 'Summary Unavailable')
        synthesis_input += f"Block {i+1} Summary (Ref: {block_ref_val}): {block_summary_val}\n"

    try:
        formatted_entities_str = json.dumps(final_entities, indent=2)
    except Exception as json_err:
        logging.warning(f"Could not format final_entities for prompt: {json_err}")
        formatted_entities_str = "Error formatting entities." # Fallback

    # Generate the prompt using the dedicated function
    try:
        reduce_prompt = getReducePrompt(
            num_blocks=num_blocks,
            formatted_entities_str=formatted_entities_str,
            synthesis_input=synthesis_input
        )
    except Exception as prompt_err:
         logging.error(f"Failed to generate reduce prompt: {prompt_err}")
         # Return error dictionary if prompt generation fails
         return {
            "error": f"Failed during prompt generation: {prompt_err}",
            "document_type": "Analysis Failed", "structure": [], "overall_summary": "", "preliminary_key_entities": {}
         }

    # Call LLM for reduction using the centralized function
    try:
        # Use imported system message (reduce_system_message)
        final_analysis = call_llm_json_mode(
            system_message=reduce_system_message,
            prompt=reduce_prompt
        )
        logging.info("Reduce phase complete. Document overview synthesized.")
        return final_analysis
    except Exception as e:
        logging.error(f"Failed to synthesize document overview (Reduce phase): {e}")
        return {
            "error": f"Failed during final synthesis: {e}",
            "document_type": "Analysis Failed", "structure": [], "overall_summary": "", "preliminary_key_entities": {}
        }

# --- REVISED: Step 4 - Detailed Chunk Analysis (Uses imported prompts) ---
def analyze_chunk_details(block_info: Dict[str, Any], block_index: int) -> Optional[Dict[str, Any]]:
    """Analyzes a single fine-grained chunk for entities, relationships, etc."""
    logging.info(f"Analyzing details for chunk {chunk_id}...")
    chunk_text = block_info.get('text', '')
    chunk_id = block_info.get('chunk_id', f'Index {block_index}')

    # Generate the user prompt
    prompt = get_anal_chunk_details_prompt(block_info, )
    
    try:
        # Use the centralized JSON mode caller
        chunk_analysis_result = call_llm_json_mode(
            system_message=chunk_system_message, # Use imported system message
            prompt=prompt
        )
        return chunk_analysis_result
    except Exception as e:
        logging.error(f"Failed to analyze chunk details for {chunk_id}: {e}")
        # Return error structure
        return {
            "error": f"Analysis failed for chunk {chunk_id}: {e}",
            "entities": {},
            "relationships_interactions": [],
            "events": [],
            "keywords_topics": []
        }

