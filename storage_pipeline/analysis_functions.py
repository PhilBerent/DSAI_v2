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
import traceback
import collections
from collections import *

# import prompts # Import the whole module

# Adjust path to import from parent directory AND sibling directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir) # Add parent DSAI_v2_Scripts

from globals import *
from UtilityFunctions import *
from DSAIParams import *
from enums_constants_and_classes import *
from llm_calls import *
from prompts import *
from DSAIUtilities import *
# Import required global modules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Schemas moved to prompts.py

# --- REVISED: Step 3.1 - Map Phase Helper ---
def analyze_large_block(block_info: Dict[str, Any], block_index: int, additional_data: Any=None) -> Optional[Dict[str, Any]]:
    """Analyzes a single large text block using LLM. (Now simpler)"""
    block_ref = block_info.get('ref', f'Index {block_index}')
    logging.info(f"Analyzing large block {block_index + 1}: {block_ref} (Length: {len(block_info.get('text',''))} chars)")
    if (has_string(block_info, "\u2019")):
            aaa=3

    # Get the user prompt using the dedicated function
    try:
        prompt = get_anal_large_block_prompt(block_info)
    except Exception as prompt_err:
        logging.error(f"Failed to generate prompt for block {block_index + 1} ({block_ref}): {prompt_err}", exc_info=True)
        return None

    try:
        # Use the centralized function for the API call in JSON mode
        # Pass the specific system message and the generated prompt
        if (has_string(prompt, "\u2019")):
            aaa=3

        block_analysis_result = retry_function(call_llm_json_mode, system_msg_for_large_block_anal, prompt)
        if (has_string(block_analysis_result, "\u2019")):
            aaa=3

        # Add block reference back for context in reduce step
        block_analysis_result['block_ref'] = block_ref
        block_analysis_result['block_index'] = block_index
        return block_analysis_result
    except Exception as e:
        # The parallel caller will log this exception
        logging.error(f"LLM call failed for block {block_index + 1} ({block_ref}): {e}", exc_info=True)
        raise # Re-raise exception to be caught by parallel_llm_calls

# --- REVISED: Step 3 - Map Phase: Analyze Blocks in Parallel (Now Orchestration) ---
def perform_map_block_analysisOld(large_blocks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Performs the Map phase of Step 3: Orchestrates parallel analysis of large blocks.

    Args:
        large_blocks: List of coarse chunks/blocks from Step 2.

    Returns:
        A tuple containing:
        - block_info_list: A list of analysis results from each successfully processed block.
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
    if has_string(large_blocks, "\u2019"):
        aaa=3

    # --- Step 3.1: Run Parallel Analysis --- # 
    block_info_list_raw = parallel_llm_calls(
        function_to_run=analyze_large_block,
        num_instances=num_workers,
        input_data_list=large_blocks,
        platform=AIPlatform,
        rate_limit_sleep=RATE_LIMIT_SLEEP_SECONDS
    )

    # --- Process Results --- #
    # Filter out None results (failures)
    block_info_list = [r for r in block_info_list_raw if r is not None]

    if not block_info_list:
        logging.error("Map phase failed for all blocks. Cannot proceed to Reduce phase.")
        return [], {}

    # Sort results by original block index (important for Reduce phase)
    # The analysis function already added 'block_index'
    block_info_list.sort(key=lambda x: x.get('block_index', -1))

    # --- Consolidate Entities from Map Results --- #
    logging.info("Consolidating entities from successful map results...")
    consolidated_entities = {"characters": set(), "locations": set(), "organizations": set()}
    for i, result in enumerate(block_info_list):
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

    return block_info_list, final_entities

def perform_map_block_analysis(large_blocks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Performs the Map phase of Step 3: Orchestrates parallel analysis of large blocks.

    Args:
        large_blocks: List of coarse chunks/blocks from Step 2.

    Returns:
        A tuple containing:
        - block_info_list: A list of analysis results from each successfully processed block.
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
    if has_string(large_blocks, "\u2019"):
        aaa=3

    # --- Step 3.1: Run Parallel Analysis --- # 
    block_info_list_raw = parallel_llm_calls(
        function_to_run=analyze_large_block,
        num_instances=num_workers,
        input_data_list=large_blocks,
        platform=AIPlatform,
        rate_limit_sleep=RATE_LIMIT_SLEEP_SECONDS
    )

    # --- Process Results --- #
    # Filter out None results (failures)
    block_info_list = [r for r in block_info_list_raw if r is not None]

    if not block_info_list:
        logging.error("Map phase failed for all blocks. Cannot proceed to Reduce phase.")
        return [], {}

    # Sort results by original block index (important for Reduce phase)
    # The analysis function already added 'block_index'
    block_info_list.sort(key=lambda x: x.get('block_index', -1))

    logging.info("Consolidating entities from successful map results...")
    consolidated_entities = {"characters": set(), "locations": set(), "organizations": set()}
    for i, result in enumerate(block_info_list):
        if not isinstance(result, dict):
            logging.warning(f"Skipping invalid map result at index {i} after filtering: {result}")
            continue
        block_ref_val = result.get('block_ref', f'Index {result.get("block_index", "Unknown")}')
        entities = result.get('key_entities_in_block', {})
        if isinstance(entities, dict):
            # Extract only the 'name' field from each entity
            character_names = [char.get("name") for char in entities.get("characters", []) if isinstance(char, dict) and "name" in char]
            location_names = [loc.get("name") for loc in entities.get("locations", []) if isinstance(loc, dict) and "name" in loc]
            organization_names = [org.get("name") for org in entities.get("organizations", []) if isinstance(org, dict) and "name" in org]
            
            consolidated_entities["characters"].update(character_names)
            consolidated_entities["locations"].update(location_names)
            consolidated_entities["organizations"].update(organization_names)
        else:
            logging.warning(f"Unexpected entity format in block {block_ref_val}: {entities}")

    # Convert sets back to sorted lists for the final structure
    full_entities_list = {k: sorted(list(v)) for k, v in consolidated_entities.items()}

    return block_info_list, full_entities_list

# --- REVISED: Step 3 - Reduce Phase: Synthesize Document Overview (Uses getReducePrompt) ---
def perform_reduce_document_analysis(block_info_list: List[Dict[str, Any]], 
            final_entities: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Performs the Reduce phase of Step 3: Synthesizes the overall document analysis.
    Uses prompts defined in prompts.py.
    Args:
        block_info_list: The list of analysis results from the Map phase.
        final_entities: The dictionary of consolidated entities from the Map phase.

    Returns:
        The synthesized document analysis result matching DOCUMENT_ANALYSIS_SCHEMA,
        or an error dictionary if reduction fails.
    """
    logging.info("Starting Reduce phase: Synthesizing document overview with type-specific instructions...")

    if not block_info_list:
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
    num_blocks = len(block_info_list)
    for i, result in enumerate(block_info_list):
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
        reduce_prompt = getReducePrompt(num_blocks, formatted_entities_str, synthesis_input)
    except Exception as prompt_err:
         logging.error(f"Failed to generate reduce prompt: {prompt_err}", exc_info=True)
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
        logging.error(f"Failed to synthesize document overview (Reduce phase): {e}", exc_info=True)
        return {
            "error": f"Failed during final synthesis: {e}",
            "document_type": "Analysis Failed", "structure": [], "overall_summary": "", "preliminary_key_entities": {}
        }

# --- REVISED: Step 4 - Detailed Chunk Analysis (Uses imported prompts) ---
def analyze_chunk_details(block_info: Dict[str, Any], block_index: int, 
            doc_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Analyzes a single fine-grained chunk for entities, relationships, etc."""
    chunk_id = block_info.get('chunk_id', f'Index {block_index}')
    logging.info(f"Analyzing details for chunk {chunk_id}...")
    
    # Generate the user prompt
    prompt = get_anal_chunk_details_prompt(block_info, doc_context)
    
    try:
        # Use the centralized JSON mode caller
        chunk_analysis_result = call_llm_json_mode(
            system_message=chunk_system_message, # Use imported system message
            prompt=prompt
        )
        return chunk_analysis_result
    except Exception as e:
        logging.error(f"Failed to analyze chunk details for {chunk_id}: {e}", exc_info=True)
        # Return error structure
        return {
            "error": f"Analysis failed for chunk {chunk_id}: {e}",
            "entities": {},
            "relationships_interactions": [],
            "events": [],
            "keywords_topics": []
        }

def worker_analyze_chunk(chunk_item: Dict[str, Any], block_index: int, 
    doc_analysis: Dict[str, Any]) -> Dict[str, Any]:
    errorCount = 1
    executionError = False
    errorMessage = ""
    tb_str = ""
    while errorCount <= 3:
        try:
        # Call the original analysis function
            analysis_result = analyze_chunk_details(
                block_info=chunk_item,
                block_index=block_index,
                doc_context=doc_analysis # Access outer scope variable
            )
            # Return the original item updated with the result
            chunk_item['analysis'] = analysis_result
            chunk_item['analysis_status'] = 'success' # Mark success
            if executionError:
                logging.info(f"Worker recovered from error for chunk {chunk_item['chunk_id']} \
                    after {errorCount} retries. {errorMessage}\n{tb_str}")
            return chunk_item
        except Exception as errorMessage:
            executionError = True
            tb_str = traceback.format_exc()
            chunk_id = chunk_item.get('chunk_id', 'UNKNOWN_ID')
            logging.error(f"Worker failed for chunk {chunk_id}: {errorMessage}\n{tb_str}", exc_info=True)
            # Return the original item marked with an error
            chunk_item['analysis'] = None
            chunk_item['analysis_status'] = 'error'
            chunk_item['analysis_error'] = str(errorMessage)
            chunk_item['traceback'] = tb_str
            errorCount += 1
            # sleep for a short duration before retrying
            time.sleep(2)
        
        logging.debug(f"exeuction failed for chunk {chunk_item['chunk_id']} after \
            {errorCount} retries: {errorMessage}\n{tb_str}")    
        return chunk_item

def consolidate_entity_informationOld(block_info_list):
    def consolidate_entity_data(entity_list, entity_dict):
        for entity in entity_list:
            name = entity["name"]
            alt_names = set(entity.get("alternate_names", []))
            desc = entity.get("description", "").strip()

            # Initialize structure if name not yet seen
            if name not in entity_dict:
                entity_dict[name] = {
                    "alternate_names": set(),
                    "description_list": set()
                }

            entity_dict[name]["alternate_names"].update(alt_names)
            if desc:
                entity_dict[name]["description_list"].add(desc)

    raw_entities_data = {
        "characters": {},
        "locations": {},
        "organizations": {}
    }

    for block in block_info_list:
        key_entities = block.get("key_entities_in_block", {})
        consolidate_entity_data(key_entities.get("characters", []), raw_entities_data["characters"])
        consolidate_entity_data(key_entities.get("locations", []), raw_entities_data["locations"])
        consolidate_entity_data(key_entities.get("organizations", []), raw_entities_data["organizations"])

    # Convert sets to lists and remove duplicates
    for category in raw_entities_data:
        for name in raw_entities_data[category]:
            raw_entities_data[category][name]["alternate_names"] = sorted(list(raw_entities_data[category][name]["alternate_names"]))
            raw_entities_data[category][name]["description_list"] = sorted(list(raw_entities_data[category][name]["description_list"]))

    return raw_entities_data

def consolidate_entity_information(block_info_list):
    """
    Consolidates entity information from a list of block analyses.

    Args:
        block_info_list: A list where each element conforms to BLOCK_ANALYSIS_SCHEMA.

    Returns:
        A dictionary categorizing entities ('characters', 'locations', 'organizations').
        Each category contains a dictionary where keys are unique entity names.
        The value for each entity name is a dictionary containing:
          - 'block_list': A sorted list of block indices where the primary name appeared.
          - 'alternate_names': A list of dictionaries, each with 'alternate_name'
                               and 'block_list' (indices where it appeared).
          - 'descriptions': A list of dictionaries, each with 'description'
                            and 'block_list' (indices where it appeared).
    """
    def consolidate_entity_data(entity_list, entity_dict, block_index):
        """Updates the entity dictionary with data from a single block."""
        for entity in entity_list:
            name = entity["name"].strip()
            if not name: # Skip entities with empty names
                continue

            alt_names = entity.get("alternate_names", [])
            desc = entity.get("description", "").strip()

            # Initialize structure if name not yet seen
            if name not in entity_dict:
                entity_dict[name] = {
                    "alt_names_map": collections.defaultdict(set),
                    "descriptions_map": collections.defaultdict(set),
                    "primary_name_blocks": set() # Added set to track primary name blocks
                }

            # --- Track primary name occurrence ---
            entity_dict[name]["primary_name_blocks"].add(block_index) # Add current block index

            # Store alternate names with block index
            for alt_name in alt_names:
                cleaned_alt_name = alt_name.strip()
                if cleaned_alt_name: # Avoid empty strings
                    entity_dict[name]["alt_names_map"][cleaned_alt_name].add(block_index)

            # Store description with block index
            if desc:
                entity_dict[name]["descriptions_map"][desc].add(block_index)

    raw_entities_data = {
        "characters": {},
        "locations": {},
        "organizations": {}
    }

    # Process each block and track its index
    for block_index, block in enumerate(block_info_list):
        key_entities = block.get("key_entities_in_block", {})
        consolidate_entity_data(key_entities.get("characters", []), raw_entities_data["characters"], block_index)
        consolidate_entity_data(key_entities.get("locations", []), raw_entities_data["locations"], block_index)
        consolidate_entity_data(key_entities.get("organizations", []), raw_entities_data["organizations"], block_index)

    # --- Format the output ---
    formatted_entities_data = {
        "characters": {},
        "locations": {},
        "organizations": {}
    }

    for category in raw_entities_data:
        for name, data in raw_entities_data[category].items():
            # Format alternate names
            formatted_alt_names = []
            for alt_name, block_indices in data["alt_names_map"].items():
                formatted_alt_names.append({
                    "alternate_name": alt_name,
                    "block_list": sorted(list(block_indices))
                })
            formatted_alt_names.sort(key=lambda x: x["alternate_name"]) # Sort alphabetically

            # Format descriptions
            formatted_descriptions = []
            for description, block_indices in data["descriptions_map"].items():
                formatted_descriptions.append({
                    "description": description,
                    "block_list": sorted(list(block_indices))
                })
            formatted_descriptions.sort(key=lambda x: x["description"]) # Sort alphabetically

            # --- Format primary name block list ---
            primary_block_list = sorted(list(data["primary_name_blocks"]))

            # --- Assemble final dictionary for the entity ---
            formatted_entities_data[category][name] = {
                "block_list": primary_block_list, # Added primary name block list
                "alternate_names": formatted_alt_names,
                "descriptions": formatted_descriptions
            }

    return formatted_entities_data

def get_primary_entity_names(prelim_entity_data):
    # prelim_entity_data is a dictionary with keys 'characters', 'locations', and 'organizations' the values of each of these are also dictionaries for which the keys are the names of the entities. Return a dictionary 'prelim_primary_names' where the keys are 'characters', 'locations', and 'organizations' and the values are the keys of each of the corresponding dictionaries in 'prelim_entity_data'.
    prelim_primary_names = {
        "characters": list(prelim_entity_data["characters"].keys()),
        "locations": list(prelim_entity_data["locations"].keys()),
        "organizations": list(prelim_entity_data["organizations"].keys())
    }
    return prelim_primary_names
