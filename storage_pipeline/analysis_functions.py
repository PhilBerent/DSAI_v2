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
from nameFunctions import *
# Import required global modules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Schemas moved to prompts.py

# --- REVISED: Step 3.1 - Map Phase Helper ---
def analyze_large_block(block_info: Dict[str, Any], block_index: int, additional_data: Any=None) -> Optional[Dict[str, Any]]:
    """Analyzes a single large text block using LLM. (Now simpler)"""
    block_ref = block_info.get('ref', f'Index {block_index}')
    logging.info(f"Analyzing large block {block_index + 1}: {block_ref} (Length: {len(block_info.get('text',''))} chars)")
    # Get the user prompt using the dedicated function

    try:
        prompt = get_anal_large_block_prompt(block_info)
    except Exception as prompt_err:
        logging.error(f"Failed to generate prompt for block {block_index + 1} ({block_ref}): {prompt_err}", exc_info=True)
        return None

    try:
        # Use the centralized function for the API call in JSON mode
        # Pass the specific system message and the generated prompt

        block_analysis_result = retry_function(call_llm_json_mode, system_msg_for_large_block_anal, prompt, numRetries=7)
        block_analysis_result = fix_titles_in_names(block_analysis_result) # ensures that all titles are followed by a "."
        block_analysis_result = remove_leading_the(block_analysis_result) # removes leading "the" from names
        block_analysis_result = capitalize_all_words(block_analysis_result) # capitalizes the first letter of all words in names

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

def get_block_info_list(large_blocks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
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

    return block_info_list

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

def getIsAnAltNameDict(prelim_entity_data, primary_names_dict):
    """
    Returns a dictionary where the keys are the entity types and the values are dictionaries
    where the keys are the alternate names and the values are dictionaries with keys 'primary_names' and 'indexes'.
    The value of 'primary_names' field is a list of names.
    """
    is_an_alt_name_of_dict = {}
    for entity_type in prelim_entity_data:
        entity_data = prelim_entity_data[entity_type]
        entityDict = primary_names_dict[entity_type]
        thisEntAltNameDict = is_an_alt_name_of_dict[entity_type] = {}
        numEntities = len(entity_data)
        for i in range(numEntities):
            entity = entity_data[i]
            name = entity['name']            
            alternateNameList = entity['alternate_names']
            numAltNames = len(alternateNameList)
            for j in range(numAltNames):
                alt_name = alternateNameList[j].get('alternate_name', '')
                if alt_name not in thisEntAltNameDict:
                    thisEntAltNameDict[alt_name] = {"primary_names": [], "indexes": []}
                thisEntAltNameDict[alt_name]['primary_names'].append(name)
                thisEntAltNameDict[alt_name]['indexes'].append(i)

    return is_an_alt_name_of_dict


def consolidate_entity_information(block_info_list):
    """
    Consolidates entity information from a list of block analyses.

    Returns:
        - formatted_entities_data: a dict where 'characters', 'locations', and 'organizations'
          are lists of entity dicts (each with 'name', 'block_list', 'alternate_names', 'descriptions').
        - primary_name_dict: maps each primary name to its index in the list.
        - alternate_names_dict: maps each alternate name to a list of indices where it appears.
    """

    def consolidate_entity_data(entity_list, entity_data_by_name, block_index):
        """Updates the entity dictionary keyed by primary name."""
        for entity in entity_list:
            name = entity["name"].strip()
            if not name:
                continue

            alt_names = entity.get("alternate_names", [])
            desc = entity.get("description", "")
            if desc:
                desc = desc.strip()

            if name not in entity_data_by_name:
                entity_data_by_name[name] = {
                    "primary_name_blocks": set(),
                    "alt_names_map": collections.defaultdict(set),
                    "descriptions_map": collections.defaultdict(set)
                }

            entity_data_by_name[name]["primary_name_blocks"].add(block_index)

            for alt_name in alt_names:
                cleaned_alt_name = alt_name.strip()
                if cleaned_alt_name:
                    entity_data_by_name[name]["alt_names_map"][cleaned_alt_name].add(block_index)

            if desc:
                entity_data_by_name[name]["descriptions_map"][desc].add(block_index)

    # --- Prepare raw structures ---
    raw_entities_data = {
        "characters": {},
        "locations": {},
        "organizations": {}
    }

    for block_index, block in enumerate(block_info_list):
        key_entities = block.get("key_entities_in_block", {})
        consolidate_entity_data(key_entities.get("characters", []), raw_entities_data["characters"], block_index)
        consolidate_entity_data(key_entities.get("locations", []), raw_entities_data["locations"], block_index)
        consolidate_entity_data(key_entities.get("organizations", []), raw_entities_data["organizations"], block_index)

    # --- Now transform into list-based structures ---
    prelim_entity_data = {
        "characters": [],
        "locations": [],
        "organizations": []
    }
    primary_name_dict = {
        "characters": {},
        "locations": {},
        "organizations": {}
    }
    # alt_names_dict = {
    #     "characters": {
    #         "alt_name": {
    #             "primary_names": [],
    #             "indexes": []
    #         }
    #     },
    #     "locations": {
    #         "alt_name": {
    #             "primary_names": [],
    #             "indexes": []
    #         }
    #     },
    #     "organizations": {
    #         "alt_name": {
    #             "primary_names": [],
    #             "indexes": []
    #         }
    #     }
    # }

    for category in raw_entities_data:
        name_to_entity_data = raw_entities_data[category]
        for name, data in name_to_entity_data.items():
            # Format alternate names
            formatted_alt_names = []
            for alt_name, block_indices in data["alt_names_map"].items():
                formatted_alt_names.append({
                    "alternate_name": alt_name,
                    "block_list": sorted(list(block_indices))
                })
            formatted_alt_names.sort(key=lambda x: x["alternate_name"])

            # Format descriptions
            formatted_descriptions = []
            for description, block_indices in data["descriptions_map"].items():
                formatted_descriptions.append({
                    "description": description,
                    "block_list": sorted(list(block_indices))
                })
            formatted_descriptions.sort(key=lambda x: x["description"])

            primary_block_list = sorted(list(data["primary_name_blocks"]))

            # --- Create final entity dictionary ---
            entity_entry = {
                "name": name,
                "block_list": primary_block_list,
                "alternate_names": formatted_alt_names,
                "descriptions": formatted_descriptions
            }

            # --- Update main list ---
            list_index = len(prelim_entity_data[category])
            prelim_entity_data[category].append(entity_entry)

            # --- Update primary name dictionary ---
            primary_name_dict[category][name] = list_index

            # --- Update alternate name dictionary ---
    #         for alt_name_entry in formatted_alt_names:
    #             alternate_name = alt_name_entry["alternate_name"]
    #             if alternate_name not in alt_names_dict[category]:
    #                 alt_names_dict[category][alternate_name] = {
    #                     "primary_names": [],
    #                     "indexes": []
    #                 }

    #             alt_names_dict[category][alternate_name]["primary_names"].append(name)
    #             alt_names_dict[category][alternate_name]["indexes"].append(list_index)

    # # Turn alternate_names_dict values into normal dicts
    # alt_names_dict = {
    #     cat: dict(inner) for cat, inner in alt_names_dict.items()
    # }
    alt_names_dict = getIsAnAltNameDict(prelim_entity_data, primary_name_dict)
    cmd = CharacterMatchData(prelim_entity_data)

    return prelim_entity_data, primary_name_dict, alt_names_dict, cmd

def get_primary_entity_namesOld(prelim_entity_data):
    # prelim_entity_data is a dictionary with keys 'characters', 'locations', and 'organizations' the values of each of these are also dictionaries for which the keys are the names of the entities. Return a dictionary 'prelim_primary_names' where the keys are 'characters', 'locations', and 'organizations' and the values are a list of tuples with the first element being the keys of each of the elements in the corresponding dictionaries in 'prelim_entity_data', and the second element being the length of the block_list coreresponding to the keys in 'prelim_entity_data'. The results should be sorted in descending order of the length of the block_list. 
    i=1
    prelim_primary_names = {}
    for entity_type in prelim_entity_data:
        prelim_primary_names[entity_type] = sorted(
            prelim_entity_data[entity_type].items(),
            key=lambda x: len(x[1]['block_list']),
            reverse=True
        )
        # Convert to list of tuples (name, block_list_length)
        prelim_primary_names[entity_type] = [(name, len(data['block_list'])) for name, data in prelim_primary_names[entity_type]]
        # Sort by block_list length in descending order
        prelim_primary_names[entity_type].sort(key=lambda x: x[1], reverse=True)
    return prelim_primary_names

def get_primary_entity_names(prelim_entity_data, is_alt_name_dict_in):
    prelim_primary_names = {}
    for entity_type in prelim_entity_data:
        entities = prelim_entity_data[entity_type]

        prelim_primary_names[entity_type] = sorted(
            [
                (entity['name'], len(entity['block_list']))
                for entity in entities
            ],
            key=lambda x: x[1],
            reverse=True
        )
    # create a dictionary 'prim_names_dict' where the keys are the names of the entities and the are the index of where that name appears in prelim_primary_names
    primary_names_dict = {}
    for entity_type in prelim_primary_names:
            primary_names_dict[entity_type] = {}
            for index, (name, _) in enumerate(prelim_primary_names[entity_type]):
                primary_names_dict[entity_type][name] = index
    
    
    # alt_names_dict_in is a dictionary where the keys are the entity types and the values are dictionaries where the keys are the alternate names and the values are dictionaries with keys 'primary_names' and 'indexes'. The value of 'primary_names' field is a list of names. Create a dictionary 'alt_names_dict_out' where the keys are the keys for the corresponding type, and the values are lists containing the indexes of the elements of the 'primary_names' field in the prelim_primary_names list. 
    is_an_alt_name_of_dict = {}
    for entity_type in is_alt_name_dict_in:
        alt_name_dict = is_alt_name_dict_in[entity_type]
        is_an_alt_name_of_dict[entity_type] = {}
        ent_prim_names_dict = primary_names_dict[entity_type]
        
        for alt_name, data in alt_name_dict.items():
            is_an_alt_name_of_dict[entity_type][alt_name] = []
            primary_names_list = data['primary_names']
            num_primary_names = len(primary_names_list) 
            for i in range(num_primary_names):
                name = primary_names_list[i]
                index_in_prelim_primary_names = ent_prim_names_dict[name]
                is_an_alt_name_of_dict[entity_type][alt_name].append(index_in_prelim_primary_names)

    has_alt_names_dict = {}
    # create has_alt_names_dict = {}. For every name in the prelim_primary_names list where 
    # (1) alternate_names[] is not empty & (2) at least one of the alternate_names in in primary_name_dict then the key is the name field in prelim_primary_names and the value is a list of the numbers resulting from looking up the alternate_names in primary_names_dict
    for entity_type in prelim_entity_data:
        has_alt_names_dict[entity_type] = {}
        for entity in prelim_entity_data[entity_type]:
            name_dict = primary_names_dict[entity_type]
            alt_name_index_list = []
            name = entity['name']
            alternate_names_list = entity['alternate_names']
            num_alt_names = len(alternate_names_list)
            if num_alt_names == 0:
                continue
            # Check if any of the alternate names are in primary_names_dict
            for i in range(num_alt_names):
                alt_name = alternate_names_list[i]['alternate_name']
                if alt_name in name_dict:
                    index_in_primary_names = name_dict[alt_name]
                    # If it does, check if any of the alternate names are in primary_names_dict
                    alt_name_index_list.append(index_in_primary_names)    
            has_alt_names_dict[entity_type][name] = alt_name_index_list
    
    return prelim_primary_names, primary_names_dict, is_an_alt_name_of_dict, has_alt_names_dict


class CharacterMatchData:
    def __init__(self, primary_entity_data):
        self.full_names_dict = {}
        self.name_no_title_dict = {}
        self.first_names_dict = {}
        self.last_names_dict = {}
        self.first_and_last_names_dict = {}
        self.name_details_dict = {}
        self.name_details_list = []
        self.full_name_list = []

        self.full_name_list = [item['name'] for item in primary_entity_data['characters']]

        num_names = len(self.full_name_list)

        for i in range(num_names):
            name = self.full_name_list[i]
            name_details = NameDetails(name)

            self.name_details_dict[name] = name_details
            self.name_details_list.append(name_details)

            first_name = name_details.first_name
            last_name = name_details.last_name
            name_no_title = name_details.name_no_title
            first_and_last_name = name_details.first_and_last_name

            self.full_names_dict[name] = i
            if first_name:
                self.first_names_dict.setdefault(first_name, []).append(i)
            if last_name:
                self.last_names_dict.setdefault(last_name, []).append(i)
            if name_no_title:
                self.name_no_title_dict.setdefault(name_no_title, []).append(i)
            if first_and_last_name:
                self.first_and_last_names_dict.setdefault(first_and_last_name, []).append(i)
    
    # db delete this function
    def remove_name(self, name: str):
        # Remove the name itself from all direct mappings
        self.full_names_dict.pop(name, None)
        self.name_details_dict.pop(name, None)

        # Retrieve the NameDetails instance to access its components
        name_details = NameDetails(name)

        self.name_no_title_dict.pop(name_details.name_no_title, None)
        self.first_names_dict.pop(name_details.first_name, None)
        self.last_names_dict.pop(name_details.last_name, None)
        self.first_and_last_names_dict.pop(name_details.first_and_last_name, None)

        # Remove from name_details_list and full_name_list
        self.name_details_list = [nd for nd in self.name_details_list if nd.input_name != name]
        self.full_name_list = [n for n in self.full_name_list if n != name]
    # ed
    def as_dict(self):
        return {
            "full_names_dict": self.full_names_dict,
            "name_no_title_dict": self.name_no_title_dict,
            "first_names_dict": self.first_names_dict,
            "last_names_dict": self.last_names_dict,
            "first_and_last_names_dict": self.first_and_last_names_dict,
            "full_name_list": self.full_name_list,
            "name_details_list": self.name_details_list,
            "name_details_dict": self.name_details_dict
        }
# db
# def getPrimaryCharsMatchDict(primary_entity_data):
#     # create a dictionary 'character_name_match_dict' where the keys are the names of the characters and the values are the indexes of the elements in the prelim_primary_names list. The results should be sorted in descending order of the length of the block_list. 
#     character_name_match_dict = {}
#     full_names_dict = {}
#     name_no_title_dict = {}
#     first_names_dict = {}    
#     last_names_dict = {}
#     first_and_last_names_dict = {}
#     nameDetailsDict = {}
#     nameDetailsList = []
#     fullNameList = [name for name, _ in primary_entity_data['characters']]
#     num_names = len(fullNameList)
#     for i in range(num_names):
#         name = fullNameList[i]
#         name_details = NameDetails(name)
#         nameDetailsDict[name] = name_details
#         nameDetailsList.append(name_details)
#         first_name = name_details.first_name
#         last_name = name_details.last_name
#         name_no_title = name_details.name_no_title
#         first_and_last_name = name_details.first_and_last_name
#         full_names_dict[name] = i
#         if first_name:
#             # Check if first name is already in the dictionary
#             if first_name in first_names_dict:
#                 # If it is, append the index to the list
#                 first_names_dict[first_name].append(i)
#             else:
#                 # If it isn't, create a new list with the index
#                 first_names_dict[first_name] = [i]
#         if last_name:
#             # Check if last name is already in the dictionary
#             if last_name in last_names_dict:
#                 # If it is, append the index to the list
#                 last_names_dict[last_name].append(i)
#             else:
#                 # If it isn't, create a new list with the index
#                 last_names_dict[last_name] = [i]
#         if name_no_title:
#             # Check if name without title is already in the dictionary
#             if name_no_title in name_no_title_dict:
#                 # If it is, append the index to the list
#                 name_no_title_dict[name_no_title].append(i)
#             else:
#                 # If it isn't, create a new list with the index
#                 name_no_title_dict[name_no_title] = [i]
#         if first_and_last_name:
#             # Check if name without title is already in the dictionary
#             if first_and_last_name in first_and_last_names_dict:
#                 # If it is, append the index to the list
#                 first_and_last_names_dict[first_and_last_name].append(i)
#             else:
#                 # If it isn't, create a new list with the index
#                 first_and_last_names_dict[first_and_last_name] = [i]
        
#         # Add the name to the character_name_match_dict
#         character_name_match_dict['full_names_dict'] = full_names_dict
#         character_name_match_dict['name_no_title_dict'] = name_no_title_dict
#         character_name_match_dict['first_names_dict'] = first_names_dict
#         character_name_match_dict['last_names_dict'] = last_names_dict
#         character_name_match_dict['first_and_last_names_dict'] = first_and_last_names_dict
#         character_name_match_dict['full_name_list'] = fullNameList
#         character_name_match_dict['name_details_list'] = nameDetailsList
#         character_name_match_dict['name_details_dict'] = nameDetailsDict

#     return character_name_match_dict
# ed
def removeFromCharNameMatchDict(character_name_match_dict, name):
    # remove the name from the character_name_match_dict
    if name in character_name_match_dict['full_names_dict']:
        del character_name_match_dict['full_names_dict'][name]
    if name in character_name_match_dict['name_no_title_dict']:
        del character_name_match_dict['name_no_title_dict'][name]
    if name in character_name_match_dict['first_names_dict']:
        del character_name_match_dict['first_names_dict'][name]
    if name in character_name_match_dict['last_names_dict']:
        del character_name_match_dict['last_names_dict'][name]
    if name in character_name_match_dict['first_and_last_names_dict']:
        del character_name_match_dict['first_and_last_names_dict'][name]
    if name in character_name_match_dict['parsed_names_dict']:
        del character_name_match_dict['parsed_names_dict'][name]
        
    full_names_dict = character_name_match_dict['full_names_dict']
    name_no_title_dict = character_name_match_dict['name_no_title_dict']
    first_names_dict = character_name_match_dict['first_names_dict']
    last_names_dict = character_name_match_dict['last_names_dict']
    first_and_last_names_dict = character_name_match_dict['first_and_last_names_dict']
    parsed_names_dict = character_name_match_dict['parsed_names_dict']

    return character_name_match_dict, full_names_dict, name_no_title_dict, first_names_dict, \
        last_names_dict, first_and_last_names_dict, parsed_names_dict  