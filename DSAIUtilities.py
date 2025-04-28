#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions specific to the DataSphere AI project."""

import logging
import random
from typing import List, Dict, Any, Optional, Callable
import sys
import os
import sqlite3
import pinecone
import tiktoken
from DSAIParams import *
import traceback
import copy
import re


# Attempt to import tiktoken, fall back if unavailable
try:
    import tiktoken
except ImportError:
    logging.warning("tiktoken library not found. Token estimations will use rough character counts.")
    tiktoken = None

# Adjust path to import from parent directory (DSAI_v2_Scripts)
# This assumes DSAIUtilities.py is directly inside DSAI_v2_Scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# If DSAIUtilities is in a subdirectory, adjust parent_dir calculation accordingly
parent_dir = script_dir
sys.path.insert(0, parent_dir)

# Import required global modules
try:
    from globals import *
    from UtilityFunctions import *
    from DSAIParams import *
    from DSAIParams import AIPlatform # Explicitly import AIPlatform
    # Import prompt details if needed directly, or rely on prompt_generator_func
    # from prompts import system_msg_for_large_block_anal # Example
except ImportError as e:
    print(f"Error importing core modules (globals, UtilityFunctions, DSAIParams) in DSAIUtilities: {e}")
    raise

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def deleteAll_dbs_and_Indexes(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    try:
        # Get all table names in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("No tables found in the database.")
        else:
            confirm = input("WARNING: This will permanently delete all tables and indexes. Continue? (yes/no): ").strip().lower()
            if confirm not in ("yes", "y"):
                print("Operation canceled.")
                conn.close()
                return False

            # Delete all tables
            for (table_name,) in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                print(f"Deleted table: {table_name}")

            # Delete sqlite_sequence if it exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
            if cursor.fetchone():
                cursor.execute("DELETE FROM sqlite_sequence")
                print("Deleted sqlite_sequence (auto-increment tracking).")

            conn.commit()

        # ---- Pinecone: Permanently delete all indexes ----
        try:
            pinecone_indexes = pinecone.list_indexes()  # List all indexes
            for index_name in pinecone_indexes:
                pinecone.delete_index(index_name)  # Permanently delete index
                print(f"Deleted Pinecone index: {index_name}")

        except Exception as e:
            print(f"Error deleting Pinecone indexes: {e}")

    except Exception as e:
        print(f"Error while deleting database tables: {e}")

    finally:
        conn.close()
        print("All databases and indexes permanently removed.")
        return True

def count_tokens_in_document(document_text, emb_model_name):
    """Counts the total number of tokens in a document using OpenAI's tokenizer."""
    encoding = tiktoken.encoding_for_model(emb_model_name)  # Load the tokenizer for the model
    tokens = encoding.encode(document_text)  # Convert document text into tokens
    
    return len(tokens)  # Return total token count

def get_optimal_batch_size(total_tokens, chunk_size, max_batch_size=200):
    """Calculate the optimal batch size to ensure exactly 20 API calls while staying within OpenAI limits."""
    num_chunks = max(1, total_tokens // chunk_size)  # Ensure at least 1 chunk
    
    # Compute batch size to ensure exactly 20 API calls
    batch_size = max(1, num_chunks // 20)

    # Cap batch size if needed to prevent API slowdowns
    batch_size = min(batch_size, max_batch_size)

    return batch_size

def calc_est_tokens_per_call(
    data_list: List[Dict[str, Any]],
    num_blocks_for_sample: int,
    estimated_output_token_fraction: float,
    system_message: str,
    prompt_generator_func: Callable[[Dict[str, Any]], str],
    additional_data: Any = None
) -> Optional[float]:
    """Estimates the total tokens per LLM call based on random samples.

    Args:
        data_list: List of data items (e.g., large_blocks) to sample from.
        num_blocks_for_sample: How many items to randomly sample.
        estimated_output_token_fraction: Estimated output tokens as a fraction of input tokens.
        system_message: The system message used in the LLM call.
        prompt_generator_func: A function that takes one item from data_list
                               and returns the corresponding user prompt string.

    Returns:
        The estimated total tokens per call, or None if estimation fails.
    """
    if not data_list:
        logging.warning("Cannot estimate tokens: data_list is empty.")
        return None

    sample_size = min(num_blocks_for_sample, len(data_list))
    if sample_size <= 0:
        logging.warning("Cannot estimate tokens: num_blocks_for_sample is zero or negative.")
        return None

    logging.info(f"Estimating tokens locally based on {sample_size} random sample blocks...")
    encoding = None
    if tiktoken:
        try:
            # Explicitly get cl100k_base for OpenAI and Gemini (as a good approximation)
            if AIPlatform.upper() in ["OPENAI", "GEMINI"]:
                encoding = tiktoken.get_encoding("cl100k_base")
                logging.debug(f"Using explicit tiktoken encoding 'cl100k_base' for platform {AIPlatform}.")
            else:
                # Fallback for unknown platforms - maybe try encoding_for_model or warn
                logging.warning(f"Unsupported AIPlatform '{AIPlatform}' for tiktoken estimation. Using rough char count.")
                encoding = None # Ensure it's None
        except Exception as e:
            logging.warning(f"tiktoken get_encoding('cl100k_base') failed: {e}. Using rough char count.")
            encoding = None # Set back to None on error
    else:
         logging.debug("tiktoken not available. Using rough character count.")

    total_input_tokens_sampled = 0
    successful_samples = 0

    # Randomly select indices without replacement
    try:
        sample_indices = random.sample(range(len(data_list)), sample_size)
    except ValueError:
        logging.error("Sample size requested is larger than the data list size.")
        # Should not happen due to min() check above, but defensive coding
        return None

    for i, index in enumerate(sample_indices):
        block_info = data_list[index]
        try:
            logging.debug(f"Estimating tokens for sample {i+1}/{sample_size} (Index: {index})...")

            # Generate the user prompt for this sample block
            user_prompt = prompt_generator_func(block_info, additional_data)

            # Construct the full text that would be tokenized
            # Note: For Gemini, system message isn't part of the tokenized 'prompt',
            # but including it gives a more conservative estimate for rate limits.
            # Adjust this logic based on exact API token counting if needed.
            # For simplicity and safety, we'll include both for estimation.
            full_prompt_text = system_message + "\n\n" + user_prompt

            if encoding:
                input_tokens = len(encoding.encode(full_prompt_text))
            else:
                # Rough approximation if tiktoken failed or unavailable
                input_tokens = len(full_prompt_text) // 4

            total_input_tokens_sampled += input_tokens
            successful_samples += 1
            logging.debug(f"Sample {i+1}: Estimated Input Tokens = {input_tokens}")

        except Exception as sample_exc:
            logging.warning(f"Error estimating tokens for sample block {i+1} (Index: {index}): {sample_exc}")

    if successful_samples > 0:
        average_input_tokens = total_input_tokens_sampled / successful_samples
        estimated_output_tokens = average_input_tokens * estimated_output_token_fraction
        estimated_total_tokens_per_call = average_input_tokens + estimated_output_tokens
        logging.info(f"Token Estimate: Avg Input={average_input_tokens:.0f}, Est Output={estimated_output_tokens:.0f} (Frac: {estimated_output_token_fraction}), Est Total/Call={estimated_total_tokens_per_call:.0f}")
        return estimated_total_tokens_per_call
    else:
        logging.warning("Failed to estimate tokens for any sample blocks. Cannot proceed with dynamic worker calculation.")
        return None


import copy
import re

def fix_titles_in_names(block_analysis_result):
    fixed_result = copy.deepcopy(block_analysis_result)
    
    titles = ["Mr", "Mrs", "Miss", "Ms"]
    
    # Pattern: match title only if NOT immediately followed by a period
    title_patterns = {title: re.compile(rf'\b{title}\b(?!\.)') for title in titles}
    
    def correct_name(name):
        for title, pattern in title_patterns.items():
            name = pattern.sub(f"{title}.", name)
        return name

    characters = fixed_result.get('key_entities_in_block', {}).get('characters', [])
    
    for character in characters:
        if 'name' in character:
            character['name'] = correct_name(character['name'])
        
        if 'alternate_names' in character and isinstance(character['alternate_names'], list):
            character['alternate_names'] = [correct_name(alt_name) for alt_name in character['alternate_names']]

    return fixed_result

def remove_leading_the(block_analysis_result):
    fixed_result = copy.deepcopy(block_analysis_result)
    
    # Precompiled regex to match "the " or "The " at the start
    the_pattern = re.compile(r'^(the\s)', re.IGNORECASE)
    
    def clean_name(name):
        return the_pattern.sub('', name)
    
    # Process locations
    locations = fixed_result.get('key_entities_in_block', {}).get('locations', [])
    for location in locations:
        if 'name' in location:
            location['name'] = clean_name(location['name'])
        if 'alternate_names' in location and isinstance(location['alternate_names'], list):
            location['alternate_names'] = [clean_name(alt_name) for alt_name in location['alternate_names']]
    
    # Process organizations
    organizations = fixed_result.get('key_entities_in_block', {}).get('organizations', [])
    for organization in organizations:
        if 'name' in organization:
            organization['name'] = clean_name(organization['name'])
        if 'alternate_names' in organization and isinstance(organization['alternate_names'], list):
            organization['alternate_names'] = [clean_name(alt_name) for alt_name in organization['alternate_names']]
    
    return fixed_result

import copy

def capitalize_first_letter(block_analysis_result):
    fixed_result = copy.deepcopy(block_analysis_result)
    
    def capitalize_name(name):
        return name[:1].upper() + name[1:] if name else name

    key_entities = fixed_result.get('key_entities_in_block', {})
    
    for entity_type in ['characters', 'locations', 'organizations']:
        entities = key_entities.get(entity_type, [])
        for entity in entities:
            if 'name' in entity:
                entity['name'] = capitalize_name(entity['name'])
            if 'alternate_names' in entity and isinstance(entity['alternate_names'], list):
                entity['alternate_names'] = [capitalize_name(alt_name) for alt_name in entity['alternate_names']]
    
    return fixed_result

def capitalize_all_words(block_analysis_result):
    fixed_result = copy.deepcopy(block_analysis_result)
    
    def title_case_name(name):
        return name.title() if name else name

    key_entities = fixed_result.get('key_entities_in_block', {})
    
    for entity_type in ['characters', 'locations', 'organizations']:
        entities = key_entities.get(entity_type, [])
        for entity in entities:
            if 'name' in entity:
                entity['name'] = title_case_name(entity['name'])
            if 'alternate_names' in entity and isinstance(entity['alternate_names'], list):
                entity['alternate_names'] = [title_case_name(alt_name) for alt_name in entity['alternate_names']]
    
    return fixed_result
