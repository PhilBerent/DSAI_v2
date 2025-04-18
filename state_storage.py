#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles saving and loading of intermediate pipeline state."""

import os
import pickle
import logging
import sys
from typing import List, Dict, Any, Optional, Callable

# Adjust path to import from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = script_dir # Assumes state_storage.py is directly in DSAI_v2_Scripts
sys.path.insert(0, parent_dir)

# Import required global modules
try:
    from globals import *
    from UtilityFunctions import *
    from DSAIParams import *
    from enums_and_constants import * # For potential use of constants
except ImportError as e:
    print(f"Error importing core modules in state_storage: {e}")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the directory to store state files
STATE_DIR = os.path.join(parent_dir, "_state") # Store in a subdirectory
if not os.path.exists(STATE_DIR):
    os.makedirs(STATE_DIR)
    logging.info(f"Created state storage directory: {STATE_DIR}")

def get_state_file_path(file_id: str, stage: str) -> str:
    """Constructs the standardized path for a state file."""
    filename = f"{file_id}_{stage}.pkl"
    return os.path.join(STATE_DIR, filename)

def save_state(file_id: str, stage: str, data: Any):
    """Saves the given data to a pickle file for a specific stage.

    Args:
        file_id: A unique identifier for the document being processed (e.g., filename without extension).
        stage: The name of the pipeline stage (e.g., 'Start', 'LargeBlockAnalysisCompleted').
        data: The Python object(s) to save.
    """
    file_path = get_state_file_path(file_id, stage)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Successfully saved state for stage '{stage}' (ID: {file_id}) to {file_path}")
    except Exception as e:
        logging.error(f"Error saving state for stage '{stage}' (ID: {file_id}) to {file_path}: {e}", exc_info=True)
        raise

def load_state(file_id: str, stage: str) -> Any:
    """Loads data from a pickle file for a specific stage.

    Args:
        file_id: A unique identifier for the document being processed.
        stage: The name of the pipeline stage whose state needs to be loaded.

    Returns:
        The loaded Python object(s).

    Raises:
        FileNotFoundError: If the state file for the given stage and file_id does not exist.
        Exception: For other errors during loading.
    """
    file_path = get_state_file_path(file_id, stage)
    logging.info(f"Attempting to load state for stage '{stage}' (ID: {file_id}) from {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"State file not found: {file_path}")
        raise FileNotFoundError(f"State file for stage '{stage}' (ID: {file_id}) not found at {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Successfully loaded state for stage '{stage}' (ID: {file_id})")
        return data
    except Exception as e:
        logging.error(f"Error loading state for stage '{stage}' (ID: {file_id}) from {file_path}: {e}", exc_info=True)
        raise

def check_state_exists(file_id: str, stage: str) -> bool:
    """Checks if a state file exists for a given stage and file_id."""
    file_path = get_state_file_path(file_id, stage)
    exists = os.path.exists(file_path)
    logging.debug(f"Checking existence of state file for stage '{stage}' (ID: {file_id}) at {file_path}: {exists}")
    return exists

# Example usage (can be removed or kept for testing)
if __name__ == '__main__':
    test_file_id = "example_document"
    test_stage_1 = CodeStages.Start.value # Assuming CodeStages is an Enum in enums_and_constants
    test_stage_2 = CodeStages.LargeBlockAnalysisCompleted.value

    # --- Test Saving ---
    print("--- Testing Save ---")
    data_to_save_1 = {'text': 'some document text', 'blocks': [1, 2, 3]}
    try:
        save_state(test_file_id, test_stage_1, data_to_save_1)
    except Exception as e:
        print(f"Save failed: {e}")

    data_to_save_2 = {'analysis': {'summary': '...', 'entities': []}}
    try:
        save_state(test_file_id, test_stage_2, data_to_save_2)
    except Exception as e:
        print(f"Save failed: {e}")


    # --- Test Loading ---
    print("--- Testing Load ---")
    if check_state_exists(test_file_id, test_stage_1):
        try:
            loaded_data_1 = load_state(test_file_id, test_stage_1)
            print(f"Loaded data for stage '{test_stage_1}': {loaded_data_1}")
            assert loaded_data_1 == data_to_save_1
        except Exception as e:
            print(f"Load failed for stage '{test_stage_1}': {e}")
    else:
        print(f"State file for stage '{test_stage_1}' does not exist.")

    if check_state_exists(test_file_id, test_stage_2):
        try:
            loaded_data_2 = load_state(test_file_id, test_stage_2)
            print(f"Loaded data for stage '{test_stage_2}': {loaded_data_2}")
            assert loaded_data_2 == data_to_save_2
        except Exception as e:
            print(f"Load failed for stage '{test_stage_2}': {e}")
    else:
        print(f"State file for stage '{test_stage_2}' does not exist.")


    # --- Test Loading Non-existent State ---
    print("--- Testing Load Non-existent ---")
    non_existent_stage = "NonExistentStage"
    try:
        load_state(test_file_id, non_existent_stage)
    except FileNotFoundError as e:
        print(f"Correctly caught FileNotFoundError: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    print("--- State Storage Test Complete ---") 