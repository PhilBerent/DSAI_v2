import json
import os
import gzip # Import gzip for compression
from typing import Any, Dict
from UtilityFunctions import *
from globals import *
from DSAIParams import *

# Assuming globals.py and enums_and_constants.py are in the same directory or sys.path is configured
from globals import StateStorageDirectory, LargeBlockAnalysisCompletedFile, IterativeAnalysisCompletedFile
from enums_and_constants import StateStoragePoints, CodeStages

def _get_file_path(storage_point: StateStoragePoints) -> str:
    """Maps a StateStoragePoints enum to its corresponding file path."""
    if storage_point == StateStoragePoints.LargeBlockAnalysisCompleted:
        return LargeBlockAnalysisCompletedFile
    elif storage_point == StateStoragePoints.IterativeAnalysisCompleted:
        return IterativeAnalysisCompletedFile
    else:
        # Should not happen if enums are used correctly
        raise ValueError(f"Unknown StateStoragePoints: {storage_point}")

def save_state(data_to_save: Dict[str, Any], storage_point: StateStoragePoints):
    """
    Saves the provided data to a compressed JSON file corresponding to the storage point.

    Args:
        data_to_save: The dictionary containing the state to save.
        storage_point: The enum indicating which state point is being saved.
    """
    file_path = _get_file_path(storage_point)
    os.makedirs(StateStorageDirectory, exist_ok=True) # Ensure the directory exists
    try:
        # Serialize data to JSON string first
        json_string = json.dumps(data_to_save, indent=4, ensure_ascii=False)
        # Encode JSON string to bytes
        json_bytes = json_string.encode('utf-8')

        # Write compressed bytes to file
        with gzip.open(file_path, 'wb') as f_gz:
            f_gz.write(json_bytes)

        print(f"State successfully saved and compressed to {file_path}")
    except Exception as e:
        print(f"Error saving compressed state to {file_path}: {e}")

def load_state(run_from: CodeStages) -> Dict[str, Any]:
    """
    Loads and decompresses state from a JSON file corresponding to the run_from point.

    Args:
        run_from: The enum indicating which state point to load from.

    Returns:
        A dictionary containing the loaded state.

    Raises:
        FileNotFoundError: If the required state file does not exist.
        ValueError: If an invalid RunFromType is provided.
    """
    if run_from == CodeStages.LargeBlockAnalysisCompleted:
        file_path = LargeBlockAnalysisCompletedFile
    elif run_from == CodeStages.IterativeAnalysisCompleted:
        file_path = IterativeAnalysisCompletedFile
    else:
        # Loading state is only valid for specific resume points
        raise ValueError(f"Cannot load state for RunFromType: {run_from}. Only specific resume points are supported.")

    try:
        # Read compressed bytes from file
        with gzip.open(file_path, 'rb') as f_gz:
            json_bytes = f_gz.read()

        # Decode bytes back to JSON string
        json_string = json_bytes.decode('utf-8')
        # Deserialize JSON string
        loaded_data = json.loads(json_string)

        print(f"State successfully loaded and decompressed from {file_path}")
        return loaded_data
    except FileNotFoundError:
        print(f"Error: Compressed state file not found at {file_path}. Cannot resume from {run_from}.")
        raise
    except gzip.BadGzipFile:
        print(f"Error: File at {file_path} is not a valid gzip file. It might be an uncompressed old state file.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from decompressed state file {file_path}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred loading compressed state from {file_path}: {e}")
        raise

# Example usage (updated for compression)
if __name__ == '__main__':
    # Example data structure for LargeBlockAnalysisCompleted
    example_map_data = {
        'large_blocks': [{'id': 0, 'text': 'Block 1...', 'metadata': {'start_char': 0, 'end_char': 1000}}],
        'map_results': [{'analysis': {'summary': 'Summary 1'}, 'metadata': {'tokens_used': 100}}],
        'final_entities': {'PERSON': ['Alice', 'Bob']}
    }

    # Example data structure for IterativeAnalysisCompleted
    example_reduce_data = {
        'doc_analysis_result': {
            'structure': {'books': [], 'chapters': []},
            'preliminary_key_entities': {'PERSON': ['Alice', 'Bob', 'Charlie']},
            'overall_summary': 'This is the final summary.'
        }
    }

    # Test saving (compressed)
    try:
        save_state(example_map_data, StateStoragePoints.LargeBlockAnalysisCompleted)
        save_state(example_reduce_data, StateStoragePoints.IterativeAnalysisCompleted)
    except Exception as e:
        print(f"Compressed save test failed: {e}")

    # Test loading (decompressed)
    try:
        loaded_map_state = load_state(CodeStages.LargeBlockAnalysisCompleted)
        print("\nLoaded LargeBlockAnalysisCompleted State (Decompressed):")
        # print(json.dumps(loaded_map_state, indent=2))

        loaded_reduce_state = load_state(CodeStages.IterativeAnalysisCompleted)
        print("\nLoaded IterativeAnalysisCompleted State (Decompressed):")
        # print(json.dumps(loaded_reduce_state, indent=2))

    except FileNotFoundError:
        print("Compressed load test failed: State file not found (might be the first run).")
    except Exception as e:
        print(f"Compressed load test failed: {e}") 