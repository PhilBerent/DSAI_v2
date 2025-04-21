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

def save_state(data_to_save: Dict[str, Any], file_path: str):
    os.makedirs(StateStorageDirectory, exist_ok=True)  # Ensure the directory exists
    try:
        # Serialize data to JSON with ASCII-only encoding
        json_string = json.dumps(data_to_save, indent=4, ensure_ascii=True)

        # Encode JSON string to bytes (UTF-8 is fine since everything is now ASCII)
        json_bytes = json_string.encode('utf-8')

        # Write compressed bytes to file
        with gzip.open(file_path, 'wb') as f_gz:
            f_gz.write(json_bytes)

        print(f"State successfully saved and compressed to {file_path}")
    except Exception as e:
        print(f"Error saving compressed state to {file_path}: {e}")

def loadStateLBA():
    """
    Loads the state from the LargeBlockAnalysisCompleted file.

    Returns:
        A dictionary containing the loaded state.
    """
    file_path = LargeBlockAnalysisCompletedFile
    with gzip.open(file_path, 'rb') as f_gz:
        json_bytes = f_gz.read()

    # Decode bytes back to JSON string
    json_string = json_bytes.decode('utf-8')
    # Deserialize JSON string
    loaded_state = json.loads(json_string)
    large_blocks = loaded_state.get("large_blocks")
    map_results = loaded_state.get("map_results")
    final_entities = loaded_state.get("final_entities")
    raw_text = loaded_state.get("raw_text")
    return large_blocks, map_results, final_entities, raw_text

def loadStateIA():
    #   Loads the state from the IterativeAnalysisCompleted file.
    file_path = IterativeAnalysisCompletedFile
    with gzip.open(file_path, 'rb') as f_gz:
        json_bytes = f_gz.read()
    loaded_state = json.loads(json_bytes.decode('utf-8'))
    doc_analysis_result = loaded_state.get("doc_analysis_result")
    large_blocks = loaded_state.get("large_blocks")
    map_results = loaded_state.get("map_results") 
    raw_text = loaded_state.get("raw_text")
    return doc_analysis_result, large_blocks, map_results, raw_text

