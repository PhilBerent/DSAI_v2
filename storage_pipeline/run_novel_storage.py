#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main script to run the novel storage pipeline."""

import logging
import time
import os
import uuid
import sys
from typing import List, Dict, Any, Optional

# Adjust path to import from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from globals import *
from UtilityFunctions import *
from DSAIParams import * # Imports RunCodeFrom, StateStorageList, DocToAddPath etc.
# Import enums for state management and the list of stages
from enums_constants_and_classes import CodeStages, StateStoragePoints, Code_Stages_List
from primary_analysis_stages import *
from alias_resolution import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(document_path: str):
    """Executes the storage pipeline for a given document using a stage-based approach."""
    start_time = time.time()
    file_id = os.path.splitext(os.path.basename(document_path))[0] # Use filename without ext as ID    
    logging.info(f"--- Starting Storage Pipeline for: {file_id} (Run Mode: {RunCodeFrom.value}) ---")

    # Initialize variables to hold state between stages
    raw_text = large_blocks = block_info_list = full_entities_list = chunks_with_analysis = \
        embeddings_dict = doc_analysis = graph_nodes = graph_edges = None

    try:
        # Determine the starting index in the stages list
        try:
            start_index = Code_Stages_List.index(RunCodeFrom.value)
            logging.info(f"Pipeline will run stages starting from index {start_index}: {Code_Stages_List[start_index:]}")

            # --- Execute Pipeline Stages ---
            for i, stage in enumerate(Code_Stages_List[start_index:]):
                # Determine if state should be loaded for this stage
                # Load state only if it's the *first* stage being executed in this run
                load_state_flag = (stage != 'Start' and stage == RunCodeFrom.value) # Don't load state if starting from the beginning
                logMessage = ("Loading and executing " if load_state_flag else "Executing ")+f"""stage "{stage}"\n"""
                logging.info(logMessage)

                if stage == CodeStages.Start.value:
                    # Stage 1: Initial Processing
                    (raw_text, large_blocks, block_info_list) = large_block_analysis(document_path, file_id)
                    aa=2
                elif stage == CodeStages.LargeBlockAnalysisCompleted.value:
                    if load_state_flag:
                        large_blocks, block_info_list, raw_text = loadStateLBA()
                    (prelim_entity_data, primary_names_entity_dict, entityData_alt_names_dict, char_match_data) = \
                         consolidate_entity_information(block_info_list)

                    (prelim_primary_names, primary_names_dict, is_an_alt_name_of_dict, has_alt_names_dict) = \
                        get_primary_entity_names(prelim_entity_data, entityData_alt_names_dict)
                    
                    (comparison_pairs, comp_pair_names) = \
                        get_alias_comparison_pairs(prelim_primary_names,  primary_names_dict, 
                            is_an_alt_name_of_dict, has_alt_names_dict, char_match_data)
                    d=4
                    # Stage 2: Iterative Analysis (Reduce Phase)
                    (raw_text, large_blocks, block_info_list, full_entities_list, doc_analysis) = perform_reduce_analysis(file_id, raw_text, large_blocks, block_info_list, full_entities_list)
                    a=4
                elif stage == CodeStages.ReduceAnalysisCompleted.value:
                    if load_state_flag:
                        doc_analysis, large_blocks, block_info_list, raw_text, full_entities_list = loadStateIA()
                    #  Stage 3: Iterative Analysis (Map Phase)
                    (file_id, block_info_list, doc_analysis, chunks_with_analysis, full_entities_list) = perform_detailed_chunk_analysis(file_id, raw_text, large_blocks, block_info_list, doc_analysis, full_entities_list)
                    # This is the last stage defined in the list, so we break the loop after execution
                    # If more stages were added, this break might need reconsideration
                elif stage == CodeStages.DetailedBlockAnalysisCompleted.value:
                    if load_state_flag:
                        file_id, chunks_with_analysis, doc_analysis, block_info_list, full_entities_list = loadStateDBA()
                    # --- 4. Embedding Generation --- #                        
                    (embeddings_dict, file_id, chunks_with_analysis, doc_analysis, 
                        block_info_list, full_entities_list) = get_embeddings(file_id, chunks_with_analysis, 
                        doc_analysis, block_info_list, full_entities_list)
                elif stage == CodeStages.EmbeddingsCompleted.value:
                    if load_state_flag:
                        (file_id, embeddings_dict, chunks_with_analysis, doc_analysis, 
                            block_info_list, full_entities_list) = loadStateEA()
                    # --- 5. Graph Data Construction --- #                    
                    (graph_nodes, graph_edges, file_id, embeddings_dict, chunks_with_analysis, 
                    doc_analysis, block_info_list) = perform_graph_analyisis(file_id, doc_analysis, chunks_with_analysis, embeddings_dict, block_info_list)
                else:
                    logging.warning(f"Encountered an unrecognized stage: {stage}. Skipping.")
            
        except ValueError:
            logging.error(f"error")

        # Optional: Add a small delay between stages if needed for resource reasons
        # time.sleep(1)

    except FileNotFoundError as e:
        # Specific handling for state file not found when expected
        logging.error(f"Pipeline aborted: Required state file not found. {e}")
    except ValueError as e:
        # Handling for validation errors (e.g., invalid start stage, zero chunks)
        logging.error(f"Pipeline aborted due to invalid data or configuration: {e}")
    except Exception as e:
        logging.exception(f"Pipeline execution failed due to an unexpected error: {e}") # Logs traceback
    finally:
        # Clean up resources
        if neo4j_driver:
            try:
                neo4j_driver.close()
                logging.info("Neo4j driver closed.")
            except Exception as e:
                logging.error(f"Error closing Neo4j driver: {e}", exc_info=True)
        # Pinecone client does not require explicit close in recent versions

    end_time = time.time()
    logging.info(f"--- Storage Pipeline Finished for: {document_path} (Ran from: {RunCodeFrom.value}) --- ")
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Ensure document path is provided, e.g., from DSAIParams or command line
    if not DocToAddPath or not os.path.exists(DocToAddPath):
        print(f"Error: Document path '{DocToAddPath}' not found or not specified in DSAIParams.py.")
    else:
        run_pipeline(DocToAddPath) 