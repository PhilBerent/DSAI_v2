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

# Import required global modules first
try:
    from globals import *
    from UtilityFunctions import *
    from DSAIParams import * # Imports RunCodeFrom, StateStorageList, DocToAddPath etc.
    # Import enums for state management and the list of stages
    from enums_and_constants import CodeStages, StateStoragePoints, Code_Stages_List
    from primary_analysis_stages import *
except ImportError as e:
    print(f"Error importing core modules (globals, UtilityFunctions, DSAIParams, enums): {e}")
    raise

# Pipeline components
try:
    # config_loader is likely not needed directly here anymore if DocToAddPath is from DSAIParams
    # from config_loader import DocToAddPath, Chunk_Size, Chunk_overlap # Chunk parameters are now in DSAIParams
    from storage_pipeline.db_connections import get_pinecone_index, get_neo4j_driver_local # Removed test_connections, not used
    # Import the new stage functions
    # State storage is used within the stage functions now
    # from state_storage import save_state, load_state # No longer needed here
except ImportError as e:
    print(f"Error during absolute import: {e}")
    print("Ensure you are running this script from a context where DSAI_v2_Scripts is accessible,")
    print("or that DSAI_v2_Scripts is in your PYTHONPATH.")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(document_path: str):
    """Executes the storage pipeline for a given document using a stage-based approach."""
    start_time = time.time()
    logging.info(f"--- Starting Storage Pipeline for: {document_path} (Run Mode: {RunCodeFrom.value}) ---")

    # --- 0. Setup & Connections ---
    pinecone_index = None # Initialize to ensure availability in finally block
    neo4j_driver = None   # Initialize to ensure availability in finally block
    try:
        logging.info("Setting up database connections...")
        # Connections are needed for the final stage
        # pinecone_index = get_pinecone_index()
        # neo4j_driver = get_neo4j_driver_local()
        file_id = os.path.splitext(os.path.basename(document_path))[0] # Use filename without ext as ID
        logging.info(f"Using Document ID: {file_id}")
    except Exception as e:
        logging.error(f"Pipeline setup failed: {e}")
        # Clean up any potentially partially initialized connections
        if neo4j_driver:
            neo4j_driver.close()
        # Pinecone client doesn't usually require explicit close in this manner
        return # Exit if setup fails

    # Initialize variables to hold state between stages
    raw_text = large_blocks = map_results = final_entities = chunks_with_analysis = \
        embeddings_dict = doc_analysis_result = graph_nodes = graph_edges = None

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

                logging.info(f"--- Executing Stage: {stage} (Load State: {load_state_flag}) ---")

                if stage == CodeStages.Start.value:
                    # Stage 1: Initial Processing
                    (raw_text, large_blocks, map_results, 
                    final_entities) = large_block_analysis(document_path, file_id)
                    aa=2
                elif stage == CodeStages.LargeBlockAnalysisCompleted.value:
                    if load_state_flag:
                        large_blocks, map_results, final_entities, raw_text = loadStateLBA()
                    # Stage 2: Iterative Analysis (Reduce Phase)
                    (raw_text, large_blocks, map_results, final_entities, doc_analysis_result) = perform_reduce_analysis(file_id, raw_text, large_blocks, map_results, final_entities)
                elif stage == CodeStages.ReduceAnalysisCompleted.value:
                    if load_state_flag:
                        doc_analysis_result, large_blocks, map_results, raw_text = loadStateIA()
                    #  Stage 3: Iterative Analysis (Map Phase)
                    (file_id, map_results, doc_analysis_result, chunks_with_analysis) = perform_detailed_chunk_analysis(file_id, raw_text, large_blocks, map_results, doc_analysis_result)
                    # This is the last stage defined in the list, so we break the loop after execution
                    # If more stages were added, this break might need reconsideration
                elif stage == CodeStages.DetailedBlockAnalysisCompleted.value:
                    if load_state_flag:
                        file_id, chunks_with_analysis, doc_analysis_result, map_results = loadStateDBA()
                    # --- 4. Embedding Generation --- #                        
                    (embeddings_dict, file_id, chunks_with_analysis, doc_analysis_result, 
                        map_results) = get_embeddings(file_id, chunks_with_analysis, 
                        doc_analysis_result, map_results)
                elif stage == CodeStages.EmbeddingsCompleted.value:
                    if load_state_flag:
                        file_id, embeddings_dict, chunks_with_analysis, doc_analysis_result, map_results = loadStateEA()
                    # --- 5. Graph Data Construction --- #                    
                    (graph_nodes, graph_edges, file_id, embeddings_dict, chunks_with_analysis, 
                    doc_analysis_result, map_results) = perform_graph_analyisis(file_id, doc_analysis_result, chunks_with_analysis, embeddings_dict, map_results)
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
                logging.error(f"Error closing Neo4j driver: {e}")
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