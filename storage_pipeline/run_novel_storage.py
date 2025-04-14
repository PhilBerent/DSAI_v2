#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main script to run the novel storage pipeline."""

import logging
import time
import os
import uuid
import sys
from typing import List, Dict, Any

# Adjust path to import from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import required global modules first
try:
    from globals import *
    from UtilityFunctions import *
    from DSAIParams import * # Imports RunCodeFrom, StateStorageList
    # Import enums for state management
    from enums_and_constants import RunFromType, StateStoragePoints
except ImportError as e:
    print(f"Error importing core modules (globals, UtilityFunctions, DSAIParams, enums): {e}")
    raise

# Pipeline components (using absolute imports from DSAI_v2_Scripts level)
try:
    from storage_pipeline.config_loader import DocToAddPath, Chunk_size, Chunk_overlap
    from storage_pipeline.db_connections import get_pinecone_index, get_neo4j_driver_local, test_connections
    from storage_pipeline.ingestion import ingest_document
    # Import the refactored analysis functions
    from storage_pipeline.analysis import perform_map_block_analysis, perform_reduce_document_analysis, analyze_chunk_details
    from storage_pipeline.chunking import coarse_chunk_by_structure, adaptive_chunking
    from storage_pipeline.embedding import generate_embeddings
    from storage_pipeline.graph_builder import build_graph_data
    from storage_pipeline.storage import store_embeddings_pinecone, store_graph_data_neo4j, store_chunk_metadata_docstore
    # Import state storage functions
    from state_storage import save_state, load_state
except ImportError as e:
    print(f"Error during absolute import: {e}")
    print("Ensure you are running this script from a context where DSAI_v2_Scripts is accessible,")
    print("or that DSAI_v2_Scripts is in your PYTHONPATH.")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(document_path: str):
    """Executes the full storage pipeline for a given document, with state management."""
    start_time = time.time()
    logging.info(f"--- Starting Storage Pipeline for: {document_path} (Run Mode: {RunCodeFrom}) ---")

    # --- 0. Setup & Connections ---
    # Connections needed regardless of start point for later steps
    try:
        logging.info("Setting up database connections...")
        pinecone_index = get_pinecone_index()
        neo4j_driver = get_neo4j_driver_local()
        document_id = os.path.splitext(os.path.basename(document_path))[0]
        logging.info(f"Using Document ID: {document_id}")
    except Exception as e:
        logging.error(f"Pipeline setup failed: {e}")
        return

    # Initialize state variables
    raw_text: str = ""
    large_blocks: List[Dict[str, Any]] = []
    map_results: List[Dict[str, Any]] = []
    final_entities: Dict[str, List[str]] = {}
    doc_analysis_result: Dict[str, Any] = {}
    final_chunks: List[Dict[str, Any]] = []
    chunks_with_analysis: List[Dict[str, Any]] = []
    embeddings_dict: Dict[str, List[float]] = {}
    graph_nodes: List[Dict] = []
    graph_edges: List[Dict] = []

    try:
        # --- Load State or Run from Start --- #
        if RunCodeFrom == RunFromType.Start:
            logging.info("Running pipeline from the beginning.")
            # Execute all steps from ingestion onwards

            # --- 1. Ingestion ---
            logging.info("Step 1: Ingesting document...")
            raw_text = ingest_document(document_path)
            logging.info(f"Ingestion complete. Text length: {len(raw_text)} chars.")

            # --- 2. Initial Structural Scan & Coarse Chunking ---
            logging.info("Step 2: Performing initial structural scan and coarse chunking...")
            large_blocks = coarse_chunk_by_structure(raw_text)
            if not large_blocks:
                raise ValueError("Coarse chunking resulted in zero blocks. Aborting.")
            logging.info(f"Coarse chunking complete. Generated {len(large_blocks)} large blocks.")

            # --- 3.1 Map Phase --- #
            logging.info("Step 3.1: Performing Map phase block analysis...")
            map_results, final_entities = perform_map_block_analysis(large_blocks)
            if not map_results:
                 raise ValueError("Map phase analysis failed for all blocks. Aborting.")
            logging.info("Map phase analysis complete.")

            # --- Save State: LargeBlockAnalysisCompleted ---
            if StateStoragePoints.LargeBlockAnalysisCompleted in StateStorageList:
                logging.info("Saving state after Map phase (LargeBlockAnalysisCompleted)...")
                state_to_save = {
                    "large_blocks": large_blocks,
                    "map_results": map_results,
                    "final_entities": final_entities,
                    "raw_text": raw_text # Include raw_text if needed for subsequent steps like fine chunking
                }
                save_state(state_to_save, StateStoragePoints.LargeBlockAnalysisCompleted)

            # --- 3.2 Reduce Phase --- #
            logging.info("Step 3.2: Performing Reduce phase document synthesis...")
            doc_analysis_result = perform_reduce_document_analysis(map_results, final_entities)
            if "error" in doc_analysis_result:
                raise ValueError(f"Reduce phase document analysis failed: {doc_analysis_result['error']}")
            logging.info("Reduce phase analysis complete.")

            # --- Save State: IterativeAnalysisCompleted ---
            if StateStoragePoints.IterativeAnalysisCompleted in StateStorageList:
                logging.info("Saving state after Reduce phase (IterativeAnalysisCompleted)...")
                state_to_save = {
                    "doc_analysis_result": doc_analysis_result,
                    "large_blocks": large_blocks, # Needed if fine chunking depends on large blocks
                    "map_results": map_results,   # Might be useful context
                    "final_entities": final_entities, # Might be useful context
                    "raw_text": raw_text # Include raw_text for fine chunking
                }
                save_state(state_to_save, StateStoragePoints.IterativeAnalysisCompleted)

        elif RunCodeFrom == RunFromType.LargeBlockAnalysisCompleted:
            logging.info("Attempting to load state from LargeBlockAnalysisCompleted...")
            try:
                loaded_state = load_state(RunFromType.LargeBlockAnalysisCompleted)
                large_blocks = loaded_state["large_blocks"]
                map_results = loaded_state["map_results"]
                final_entities = loaded_state["final_entities"]
                raw_text = loaded_state["raw_text"]
                logging.info("State loaded. Proceeding from Reduce phase.")

                # --- 3.2 Reduce Phase --- #
                logging.info("Step 3.2: Performing Reduce phase document synthesis...")
                doc_analysis_result = perform_reduce_document_analysis(map_results, final_entities)
                if "error" in doc_analysis_result:
                    raise ValueError(f"Reduce phase document analysis failed after loading state: {doc_analysis_result['error']}")
                logging.info("Reduce phase analysis complete.")

                # --- Save State: IterativeAnalysisCompleted ---
                if StateStoragePoints.IterativeAnalysisCompleted in StateStorageList:
                    logging.info("Saving state after Reduce phase (IterativeAnalysisCompleted)...")
                    state_to_save = {
                        "doc_analysis_result": doc_analysis_result,
                        "large_blocks": large_blocks,
                        "map_results": map_results,
                        "final_entities": final_entities,
                        "raw_text": raw_text
                    }
                    save_state(state_to_save, StateStoragePoints.IterativeAnalysisCompleted)

            except (FileNotFoundError, KeyError, Exception) as e:
                logging.error(f"Failed to load or use state from {RunCodeFrom}: {e}. Aborting.")
                raise

        elif RunCodeFrom == RunFromType.IterativeAnalysisCompleted:
            logging.info("Attempting to load state from IterativeAnalysisCompleted...")
            try:
                loaded_state = load_state(RunFromType.IterativeAnalysisCompleted)
                doc_analysis_result = loaded_state["doc_analysis_result"]
                large_blocks = loaded_state["large_blocks"] # Load if needed by subsequent steps
                map_results = loaded_state["map_results"]   # Load if needed
                final_entities = loaded_state["final_entities"] # Load if needed
                raw_text = loaded_state["raw_text"]     # Load for fine chunking
                logging.info("State loaded. Proceeding from Fine-grained Chunking.")
            except (FileNotFoundError, KeyError, Exception) as e:
                logging.error(f"Failed to load or use state from {RunCodeFrom}: {e}. Aborting.")
                raise

        # --- Remaining Pipeline Steps (executed if not aborted due to load failure) ---

        # --- 4. Adaptive Fine-Grained Chunking ---
        # Check if raw_text is available (should be loaded or generated)
        if not raw_text:
             raise ValueError("Raw text is unavailable for fine-grained chunking. Check state loading or initial run.")
        logging.info("Step 4: Performing adaptive fine-grained chunking...")
        final_chunks = adaptive_chunking(
            raw_text,
            document_structure=doc_analysis_result, # Use loaded/generated analysis
            target_chunk_size=Chunk_size,
            chunk_overlap=Chunk_overlap
        )
        if not final_chunks:
             raise ValueError("Chunking resulted in zero chunks. Aborting.")
        logging.info(f"Chunking complete. Generated {len(final_chunks)} chunks.")
        # Add document_id to chunk metadata
        for chunk in final_chunks:
             chunk['metadata']['document_id'] = document_id

        # --- 5. Chunk-Level Analysis ---
        logging.info("Step 5: Performing detailed chunk analysis...")
        processed_chunks = 0
        for i, chunk in enumerate(final_chunks):
            logging.info(f"Analyzing chunk {i+1}/{len(final_chunks)} (ID: {chunk['chunk_id']})...")
            try:
                chunk_analysis_result = analyze_chunk_details(
                    chunk_text=chunk['text'],
                    chunk_id=chunk['chunk_id'],
                    doc_context=doc_analysis_result # Provide doc context
                )
                chunk['analysis'] = chunk_analysis_result
                chunks_with_analysis.append(chunk)
                processed_chunks += 1
            except Exception as e:
                logging.error(f"Failed to analyze chunk {chunk['chunk_id']}: {e}. Skipping chunk.")
        logging.info(f"Chunk analysis complete. Successfully analyzed {processed_chunks}/{len(final_chunks)} chunks.")
        if not chunks_with_analysis:
             raise ValueError("Chunk analysis failed for all chunks. Aborting.")

        # --- 6. Embedding Generation ---
        logging.info("Step 6: Generating embeddings...")
        embeddings_dict = generate_embeddings(chunks_with_analysis)
        if not embeddings_dict:
            raise ValueError("Embedding generation failed for all processed chunks. Aborting.")
        logging.info(f"Embedding generation complete. Generated {len(embeddings_dict)} embeddings.")

        # --- 7. Graph Data Construction ---
        logging.info("Step 7: Constructing graph data...")
        graph_nodes, graph_edges = build_graph_data(document_id, doc_analysis_result, chunks_with_analysis)
        logging.info(f"Graph construction complete. Nodes: {len(graph_nodes)}, Edges: {len(graph_edges)}.")

        # --- 8. Data Storage ---
        logging.info("Step 8: Storing data...")
        store_embeddings_pinecone(pinecone_index, embeddings_dict, chunks_with_analysis)
        store_graph_data_neo4j(neo4j_driver, graph_nodes, graph_edges)
        store_chunk_metadata_docstore(chunks_with_analysis)
        logging.info("Data storage complete.")

    except Exception as e:
        logging.exception(f"Pipeline execution failed: {e}")
    finally:
        # Clean up resources
        if 'neo4j_driver' in locals() and neo4j_driver:
            neo4j_driver.close()
            logging.info("Neo4j driver closed.")

    end_time = time.time()
    logging.info(f"--- Storage Pipeline Finished for: {document_path} (Run Mode: {RunCodeFrom}) --- ")
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_pipeline(DocToAddPath) 