#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions representing the primary analysis stages of the pipeline."""

import logging
import time
import os
import uuid
import sys
from typing import List, Dict, Any, Tuple, Optional

# Adjust path to import from parent directory (DSAI_v2_Scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import required global modules first
try:
    from globals import *
    from UtilityFunctions import *
    # Import specific params needed, maybe not all with *
    from DSAIParams import Chunk_Size, Chunk_overlap, StateStorageList
    # Import enums needed for stage functions and state storage
    from enums_and_constants import CodeStages, StateStoragePoints
except ImportError as e:
    print(f"Error importing core modules in primary_analysis_stages: {e}")
    raise

# Import pipeline components
try:
    from storage_pipeline.ingestion import ingest_document
    from storage_pipeline.analysis_functions import perform_map_block_analysis, perform_reduce_document_analysis, analyze_chunk_details
    from storage_pipeline.chunking import coarse_chunk_by_structure, adaptive_chunking
    from storage_pipeline.embedding import generate_embeddings
    from storage_pipeline.graph_builder import build_graph_data
    from storage_pipeline.storage import store_embeddings_pinecone, store_graph_data_neo4j, store_chunk_metadata_docstore
    # Import state storage functions using the original module structure
    from state_storage import save_state as original_save_state, load_state as original_load_state
except ImportError as e:
    print(f"Error during absolute import in primary_analysis_stages: {e}")
    print("Ensure necessary modules exist and DSAI_v2_Scripts is accessible.")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Stage 1: Start -> LargeBlockAnalysisCompleted ---
def perform_initial_processing(document_path: str, file_id: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]]]:
    """Handles ingestion, coarse chunking, and map-phase analysis.
       Saves state using original_save_state if configured.
    """
    # [Code for Ingestion, Coarse Chunking, Map Phase remains unchanged]
    logging.info(f"Stage 1: Starting Initial Processing for {file_id}...")

    # --- 1. Ingestion ---
    logging.info("Step 1.1: Ingesting document...")
    try:
        raw_text = ingest_document(document_path)
        if not raw_text:
            raise ValueError("Ingestion resulted in empty text.")
        logging.info(f"Ingestion complete. Text length: {len(raw_text)} chars.")
    except Exception as e:
        logging.error(f"Ingestion failed for {document_path}: {e}", exc_info=True)
        raise ValueError(f"Ingestion failed: {e}") from e

    # --- 2. Initial Structural Scan & Coarse Chunking ---
    logging.info("Step 1.2: Performing initial structural scan and coarse chunking...")
    try:
        large_blocks = coarse_chunk_by_structure(raw_text)
        if not large_blocks:
            raise ValueError("Coarse chunking resulted in zero blocks.")
        logging.info(f"Coarse chunking complete. Generated {len(large_blocks)} large blocks.")
    except Exception as e:
        logging.error(f"Coarse chunking failed: {e}", exc_info=True)
        raise ValueError(f"Coarse chunking failed: {e}") from e

    # --- 3.1 Map Phase --- #
    logging.info("Step 1.3: Performing Map phase block analysis...")
    try:
        map_results, final_entities = perform_map_block_analysis(large_blocks)
        if not map_results:
             logging.warning("Map phase analysis returned no results. Check individual block analysis logs.")
        logging.info(f"Map phase analysis complete. Got {len(map_results)} results.")
    except Exception as e:
        logging.error(f"Map phase analysis failed: {e}", exc_info=True)
        raise ValueError(f"Map phase analysis failed: {e}") from e

    # --- Save State: LargeBlockAnalysisCompleted ---
    # Use the original save_state(data, storage_point_enum) signature
    if StateStoragePoints.LargeBlockAnalysisCompleted in StateStorageList:
        logging.info(f"Saving state for {StateStoragePoints.LargeBlockAnalysisCompleted.name} (using original state_storage)... ")
        state_to_save = {
            "large_blocks": large_blocks,
            "map_results": map_results,
            "final_entities": final_entities,
            "raw_text": raw_text
        }
        try:
            original_save_state(state_to_save, StateStoragePoints.LargeBlockAnalysisCompleted)
            # No file_id or stage value needed here as per original signature
        except Exception as e:
            # Log error as per original state_storage.py (it uses print)
            logging.error(f"Original state_storage.py save failed for {StateStoragePoints.LargeBlockAnalysisCompleted.name}: {e}", exc_info=True)
            # Decide if this should raise an error and stop the pipeline
            # raise

    logging.info(f"Stage 1: Initial Processing complete for {file_id}.")
    return raw_text, large_blocks, map_results, final_entities


# --- Stage 2: LargeBlockAnalysisCompleted -> IterativeAnalysisCompleted ---
def perform_iterative_analysis(
    load_state_flag: bool,
    file_id: str, # Keep file_id for logging/context if needed elsewhere
    # Data passed if not loading state
    raw_text_in: Optional[str] = None,
    large_blocks_in: Optional[List[Dict[str, Any]]] = None,
    map_results_in: Optional[List[Dict[str, Any]]] = None,
    final_entities_in: Optional[Dict[str, List[str]]] = None
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]], Optional[Dict[str, List[str]]], Dict[str, Any]]:
    """Handles the reduce phase of the analysis.
       Loads state using original_load_state if load_state_flag is True.
       Saves state using original_save_state if configured.
    """
    logging.info(f"Stage 2: Starting Iterative Analysis for {file_id} (Load State: {load_state_flag})...")
    raw_text: Optional[str] = None
    large_blocks: Optional[List[Dict[str, Any]]] = None
    map_results: Optional[List[Dict[str, Any]]] = None
    final_entities: Optional[Dict[str, List[str]]] = None
    doc_analysis_result: Dict[str, Any] = {}

    if load_state_flag:
        # Use the original load_state(run_from: CodeStages) signature
        # We want to load the state saved *before* this stage, which corresponds to LargeBlockAnalysisCompleted
        stage_to_load_from = CodeStages.LargeBlockAnalysisCompleted
        logging.info(f"Attempting to load state using original state_storage.load_state(run_from={stage_to_load_from.name})...")
        try:
            loaded_state = original_load_state(stage_to_load_from)

            # Validate and assign loaded data (assuming keys match what was saved)
            large_blocks = loaded_state.get("large_blocks")
            map_results = loaded_state.get("map_results")
            final_entities = loaded_state.get("final_entities")
            raw_text = loaded_state.get("raw_text")

            # Check for missing critical data needed for this stage
            if map_results is None or final_entities is None or raw_text is None:
                 missing_keys = [k for k,v in {"map_results": map_results, "final_entities": final_entities, "raw_text": raw_text}.items() if v is None]
                 logging.error(f"Loaded state (using original state_storage) is missing critical keys: {missing_keys} when loading for stage {stage_to_load_from.name}")
                 raise KeyError(f"Loaded state is missing critical keys: {missing_keys}")

            logging.info(f"State loaded successfully for {file_id} using original state_storage.")
        except (FileNotFoundError, KeyError, ValueError, Exception) as e:
            # Log error (original load_state uses print, we add logging)
            logging.error(f"Original state_storage.py load failed when attempting to load for stage {stage_to_load_from.name}: {e}. Aborting.", exc_info=True)
            raise # Re-raise to stop the pipeline
    else:
        # Use data passed from previous stage, validate it
        logging.info(f"Using data passed from previous stage for {file_id}.")
        if map_results_in is None or final_entities_in is None or raw_text_in is None:
             missing_args = [k for k,v in {"map_results_in": map_results_in, "final_entities_in": final_entities_in, "raw_text_in": raw_text_in}.items() if v is None]
             logging.error(f"Missing critical data passed from previous stage to Stage 2: {missing_args}")
             raise ValueError(f"Missing critical data passed from previous stage: {missing_args}")
        raw_text = raw_text_in
        large_blocks = large_blocks_in
        map_results = map_results_in
        final_entities = final_entities_in

    # --- 3.2 Reduce Phase --- #
    # [Code for Reduce Phase analysis remains unchanged]
    logging.info("Step 2.1: Performing Reduce phase document synthesis...")
    try:
        if map_results is None or final_entities is None:
             raise ValueError("Cannot perform reduce phase: map_results or final_entities are missing.")

        doc_analysis_result = perform_reduce_document_analysis(map_results, final_entities)
        if not doc_analysis_result or "error" in doc_analysis_result:
            error_msg = doc_analysis_result.get('error', 'Unknown error') if doc_analysis_result else 'No result returned'
            error_detail = doc_analysis_result.get('error_details', 'No details provided') if doc_analysis_result else 'N/A'
            logging.error(f"Reduce phase document analysis failed or returned error: {error_msg} - Details: {error_detail}")
            raise ValueError(f"Reduce phase document analysis failed: {error_msg}")
        logging.info("Reduce phase analysis complete.")
    except Exception as e:
        logging.error(f"Exception during Reduce phase analysis: {e}", exc_info=True)
        raise ValueError(f"Reduce phase analysis failed with exception: {e}") from e


    # --- Save State: IterativeAnalysisCompleted ---
    # Use the original save_state(data, storage_point_enum) signature
    if StateStoragePoints.IterativeAnalysisCompleted in StateStorageList:
        logging.info(f"Saving state for {StateStoragePoints.IterativeAnalysisCompleted.name} (using original state_storage)...")
        state_to_save = {
            "doc_analysis_result": doc_analysis_result,
            "large_blocks": large_blocks,
            "map_results": map_results,
            "final_entities": final_entities,
            "raw_text": raw_text
        }
        try:
            original_save_state(state_to_save, StateStoragePoints.IterativeAnalysisCompleted)
        except Exception as e:
            logging.error(f"Original state_storage.py save failed for {StateStoragePoints.IterativeAnalysisCompleted.name}: {e}", exc_info=True)
            # raise

    logging.info(f"Stage 2: Iterative Analysis complete for {file_id}.")
    return raw_text, large_blocks, map_results, final_entities, doc_analysis_result


# --- Stage 3: IterativeAnalysisCompleted -> End ---
def perform_downstream_processing(
    load_state_flag: bool,
    file_id: str, # Keep file_id for logging/context
    pinecone_index: Any,
    neo4j_driver: Any,
    # Data passed if not loading state
    raw_text_in: Optional[str] = None,
    large_blocks_in: Optional[List[Dict[str, Any]]] = None,
    map_results_in: Optional[List[Dict[str, Any]]] = None,
    doc_analysis_result_in: Optional[Dict[str, Any]] = None
) -> None:
    """Handles fine-grained chunking, chunk analysis, embedding, graph building, and storage.
       Loads state using original_load_state if load_state_flag is True.
       Does not save state.
    """
    logging.info(f"Stage 3: Starting Downstream Processing for {file_id} (Load State: {load_state_flag})...")
    raw_text: Optional[str] = None
    large_blocks: Optional[List[Dict[str, Any]]] = None
    map_results: Optional[List[Dict[str, Any]]] = None
    doc_analysis_result: Optional[Dict[str, Any]] = None

    if load_state_flag:
        # Use the original load_state(run_from: CodeStages) signature
        # We want to load the state saved *before* this stage, which corresponds to IterativeAnalysisCompleted
        stage_to_load_from = CodeStages.IterativeAnalysisCompleted
        logging.info(f"Attempting to load state using original state_storage.load_state(run_from={stage_to_load_from.name})...")
        try:
            loaded_state = original_load_state(stage_to_load_from)

            # Validate and assign loaded data
            doc_analysis_result = loaded_state.get("doc_analysis_result")
            large_blocks = loaded_state.get("large_blocks")
            map_results = loaded_state.get("map_results")
            raw_text = loaded_state.get("raw_text")

            # Check for missing critical data needed for this stage
            if raw_text is None or large_blocks is None or map_results is None or doc_analysis_result is None:
                 missing_keys = [k for k,v in {
                    "raw_text": raw_text, "large_blocks": large_blocks,
                    "map_results": map_results, "doc_analysis_result": doc_analysis_result
                 }.items() if v is None]
                 logging.error(f"Loaded state (using original state_storage) is missing critical keys: {missing_keys} when loading for stage {stage_to_load_from.name}")
                 raise KeyError(f"Loaded state is missing critical keys for Stage 3: {missing_keys}")

            logging.info(f"State loaded successfully for {file_id} using original state_storage.")
        except (FileNotFoundError, KeyError, ValueError, Exception) as e:
            logging.error(f"Original state_storage.py load failed when attempting to load for stage {stage_to_load_from.name}: {e}. Aborting.", exc_info=True)
            raise # Re-raise to stop the pipeline
    else:
        # Use data passed from previous stage, validate it
        logging.info(f"Using data passed from previous stage for {file_id}.")
        if raw_text_in is None or large_blocks_in is None or map_results_in is None or doc_analysis_result_in is None:
             missing_args = [k for k,v in {
                "raw_text_in": raw_text_in, "large_blocks_in": large_blocks_in,
                "map_results_in": map_results_in, "doc_analysis_result_in": doc_analysis_result_in
             }.items() if v is None]
             logging.error(f"Missing critical data passed from previous stage to Stage 3: {missing_args}")
             raise ValueError(f"Missing critical data passed to Stage 3: {missing_args}")

        raw_text = raw_text_in
        large_blocks = large_blocks_in
        map_results = map_results_in
        doc_analysis_result = doc_analysis_result_in

    # Ensure critical data is present before proceeding
    if raw_text is None or large_blocks is None or map_results is None or doc_analysis_result is None:
         raise ValueError(f"Critical data unavailable for Stage 3 processing (file_id: {file_id}). Check state or pipeline flow.")

    # --- 4. Adaptive Fine-Grained Chunking --- #
    # [Code for Adaptive Chunking remains unchanged] ...
    logging.info("Step 3.1: Performing adaptive fine-grained chunking...")
    final_chunks: List[Dict[str, Any]] = [] # Initialize here
    try:
        # Pass arguments validated above
        final_chunks = adaptive_chunking(
            structural_units=large_blocks,
            map_results=map_results,
            target_chunk_size=Chunk_Size, # From DSAIParams
            chunk_overlap=Chunk_overlap   # From DSAIParams
        )
        if not final_chunks:
             logging.warning("Adaptive chunking resulted in zero final chunks.")
        else:
             logging.info(f"Adaptive chunking complete. Generated {len(final_chunks)} chunks.")
             # Add document_id to chunk metadata
             for chunk in final_chunks:
                 if 'metadata' not in chunk:
                      chunk['metadata'] = {}
                 chunk['metadata']['document_id'] = file_id
    except Exception as e:
        logging.error(f"Adaptive fine-grained chunking failed: {e}", exc_info=True)
        raise ValueError(f"Adaptive chunking failed: {e}") from e

    if not final_chunks:
        logging.warning("Skipping subsequent steps as no final chunks were generated.")
        logging.info(f"Stage 3: Downstream Processing complete (skipped storage) for {file_id}.")
        return

    # --- 5. Chunk-Level Analysis --- #
    # [Code for Chunk Analysis remains unchanged] ...
    logging.info("Step 3.2: Performing detailed chunk analysis...")
    chunks_with_analysis: List[Dict[str, Any]] = []
    processed_chunks = 0
    failed_chunks = 0
    try:
        # Doc analysis result checked earlier
        for i, chunk in enumerate(final_chunks):
            chunk_id = chunk.get('chunk_id', f'generated_{i}') # Ensure some ID exists
            logging.debug(f"Analyzing chunk {i+1}/{len(final_chunks)} (ID: {chunk_id})...")
            try:
                chunk_analysis_result = analyze_chunk_details(
                    chunk_text=chunk['text'],
                    chunk_id=chunk_id,
                    doc_context=doc_analysis_result # Provide doc context
                )
                chunk['analysis'] = chunk_analysis_result
                chunk['chunk_id'] = chunk_id # Ensure ID is stored back if generated
                chunks_with_analysis.append(chunk)
                processed_chunks += 1
            except Exception as e:
                failed_chunks += 1
                logging.error(f"Failed to analyze chunk {chunk_id}: {e}. Skipping chunk.", exc_info=False)

        logging.info(f"Chunk analysis complete. Successfully analyzed {processed_chunks}/{len(final_chunks)} chunks. Failed: {failed_chunks}")
        if not chunks_with_analysis:
             raise ValueError("Chunk analysis failed for all chunks. Aborting subsequent steps.")

    except Exception as e:
        logging.error(f"Error during chunk analysis process: {e}", exc_info=True)
        raise ValueError(f"Chunk analysis process failed: {e}") from e


    # --- 6. Embedding Generation --- #
    # [Code for Embedding Generation remains unchanged] ...
    logging.info("Step 3.3: Generating embeddings...")
    embeddings_dict: Dict[str, List[float]] = {}
    try:
        embeddings_dict = generate_embeddings(chunks_with_analysis) # Pass chunks that have analysis
        if not embeddings_dict:
            raise ValueError("Embedding generation failed for all processed chunks.")
        logging.info(f"Embedding generation complete. Generated {len(embeddings_dict)} embeddings.")
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}", exc_info=True)
        raise ValueError(f"Embedding generation failed: {e}") from e


    # --- 7. Graph Data Construction --- #
    # [Code for Graph Construction remains unchanged] ...
    logging.info("Step 3.4: Constructing graph data...")
    graph_nodes: List[Dict] = []
    graph_edges: List[Dict] = []
    try:
        graph_nodes, graph_edges = build_graph_data(file_id, doc_analysis_result, chunks_with_analysis)
        logging.info(f"Graph construction complete. Nodes: {len(graph_nodes)}, Edges: {len(graph_edges)}.")
    except Exception as e:
        logging.error(f"Graph data construction failed: {e}", exc_info=True)
        raise ValueError(f"Graph data construction failed: {e}") from e


    # --- 8. Data Storage --- #
    # [Code for Data Storage remains unchanged] ...
    logging.info("Step 3.5: Storing data...")
    try:
        if pinecone_index is None or neo4j_driver is None:
            raise ConnectionError("Database connections are not available for storage.")

        if not embeddings_dict:
            logging.warning("No embeddings generated, skipping Pinecone storage.")
        else:
            store_embeddings_pinecone(pinecone_index, embeddings_dict, chunks_with_analysis)

        if not graph_nodes and not graph_edges:
            logging.warning("No graph data generated, skipping Neo4j storage.")
        else:
            store_graph_data_neo4j(neo4j_driver, graph_nodes, graph_edges)

        if not chunks_with_analysis:
            logging.warning("No analyzed chunks available, skipping Docstore storage.")
        else:
            store_chunk_metadata_docstore(chunks_with_analysis)

        logging.info("Data storage calls complete.")
    except ConnectionError as e:
        logging.error(f"Database connection error during storage: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"Data storage failed: {e}", exc_info=True)
        raise ValueError(f"Data storage step failed: {e}") from e


    logging.info(f"Stage 3: Downstream Processing complete for {file_id}.")

# Note: Calls to state_storage functions now use the imported original functions:
# original_save_state(data: Dict, storage_point: StateStoragePoints)
# original_load_state(run_from: CodeStages) -> Dict
# These calls do not involve file_id or stage string values.