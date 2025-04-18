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
    from DSAIParams import * # Imports Chunk_Size, Chunk_overlap, StateStorageList etc
    from enums_and_constants import CodeStages, StateStoragePoints # Import enums
except ImportError as e:
    print(f"Error importing core modules in primary_analysis_stages: {e}")
    raise

# Import pipeline components (using absolute imports from DSAI_v2_Scripts level)
try:
    from storage_pipeline.ingestion import ingest_document
    from storage_pipeline.analysis_functions import perform_map_block_analysis, perform_reduce_document_analysis, analyze_chunk_details
    from storage_pipeline.chunking import coarse_chunk_by_structure, adaptive_chunking
    from storage_pipeline.embedding import generate_embeddings
    from storage_pipeline.graph_builder import build_graph_data
    from storage_pipeline.storage import store_embeddings_pinecone, store_graph_data_neo4j, store_chunk_metadata_docstore
    # Import state storage functions from the top-level directory
    import state_storage
except ImportError as e:
    print(f"Error during absolute import in primary_analysis_stages: {e}")
    print("Ensure necessary modules exist and DSAI_v2_Scripts is accessible.")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Stage 1: Start -> LargeBlockAnalysisCompleted ---
def perform_initial_processing(document_path: str, file_id: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]]]:
    """Handles ingestion, coarse chunking, and map-phase analysis.

    Args:
        document_path: Path to the input document.
        file_id: Unique identifier for the document.

    Returns:
        A tuple containing:
        - raw_text: The ingested text content.
        - large_blocks: List of coarse chunks.
        - map_results: List of analysis results from the map phase.
        - final_entities: Consolidated dictionary of entities from the map phase.

    Raises:
        ValueError: If ingestion, chunking, or map phase fails critically.
    """
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
             # Depending on implementation, this might be okay if some blocks fail
             # But if ALL fail, it's an issue.
             logging.warning("Map phase analysis returned no results. Check individual block analysis logs.")
             # Consider if this should be a hard fail
             # raise ValueError("Map phase analysis failed for all blocks.")
        logging.info(f"Map phase analysis complete. Got {len(map_results)} results.")
    except Exception as e:
        logging.error(f"Map phase analysis failed: {e}", exc_info=True)
        raise ValueError(f"Map phase analysis failed: {e}") from e

    # --- Save State: LargeBlockAnalysisCompleted ---
    # Use the correct function signature: save_state(file_id, stage_value, data)
    if StateStoragePoints.LargeBlockAnalysisCompleted in StateStorageList:
        logging.info(f"Saving state after Stage 1 (LargeBlockAnalysisCompleted) for {file_id}...")
        state_to_save = {
            "large_blocks": large_blocks,
            "map_results": map_results,
            "final_entities": final_entities,
            "raw_text": raw_text
        }
        try:
            state_storage.save_state(file_id, CodeStages.LargeBlockAnalysisCompleted.value, state_to_save)
        except Exception as e:
            # Log error but potentially continue if saving state isn't critical for immediate flow
            logging.error(f"Failed to save state for {CodeStages.LargeBlockAnalysisCompleted.value} (ID: {file_id}): {e}", exc_info=True)
            # Decide if this should raise an error and stop the pipeline
            # raise

    logging.info(f"Stage 1: Initial Processing complete for {file_id}.")
    return raw_text, large_blocks, map_results, final_entities


# --- Stage 2: LargeBlockAnalysisCompleted -> IterativeAnalysisCompleted ---
def perform_iterative_analysis(
    load_state_flag: bool,
    file_id: str,
    # Data passed if not loading state
    raw_text_in: Optional[str] = None,
    large_blocks_in: Optional[List[Dict[str, Any]]] = None,
    map_results_in: Optional[List[Dict[str, Any]]] = None,
    final_entities_in: Optional[Dict[str, List[str]]] = None
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]], Optional[Dict[str, List[str]]], Dict[str, Any]]:
    """Handles the reduce phase of the analysis.

    Args:
        load_state_flag: If True, load input data from the previous stage's saved state.
        file_id: Unique identifier for the document.
        raw_text_in: Raw text (if not loading state).
        large_blocks_in: Coarse chunks (if not loading state).
        map_results_in: Map phase results (if not loading state).
        final_entities_in: Consolidated entities (if not loading state).

    Returns:
        A tuple containing:
        - raw_text: The raw text (loaded or passed through).
        - large_blocks: Coarse chunks (loaded or passed through).
        - map_results: Map results (loaded or passed through).
        - final_entities: Consolidated entities (loaded or passed through).
        - doc_analysis_result: The result of the reduce phase analysis.

    Raises:
        FileNotFoundError: If load_state_flag is True and the state file is missing.
        KeyError: If the loaded state file is missing required keys.
        ValueError: If the reduce phase analysis fails or input data is invalid.
        Exception: For other unexpected errors during loading or processing.
    """
    logging.info(f"Stage 2: Starting Iterative Analysis for {file_id} (Load State: {load_state_flag})...")
    raw_text: Optional[str] = None
    large_blocks: Optional[List[Dict[str, Any]]] = None
    map_results: Optional[List[Dict[str, Any]]] = None
    final_entities: Optional[Dict[str, List[str]]] = None
    doc_analysis_result: Dict[str, Any] = {} # Placeholder

    if load_state_flag:
        logging.info(f"Attempting to load state from {CodeStages.LargeBlockAnalysisCompleted.value} for {file_id}...")
        try:
            # Use the correct function signature: load_state(file_id, stage_value)
            loaded_state = state_storage.load_state(file_id, CodeStages.LargeBlockAnalysisCompleted.value)

            # Validate and assign loaded data
            large_blocks = loaded_state.get("large_blocks")
            map_results = loaded_state.get("map_results")
            final_entities = loaded_state.get("final_entities")
            raw_text = loaded_state.get("raw_text")

            # Check for missing critical data
            if map_results is None or final_entities is None or raw_text is None:
                 missing_keys = [k for k,v in {"map_results": map_results, "final_entities": final_entities, "raw_text": raw_text}.items() if v is None]
                 logging.error(f"Loaded state from {CodeStages.LargeBlockAnalysisCompleted.value} is missing critical keys: {missing_keys}")
                 raise KeyError(f"Loaded state is missing critical keys: {missing_keys}")

            logging.info(f"State loaded successfully for {file_id}.")
        except (FileNotFoundError, KeyError, Exception) as e:
            logging.error(f"Failed to load or use state from {CodeStages.LargeBlockAnalysisCompleted.value} for {file_id}: {e}. Aborting.", exc_info=True)
            raise # Re-raise to stop the pipeline
    else:
        # Use data passed from previous stage, validate it
        logging.info(f"Using data passed from previous stage for {file_id}.")
        if map_results_in is None or final_entities_in is None or raw_text_in is None:
             missing_args = [k for k,v in {"map_results_in": map_results_in, "final_entities_in": final_entities_in, "raw_text_in": raw_text_in}.items() if v is None]
             # This should not happen if the previous stage returned correctly, but check defensively
             logging.error(f"Missing critical data passed from previous stage to Stage 2: {missing_args}")
             raise ValueError(f"Missing critical data passed from previous stage: {missing_args}")
        raw_text = raw_text_in
        large_blocks = large_blocks_in # Can be None/empty if not needed by reduce?
        map_results = map_results_in
        final_entities = final_entities_in

    # --- 3.2 Reduce Phase --- #
    logging.info("Step 2.1: Performing Reduce phase document synthesis...")
    try:
        # Ensure map_results and final_entities are not None before passing
        if map_results is None or final_entities is None:
             # This check might be redundant due to checks above, but adds safety
             raise ValueError("Cannot perform reduce phase: map_results or final_entities are missing.")

        doc_analysis_result = perform_reduce_document_analysis(map_results, final_entities)
        if not doc_analysis_result or "error" in doc_analysis_result:
            # Log the error details if available
            error_msg = doc_analysis_result.get('error', 'Unknown error') if doc_analysis_result else 'No result returned'
            error_detail = doc_analysis_result.get('error_details', 'No details provided') if doc_analysis_result else 'N/A'
            logging.error(f"Reduce phase document analysis failed or returned error: {error_msg} - Details: {error_detail}")
            raise ValueError(f"Reduce phase document analysis failed: {error_msg}")
        logging.info("Reduce phase analysis complete.")
    except Exception as e:
        # Catch potential errors from perform_reduce_document_analysis itself
        logging.error(f"Exception during Reduce phase analysis: {e}", exc_info=True)
        # Raise as ValueError to be caught by run_pipeline's specific handler if desired, or re-raise original
        raise ValueError(f"Reduce phase analysis failed with exception: {e}") from e

    # --- Save State: IterativeAnalysisCompleted ---
    if StateStoragePoints.IterativeAnalysisCompleted in StateStorageList:
        logging.info(f"Saving state after Stage 2 (IterativeAnalysisCompleted) for {file_id}...")
        state_to_save = {
            "doc_analysis_result": doc_analysis_result,
            "large_blocks": large_blocks, # Save even if None/empty, loaded state reflects this
            "map_results": map_results,
            "final_entities": final_entities,
            "raw_text": raw_text
        }
        try:
            state_storage.save_state(file_id, CodeStages.IterativeAnalysisCompleted.value, state_to_save)
        except Exception as e:
            logging.error(f"Failed to save state for {CodeStages.IterativeAnalysisCompleted.value} (ID: {file_id}): {e}", exc_info=True)
            # Decide if this should raise an error and stop the pipeline
            # raise

    logging.info(f"Stage 2: Iterative Analysis complete for {file_id}.")
    # Ensure return values are not None where they shouldn't be (doc_analysis_result handled by checks)
    return raw_text, large_blocks, map_results, final_entities, doc_analysis_result


# --- Stage 3: IterativeAnalysisCompleted -> End ---
def perform_downstream_processing(
    load_state_flag: bool,
    file_id: str,
    pinecone_index: Any, # Replace Any with actual type if known (e.g., pinecone.Index)
    neo4j_driver: Any, # Replace Any with actual type (e.g., neo4j.Driver)
    # Data passed if not loading state
    raw_text_in: Optional[str] = None,
    large_blocks_in: Optional[List[Dict[str, Any]]] = None,
    map_results_in: Optional[List[Dict[str, Any]]] = None,
    # final_entities_in: Optional[Dict[str, List[str]]] = None, # Not needed downstream
    doc_analysis_result_in: Optional[Dict[str, Any]] = None
) -> None: # This function does not return pipeline state
    """Handles fine-grained chunking, chunk analysis, embedding, graph building, and storage.

    Args:
        load_state_flag: If True, load input data from the previous stage's saved state.
        file_id: Unique identifier for the document.
        pinecone_index: Initialized Pinecone index object.
        neo4j_driver: Initialized Neo4j driver object.
        raw_text_in: Raw text (if not loading state).
        large_blocks_in: Coarse chunks (if not loading state, needed for chunking).
        map_results_in: Map results (if not loading state, needed for chunking).
        doc_analysis_result_in: Reduce phase results (if not loading state).

     Raises:
        FileNotFoundError: If load_state_flag is True and the state file is missing.
        KeyError: If the loaded state file is missing required keys.
        ValueError: If critical data is missing or subsequent steps fail.
        ConnectionError: If database connections are missing when needed.
        Exception: For other unexpected errors during loading or processing.
    """
    logging.info(f"Stage 3: Starting Downstream Processing for {file_id} (Load State: {load_state_flag})...")
    raw_text: Optional[str] = None
    large_blocks: Optional[List[Dict[str, Any]]] = None
    map_results: Optional[List[Dict[str, Any]]] = None
    doc_analysis_result: Optional[Dict[str, Any]] = None
    # Other variables are defined within the steps

    if load_state_flag:
        logging.info(f"Attempting to load state from {CodeStages.IterativeAnalysisCompleted.value} for {file_id}...")
        try:
            # Use the correct function signature: load_state(file_id, stage_value)
            loaded_state = state_storage.load_state(file_id, CodeStages.IterativeAnalysisCompleted.value)

            # Validate and assign loaded data
            doc_analysis_result = loaded_state.get("doc_analysis_result")
            large_blocks = loaded_state.get("large_blocks")
            map_results = loaded_state.get("map_results")
            raw_text = loaded_state.get("raw_text")

            # Check for missing critical data needed for this stage
            # raw_text, large_blocks, map_results, doc_analysis_result seem crucial for chunking/analysis
            if raw_text is None or large_blocks is None or map_results is None or doc_analysis_result is None:
                 missing_keys = [k for k,v in {
                    "raw_text": raw_text,
                    "large_blocks": large_blocks,
                    "map_results": map_results,
                    "doc_analysis_result": doc_analysis_result
                 }.items() if v is None]
                 logging.error(f"Loaded state from {CodeStages.IterativeAnalysisCompleted.value} is missing critical keys: {missing_keys}")
                 raise KeyError(f"Loaded state is missing critical keys for Stage 3: {missing_keys}")

            logging.info(f"State loaded successfully for {file_id}.")
        except (FileNotFoundError, KeyError, Exception) as e:
            logging.error(f"Failed to load or use state from {CodeStages.IterativeAnalysisCompleted.value} for {file_id}: {e}. Aborting.", exc_info=True)
            raise # Re-raise to stop the pipeline
    else:
        # Use data passed from previous stage, validate it
        logging.info(f"Using data passed from previous stage for {file_id}.")
         # Check for missing critical data needed for this stage
        if raw_text_in is None or large_blocks_in is None or map_results_in is None or doc_analysis_result_in is None:
             missing_args = [k for k,v in {
                "raw_text_in": raw_text_in,
                "large_blocks_in": large_blocks_in,
                "map_results_in": map_results_in,
                "doc_analysis_result_in": doc_analysis_result_in
             }.items() if v is None]
             logging.error(f"Missing critical data passed from previous stage to Stage 3: {missing_args}")
             raise ValueError(f"Missing critical data passed to Stage 3: {missing_args}")

        raw_text = raw_text_in
        large_blocks = large_blocks_in
        map_results = map_results_in
        doc_analysis_result = doc_analysis_result_in

    # Ensure critical data is present before proceeding (might be slightly redundant after checks above)
    if raw_text is None or large_blocks is None or map_results is None or doc_analysis_result is None:
         # This path should ideally not be reached if validation worked
         raise ValueError(f"Critical data unavailable for Stage 3 processing (file_id: {file_id}). Check state or pipeline flow.")

    # --- 4. Adaptive Fine-Grained Chunking ---
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
             # This might be acceptable in some cases, but often indicates an issue
             logging.warning("Adaptive chunking resulted in zero final chunks.")
             # Decide whether to abort or continue (e.g., if no content suitable for fine chunks)
             # For now, let's allow continuation but subsequent steps will likely do nothing.
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

    # Proceed only if chunks were generated
    if not final_chunks:
        logging.warning("Skipping subsequent steps as no final chunks were generated.")
        logging.info(f"Stage 3: Downstream Processing complete (skipped storage) for {file_id}.")
        return # Exit the function early

    # --- 5. Chunk-Level Analysis ---
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
                # Add analysis result to the chunk dictionary
                chunk['analysis'] = chunk_analysis_result
                chunk['chunk_id'] = chunk_id # Ensure ID is stored back if generated
                chunks_with_analysis.append(chunk)
                processed_chunks += 1
            except Exception as e:
                failed_chunks += 1
                logging.error(f"Failed to analyze chunk {chunk_id}: {e}. Skipping chunk.", exc_info=False) # Avoid excessive traceback logging in loop

        logging.info(f"Chunk analysis complete. Successfully analyzed {processed_chunks}/{len(final_chunks)} chunks. Failed: {failed_chunks}")
        if not chunks_with_analysis:
             # If all chunks failed analysis, we cannot proceed to embedding/storage
             raise ValueError("Chunk analysis failed for all chunks. Aborting subsequent steps.")

    except Exception as e:
        # Catch errors in the loop setup or if the initial check fails
        logging.error(f"Error during chunk analysis process: {e}", exc_info=True)
        raise ValueError(f"Chunk analysis process failed: {e}") from e

    # --- 6. Embedding Generation ---
    logging.info("Step 3.3: Generating embeddings...")
    embeddings_dict: Dict[str, List[float]] = {}
    try:
        embeddings_dict = generate_embeddings(chunks_with_analysis) # Pass chunks that have analysis
        if not embeddings_dict:
            # This implies embedding failed for all successfully analyzed chunks
            raise ValueError("Embedding generation failed for all processed chunks.")
        logging.info(f"Embedding generation complete. Generated {len(embeddings_dict)} embeddings.")
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}", exc_info=True)
        raise ValueError(f"Embedding generation failed: {e}") from e

    # --- 7. Graph Data Construction ---
    logging.info("Step 3.4: Constructing graph data...")
    graph_nodes: List[Dict] = []
    graph_edges: List[Dict] = []
    try:
        # Doc analysis result checked earlier
        graph_nodes, graph_edges = build_graph_data(file_id, doc_analysis_result, chunks_with_analysis)
        logging.info(f"Graph construction complete. Nodes: {len(graph_nodes)}, Edges: {len(graph_edges)}.")
    except Exception as e:
        logging.error(f"Graph data construction failed: {e}", exc_info=True)
        raise ValueError(f"Graph data construction failed: {e}") from e

    # --- 8. Data Storage ---
    logging.info("Step 3.5: Storing data...")
    try:
        # Ensure required data and connections are present
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
            # Assuming store_chunk_metadata_docstore exists and takes chunks_with_analysis
            store_chunk_metadata_docstore(chunks_with_analysis)

        logging.info("Data storage calls complete.")
    except ConnectionError as e:
        logging.error(f"Database connection error during storage: {e}", exc_info=True)
        raise # Re-raise connection errors as they might be critical
    except Exception as e:
        logging.error(f"Data storage failed: {e}", exc_info=True)
        # Depending on requirements, may want to raise this or just log it
        # Raising ValueError allows pipeline to catch it
        raise ValueError(f"Data storage step failed: {e}") from e

    logging.info(f"Stage 3: Downstream Processing complete for {file_id}.")
    # No return value needed as this is the final stage

# Note: Calls to state_storage functions have been updated assuming signatures:
# state_storage.save_state(file_id: str, stage_value: str, data: Any)
# state_storage.load_state(file_id: str, stage_value: str) -> Any
# Verify these match the actual implementation in state_storage.py. 