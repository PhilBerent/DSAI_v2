#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions representing the primary analysis stages of the pipeline."""

import logging
import time
import os
import uuid
import sys
from typing import List, Dict, Any, Tuple, Optional
import traceback # Import traceback for detailed error logging in worker

# Adjust path to import from parent directory (DSAI_v2_Scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import required global modules first
try:
    from globals import *
    from UtilityFunctions import *
    # Import specific params needed
    from DSAIParams import *
    # Import enums needed
    from enums_constants_and_classes import CodeStages, StateStoragePoints
    # Import prompt components
    from prompts import *
    from analysis_functions import *
    from state_storage import *
except ImportError as e:
    print(f"Error importing core modules/prompts/params in primary_analysis_stages: {e}")
    raise

# Import pipeline components
try:
    from storage_pipeline.ingestion import ingest_document
    from storage_pipeline.analysis_functions import perform_map_block_analysis, perform_reduce_document_analysis, analyze_chunk_details
    from storage_pipeline.chunking import coarse_chunk_by_structure, adaptive_chunking
    from storage_pipeline.embedding import generate_embeddings
    from graph_functions import *
    from storage_pipeline.storage import store_embeddings_pinecone, store_graph_data_neo4j, store_chunk_metadata_docstore
    # Import state storage functions using the original module structure
    from DSAIUtilities import calc_est_tokens_per_call # Token estimation
    from llm_calls import calc_num_instances, parallel_llm_calls # Parallel execution
except ImportError as e:
    print(f"Error during absolute import in primary_analysis_stages: {e}")
    print("Ensure necessary modules exist and DSAI_v2_Scripts is accessible.")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Stage 1: Start -> LargeBlockAnalysisCompleted ---
def large_block_analysis(document_path: str, file_id: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]]]:
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
        block_info_list, full_entities_list = perform_map_block_analysis(large_blocks)        
        a=3
        if not block_info_list:
             logging.warning("Map phase analysis returned no results. Check individual block analysis logs.")
        logging.info(f"Map phase analysis complete. Got {len(block_info_list)} results.")
    except Exception as e:
        logging.error(f"Map phase analysis failed: {e}", exc_info=True)
        raise ValueError(f"Map phase analysis failed: {e}") from e

    # --- Save State: LargeBlockAnalysisCompleted ---
    # Use the original save_state(data, storage_point_enum) signature
    if StateStoragePoints.LargeBlockAnalysisCompleted in StateStorageList:
        logging.info(f"Saving state for {StateStoragePoints.LargeBlockAnalysisCompleted.name} (using original state_storage)... ")
        state_to_save = {
            "large_blocks": large_blocks,
            "block_info_list": block_info_list,
            "raw_text": raw_text
        }
        try:
            file_path = LargeBlockAnalysisCompletedFile
            save_state(state_to_save, file_path)
        except Exception as e:
            logging.error(f"Original state_storage.py save failed for {StateStoragePoints.LargeBlockAnalysisCompleted.name}: {e}", exc_info=True)

    logging.info(f"Stage 1: Initial Processing complete for {file_id}.")
    return raw_text, large_blocks, block_info_list, full_entities_list


# --- Stage 2: LargeBlockAnalysisCompleted -> ReduceAnalysisCompleted ---
def perform_reduce_analysis(file_id: str, raw_text: Optional[str] = None,
    large_blocks: Optional[List[Dict[str, Any]]] = None, block_info_list: Optional[List[Dict[str, Any]]] = None,
    final_entities: Optional[Dict[str, List[str]]] = None) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]], Optional[Dict[str, List[str]]], Dict[str, Any]]:
    # Handles the reduce phase of the analysis.

    doc_analysis: Dict[str, Any] = {}

    # --- 3.2 Reduce Phase --- #
    # [Code for Reduce Phase analysis remains unchanged]
    logging.info("Step 2.1: Performing Reduce phase document synthesis...")
    try:
        if block_info_list is None or final_entities is None:
             raise ValueError("Cannot perform reduce phase: block_info_list or final_entities are missing.")

        doc_analysis = perform_reduce_document_analysis(block_info_list, final_entities)
        if not doc_analysis or "error" in doc_analysis:
            error_msg = doc_analysis.get('error', 'Unknown error') if doc_analysis else 'No result returned'
            error_detail = doc_analysis.get('error_details', 'No details provided') if doc_analysis else 'N/A'
            logging.error(f"Reduce phase document analysis failed or returned error: {error_msg} - Details: {error_detail}")
            raise ValueError(f"Reduce phase document analysis failed: {error_msg}")
        logging.info("Reduce phase analysis complete.")
    except Exception as e:
        logging.error(f"Exception during Reduce phase analysis: {e}", exc_info=True)
        raise ValueError(f"Reduce phase analysis failed with exception: {e}") from e


    # --- Save State: ReduceAnalysisCompleted ---
    # Use the original save_state(data, storage_point_enum) signature
    if StateStoragePoints.ReduceAnalysisCompleted in StateStorageList:
        logging.info(f"Saving state for {StateStoragePoints.ReduceAnalysisCompleted.name} (using original state_storage)...")
        state_to_save = {
            "doc_analysis": doc_analysis,
            "large_blocks": large_blocks,
            "block_info_list": block_info_list,
            "final_entities": final_entities,
            "raw_text": raw_text
        }
        try:
            file_path = ReduceAnalysisCompletedFile
            save_state(state_to_save, file_path)
        except Exception as e:
            logging.error(f"Original state_storage.py save failed for {StateStoragePoints.ReduceAnalysisCompleted.name}: {e}", exc_info=True)
            # raise

    logging.info(f"Stage 2: Iterative Analysis complete for {file_id}.")
    return raw_text, large_blocks, block_info_list, final_entities, doc_analysis


# --- Stage 3: ReduceAnalysisCompleted -> End ---
def perform_detailed_chunk_analysis(file_id: str,
    raw_text: Optional[str] = None, large_blocks: Optional[List[Dict[str, Any]]] = None,
    block_info_list: Optional[List[Dict[str, Any]]] = None, doc_analysis: Optional[Dict[str, Any]] = None,
    final_entities: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
    "Handles fine-grained chunking, PARALLEL chunk analysis, embedding, graph building, and storage."

    # --- Ensure critical data is present before proceeding (doc_analysis is now guaranteed non-Optional) --- 
    if raw_text is None or large_blocks is None or block_info_list is None:
         raise ValueError(f"Critical data unavailable for Stage 3 processing (file_id: {file_id}). Check state or pipeline flow.")
    
    # --- 4. Adaptive Fine-Grained Chunking --- #
    # [Code for Adaptive Chunking remains unchanged] ...
    logging.info("Step 3.1: Performing adaptive fine-grained chunking...")
    final_chunks: List[Dict[str, Any]] = []
    try:
        final_chunks = adaptive_chunking(large_blocks, block_info_list, Chunk_Size, Chunk_overlap)

        if not final_chunks:
             logging.warning("Adaptive chunking resulted in zero final chunks.")
        else:
             logging.info(f"Adaptive chunking complete. Generated {len(final_chunks)} chunks.")
             for i, chunk in enumerate(final_chunks):
                 if 'metadata' not in chunk:
                      chunk['metadata'] = {}
                 chunk['metadata']['document_id'] = file_id
                 # --- Ensure chunk_id exists for parallel processing matching --- 
                 if 'chunk_id' not in chunk or not chunk['chunk_id']:
                     chunk['chunk_id'] = f"{file_id}_chunk_{i}"
                     logging.debug(f"Generated chunk_id: {chunk['chunk_id']}")
    except Exception as e:
        logging.error(f"Adaptive fine-grained chunking failed: {e}", exc_info=True)
        raise ValueError(f"Adaptive chunking failed: {e}") from e

    if not final_chunks:
        logging.warning("Skipping subsequent steps as no final chunks were generated.")
        logging.info(f"Stage 3: Downstream Processing complete (skipped storage) for {file_id}.")
        return
    
    # --- 5. Parallel Chunk-Level Analysis --- #
    logging.info("Step 3.2: Performing PARALLEL detailed chunk analysis...")
    chunks_with_analysis: List[Dict[str, Any]] = []
    processed_chunks_count = 0
    failed_chunks_count = 0

    try:
        # 5.1 Estimate tokens and calculate workers
        logging.info("Estimating tokens for chunk analysis...")
        estimated_tokens_per_call = calc_est_tokens_per_call(final_chunks, 
            NumSampleBlocksForAC, EstOutputTokenFractionForAC, chunk_system_message, 
            get_anal_chunk_details_prompt, doc_analysis)

        if estimated_tokens_per_call is None:
            logging.warning("Could not estimate tokens per call for chunk analysis. Defaulting to 1 worker.")
            num_workers = 1
        else:
            num_workers = calc_num_instances(estimated_tokens_per_call)
        logging.info(f"Calculated number of workers for chunk analysis: {num_workers}")
            
        # 5.2 Define the worker function (using closure for doc_analysis)
        # 5.3 Execute in parallel
        logging.info(f"Starting parallel analysis for {len(final_chunks)} chunks with {num_workers} workers...")
        parallel_results = parallel_llm_calls(worker_analyze_chunk, num_workers, final_chunks, 
            AIPlatform, RATE_LIMIT_SLEEP_SECONDS, doc_analysis)     

        # 5.4 Process results
        # Ensure parallel_results length matches final_chunks
        if len(parallel_results) != len(final_chunks):
             logging.warning(f"Mismatch in parallel results length ({len(parallel_results)}) and input chunks ({len(final_chunks)}). Processing available results.")
             # This might indicate an issue with parallel_llm_calls

        for result_item in parallel_results:
            if result_item.get('analysis_status') == 'success':
                chunks_with_analysis.append(result_item)
                processed_chunks_count += 1
            else:
                failed_chunks_count += 1
                chunk_id = result_item.get('chunk_id', 'UNKNOWN_ID')
                error_msg = result_item.get('analysis_error', 'Unknown error')
                # Error already logged in worker, just count failure here
                logging.warning(f"Chunk {chunk_id} failed analysis in parallel worker: {error_msg}")

        logging.info(f"Parallel chunk analysis complete. Success: {processed_chunks_count}, Failed: {failed_chunks_count}")

        if not chunks_with_analysis:
             raise ValueError("Chunk analysis failed for all chunks in parallel execution. Aborting subsequent steps.")
        if StateStoragePoints.DetailedBlockAnalysisCompleted in StateStorageList:
            logging.info(f"Saving state for DetailedBlockAnalysisCompleted... ")
            state_to_save = {
                "file_id": file_id,
                "chunks_with_analysis": chunks_with_analysis,
                "doc_analysis": doc_analysis,
                "block_info_list": block_info_list,
                "final_entities": final_entities
            }
            try:
                file_path = DetailedBlockAnalysisCompletedFile
                save_state(state_to_save, file_path)
            except Exception as e:
                logging.error(f"Original state_storage.py save failed for DetailedBlockAnalysisCompleted: {e}", exc_info=True)
        return file_id, block_info_list, doc_analysis, chunks_with_analysis
    except Exception as e:
        logging.error(f"Error during parallel chunk analysis setup or execution: {e}", exc_info=True)
        raise ValueError(f"Parallel chunk analysis process failed: {e}") from e
# end funct perform_detailed_chunk_analysis return chunks_with_analysis

def get_embeddings(file_id: Optional[str] = None,
        chunks_with_analysis: Optional[List[Dict[str, Any]]] = None,
        doc_analysis: Optional[Dict[str, Any]] = None, block_info_list: Optional[List[Dict[str, Any]]] = None,
        final_entities: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[float]]:
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
    if StateStoragePoints.EmbeddingsCompleted in StateStorageList:
        logging.info(f"Saving state for EmbeddingsCompleted... ")
        state_to_save = {
            "file_id": file_id,
            "embeddings_dict": embeddings_dict,               
            "chunks_with_analysis": chunks_with_analysis,
            "doc_analysis": doc_analysis,
            "block_info_list": block_info_list,
            "final_entities": final_entities
        }
        try:
            file_path = EmbeddingsCompletedFile
            save_state(state_to_save, file_path)
        except Exception as e:
            logging.error(f"Original state_storage.py save failed for EmbeddingsCompleted: {e}", exc_info=True)

    return embeddings_dict, file_id, chunks_with_analysis, doc_analysis, block_info_list

def perform_graph_analyisis(file_id: str, doc_analysis: Dict[str, Any], 
        chunks_with_analysis: List[Dict[str, Any]], embeddings_dict: List[Dict[str, Any]],
        block_info_list: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    logging.info("Step 3.4: Constructing graph data...")
    graph_nodes: List[Dict] = []
    graph_edges: List[Dict] = []
    try:
        graph_nodes, graph_edges = build_graph_data(file_id, doc_analysis, chunks_with_analysis)
        logging.info(f"Graph construction complete. Nodes: {len(graph_nodes)}, Edges: {len(graph_edges)}.")
    except Exception as e:
        logging.error(f"Graph data construction failed: {e}", exc_info=True)
        raise ValueError(f"Graph data construction failed: {e}") from e
    if StateStoragePoints.GraphAnalysisCompleted in StateStorageList:
        logging.info(f"Saving state for EmbeddingsCompleted... ")
        state_to_save = {
            "file_id": file_id,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "embeddings_dict": embeddings_dict,               
            "chunks_with_analysis": chunks_with_analysis,
            "doc_analysis": doc_analysis,
            "block_info_list": block_info_list,
        }
        try:
            file_path = GraphAnalysisCompletedFile
            save_state(state_to_save, file_path)
        except Exception as e:
            logging.error(f"Original state_storage.py save failed for EmbeddingsCompleted: {e}", exc_info=True)
    
    
    return graph_nodes, graph_edges, file_id, embeddings_dict, chunks_with_analysis, doc_analysis, block_info_list
# end graph construction return graph_nodes, graph_edges
def store_data(pinecone_index, neo4j_driver, file_id: str, 
        embeddings_dict: Dict[str, List[float]], chunks_with_analysis: List[Dict[str, Any]], 
        graph_nodes: List[Dict], graph_edges: List[Dict]) -> None:
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

