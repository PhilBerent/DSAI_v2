#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main script to run the novel storage pipeline."""

import logging
import time
import os
import uuid
import sys

# Adjust path to import from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import required global modules first
try:
    from globals import *
    from UtilityFunctions import *
    from DSAIParams import *
except ImportError as e:
    print(f"Error importing core modules (globals, UtilityFunctions, DSAIParams): {e}")
    raise

# Pipeline components (using absolute imports from DSAI_v2_Scripts level)
try:
    from storage_pipeline.config_loader import DocToAddPath, Chunk_size, Chunk_overlap
    from storage_pipeline.db_connections import get_pinecone_index, get_neo4j_driver, test_connections
    from storage_pipeline.ingestion import ingest_document
    from storage_pipeline.analysis import analyze_document_structure, analyze_chunk_details
    from storage_pipeline.chunking import adaptive_chunking
    from storage_pipeline.embedding import generate_embeddings
    from storage_pipeline.graph_builder import build_graph_data
    from storage_pipeline.storage import store_embeddings_pinecone, store_graph_data_neo4j, store_chunk_metadata_docstore
except ImportError as e:
    print(f"Error during absolute import: {e}")
    print("Ensure you are running this script from a context where DSAI_v2_Scripts is accessible,")
    print("or that DSAI_v2_Scripts is in your PYTHONPATH.")
    # If running directly, the sys.path manipulation in the imported modules should handle it.
    # If still failing, might need explicit path manipulation here too.
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(document_path: str):
    """Executes the full storage pipeline for a given document."""
    start_time = time.time()
    logging.info(f"--- Starting Storage Pipeline for: {document_path} ---")

    # --- 0. Setup & Connections ---
    try:
        logging.info("Testing database connections...")
        # test_connections() # Optional: Run connection tests first
        pinecone_index = get_pinecone_index()
        neo4j_driver = get_neo4j_driver()
        # Generate a unique ID for this document ingestion run
        # Using filename for simplicity, consider more robust ID generation
        document_id = os.path.splitext(os.path.basename(document_path))[0]
        logging.info(f"Generated Document ID: {document_id}")

    except Exception as e:
        logging.error(f"Pipeline setup failed: {e}")
        return

    try:
        # --- 1. Ingestion ---
        logging.info("Step 1: Ingesting document...")
        raw_text = ingest_document(document_path)
        logging.info(f"Ingestion complete. Text length: {len(raw_text)} chars.")

        # --- 2. High-Level Analysis ---
        logging.info("Step 2: Performing high-level document analysis...")
        doc_analysis_result = analyze_document_structure(raw_text)
        # logging.info(f"Document analysis result: {doc_analysis_result}")

        # --- 3. Adaptive Chunking ---
        logging.info("Step 3: Performing adaptive chunking...")
        chunks = adaptive_chunking(
            raw_text,
            document_structure=doc_analysis_result, # Pass structure info
            target_chunk_size=Chunk_size, # Use params from config
            chunk_overlap=Chunk_overlap
        )
        if not chunks:
             raise ValueError("Chunking resulted in zero chunks. Aborting.")
        logging.info(f"Chunking complete. Generated {len(chunks)} chunks.")
        # Add document_id to chunk metadata
        for chunk in chunks:
             chunk['metadata']['document_id'] = document_id


        # --- 4. Chunk-Level Analysis ---
        logging.info("Step 4: Performing detailed chunk analysis...")
        chunks_with_analysis = []
        processed_chunks = 0
        # Process in batches or sequentially - simple sequential for now
        for i, chunk in enumerate(chunks):
            logging.info(f"Analyzing chunk {i+1}/{len(chunks)} (ID: {chunk['chunk_id']})...")
            try:
                chunk_analysis_result = analyze_chunk_details(
                    chunk_text=chunk['text'],
                    chunk_id=chunk['chunk_id'],
                    doc_context=doc_analysis_result # Provide doc context
                )
                chunk['analysis'] = chunk_analysis_result # Store analysis with chunk
                chunks_with_analysis.append(chunk)
                processed_chunks += 1
            except Exception as e:
                logging.error(f"Failed to analyze chunk {chunk['chunk_id']}: {e}. Skipping chunk.")
            # Optional: Add delay between API calls if needed
            # time.sleep(0.5)
        logging.info(f"Chunk analysis complete. Successfully analyzed {processed_chunks}/{len(chunks)} chunks.")
        if not chunks_with_analysis:
             raise ValueError("Chunk analysis failed for all chunks. Aborting.")

        # --- 5. Embedding Generation ---
        logging.info("Step 5: Generating embeddings...")
        embeddings_dict = generate_embeddings(chunks_with_analysis)
        if not embeddings_dict:
            raise ValueError("Embedding generation failed for all processed chunks. Aborting.")
        logging.info(f"Embedding generation complete. Generated {len(embeddings_dict)} embeddings.")

        # --- 6. Graph Data Construction ---
        logging.info("Step 6: Constructing graph data...")
        graph_nodes, graph_edges = build_graph_data(document_id, doc_analysis_result, chunks_with_analysis)
        logging.info(f"Graph construction complete. Nodes: {len(graph_nodes)}, Edges: {len(graph_edges)}.")

        # --- 7. Data Storage ---
        logging.info("Step 7: Storing data...")
        # Store embeddings (and basic chunk metadata) in Pinecone
        store_embeddings_pinecone(pinecone_index, embeddings_dict, chunks_with_analysis)
        # Store graph data in Neo4j
        store_graph_data_neo4j(neo4j_driver, graph_nodes, graph_edges)
        # Placeholder for dedicated document store
        store_chunk_metadata_docstore(chunks_with_analysis)
        logging.info("Data storage complete.")

    except Exception as e:
        logging.exception(f"Pipeline execution failed: {e}") # Log full traceback
    finally:
        # Clean up resources
        if 'neo4j_driver' in locals() and neo4j_driver:
            neo4j_driver.close()
            logging.info("Neo4j driver closed.")
        # Pinecone connection managed by its client library

    end_time = time.time()
    logging.info(f"--- Storage Pipeline Finished for: {document_path} --- ")
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Ensure the script is run from a context where imports work
    # Typically run from the parent directory (DSAI_v2_Scripts) or ensure PYTHONPATH is set
    run_pipeline(DocToAddPath) 