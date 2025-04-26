#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles Step 5: Vector Embedding Generation."""

import logging
from typing import List, Dict, Any
import time
import sys
import os

# Adjust path to import from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import required global modules
try:
    from globals import *
    from UtilityFunctions import *
    from DSAIParams import *
except ImportError as e:
    print(f"Error importing core modules (globals, UtilityFunctions, DSAIParams): {e}")
    raise

from .db_connections import openai_client # OpenAI client
from config_loader import Embeddings_model_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants (could be moved)
MAX_BATCH_EMBED_RETRIES = 3
INITIAL_BATCH_EMBED_BACKOFF = 1 # seconds
EMBEDDING_BATCH_SIZE = 100 # Number of texts to embed in one API call (adjust based on performance/limits)

def generate_embeddings(chunks: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Generates vector embeddings for a list of text chunks using OpenAI.

    Args:
        chunks: A list of chunk dictionaries, each containing at least 'chunk_id' and 'text'.

    Returns:
        A dictionary mapping chunk_id to its vector embedding (List[float]).
    """
    logging.info(f"Starting embedding generation for {len(chunks)} chunks using model: {Embeddings_model_name}")
    embeddings_dict = {}
    texts_to_embed = [chunk['text'] for chunk in chunks]
    chunk_ids = [chunk['chunk_id'] for chunk in chunks]

    for i in range(0, len(texts_to_embed), EMBEDDING_BATCH_SIZE):
        batch_texts = texts_to_embed[i : i + EMBEDDING_BATCH_SIZE]
        batch_ids = chunk_ids[i : i + EMBEDDING_BATCH_SIZE]
        logging.info(f"Processing embedding batch {i // EMBEDDING_BATCH_SIZE + 1} / {len(texts_to_embed) // EMBEDDING_BATCH_SIZE + 1}")

        retries = 0
        backoff = INITIAL_BATCH_EMBED_BACKOFF
        while retries < MAX_BATCH_EMBED_RETRIES:
            try:
                response = openai_client.embeddings.create(
                    input=batch_texts,
                    model=Embeddings_model_name
                )
                # Check if response.data exists and has embeddings
                if response.data and all(hasattr(item, 'embedding') for item in response.data):
                    batch_embeddings = [item.embedding for item in response.data]
                    if len(batch_embeddings) == len(batch_ids):
                        for chunk_id, embedding in zip(batch_ids, batch_embeddings):
                            embeddings_dict[chunk_id] = embedding
                        logging.info(f"Successfully processed batch {i // EMBEDDING_BATCH_SIZE + 1}")
                        break # Success, exit retry loop
                    else:
                        logging.error(f"Mismatch between batch size and embeddings returned for batch starting at index {i}. Retrying...")
                else:
                     logging.error(f"Unexpected response structure or empty data in embedding response for batch starting at index {i}. Retrying...")

            except Exception as e:
                logging.error(f"OpenAI embedding API call failed for batch starting at index {i}: {e}. Retrying in {backoff}s...", exc_info=True)

            # If loop didn't break (i.e., failed)
            retries += 1
            if retries < MAX_BATCH_EMBED_RETRIES:
                time.sleep(backoff)
                backoff *= 2 # Exponential backoff
            else:
                logging.error(f"Failed to generate embeddings for batch starting at index {i} after {MAX_BATCH_EMBED_RETRIES} retries.", exc_info=True)
                # Decide how to handle persistent failure: skip batch, raise error?
                # For now, we'll skip the failed batch IDs but continue
                logging.warning(f"Skipping chunk IDs: {batch_ids}")
                break # Exit retry loop after max retries

    successful_embeddings = len(embeddings_dict)
    if successful_embeddings < len(chunks):
        logging.warning(f"Embedding generation complete, but only {successful_embeddings}/{len(chunks)} chunks were successfully embedded.")
    else:
        logging.info("Embedding generation complete for all chunks.")

    return embeddings_dict 