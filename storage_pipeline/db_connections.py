#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles establishing connections to external services (OpenAI, Pinecone, Neo4j)."""

import openai
from pinecone import Pinecone
from neo4j import GraphDatabase
import time
import logging
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

# Import configuration parameters from sibling module
# Assuming config_loader ensures these are valid and loaded
# from .config_loader import (
#     PINECONE_INDEX_NAME_LOADED as PINECONE_INDEX_NAME, # Use renamed var
#     NEO4J_URI,
#     NEO4J_USERNAME,
#     NEO4J_PASSWORD
# )

# Use absolute import
try:
    from config_loader import (
        PINECONE_INDEX_NAME_LOADED as PINECONE_INDEX_NAME,
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_PASSWORD
    )
except ImportError as e:
    print(f"Error importing config_loader: {e}")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- OpenAI Connection --- (Implicit through API key)
# Ensure OAI_API_KEY is available from DSAIParams import
try:
    openai.api_key = LLMAPI_KEY
    logging.info("OpenAI API key configured.")
    # Initialize client using the key from DSAIParams
    client = openai.OpenAI(api_key=LLMAPI_KEY)
except NameError:
    logging.error("OAI_API_KEY not found. Ensure it's defined in DSAIParams.py and imported.")
    raise
except Exception as e:
    logging.error(f"Failed to configure OpenAI: {e}")
    raise

# --- Pinecone Connection ---
def get_pinecone_index():
    """Connects to Pinecone and returns the index object."""
    try:
        # Ensure PINECONE_API_KEY and PINECONE_ENVIRONMENT are available from DSAIParams
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        # Check if the index exists
        # Correct way to get list of index names might vary slightly by client version
        # Assuming list_indexes returns a list-like object containing info dicts
        try:
            index_info_list = pc.list_indexes()
            # Extract names - adjust if the structure is different (e.g., index_info.name)
            index_names = [index_info['name'] for index_info in index_info_list]
        except TypeError: # Handle cases where list_indexes might return just names directly (older versions?)
             try:
                  index_names = pc.list_indexes().names # Original attempt as fallback
             except Exception:
                  logging.error("Could not retrieve list of index names from Pinecone response.")
                  raise
        except AttributeError: # Handle cases where the response object doesn't have 'name'
            try:
                 index_names = pc.list_indexes() # assume it just returns names
                 if not isinstance(index_names, list):
                      raise ValueError("list_indexes() did not return a list of names.")
            except Exception as inner_e:
                 logging.error(f"Could not interpret list_indexes response structure: {inner_e}")
                 raise

        if PINECONE_INDEX_NAME not in index_names:
            logging.error(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist in list: {index_names}")
            # Potentially create it here if desired, or raise error
            # pc.create_index(PINECONE_INDEX_NAME, dimension=ModelDimension, metric='cosine') # Example
            raise ConnectionError(f"Pinecone index '{PINECONE_INDEX_NAME}' not found.")

        index = pc.Index(PINECONE_INDEX_NAME)
        logging.info(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
        stats = index.describe_index_stats()
        logging.info(f"Pinecone index stats: {stats}")
        return index
    except Exception as e:
        logging.error(f"Failed to connect to Pinecone: {e}")
        raise


# --- Neo4j Connection ---
def get_neo4j_driver_local():
    """Connects to Neo4j and returns the driver object."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logging.info(f"Successfully connected to Neo4j at: {NEO4J_URI}")
        return driver
    except Exception as e:
        logging.error(f"Failed to connect to Neo4j: {e}")
        raise

def get_neo4j_driver():
    """Connects to Neo4j and returns the driver object."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logging.info(f"Successfully connected to Neo4j at: {NEO4J_URI}")
        return driver
    except Exception as e:
        logging.error(f"Failed to connect to Neo4j: {e}")
        raise

# --- Initialize Connections (Optional: Can be lazy loaded) ---
# You might want to call these functions only when needed rather than at import time
# pinecone_index = get_pinecone_index()
# neo4j_driver = get_neo4j_driver()

def test_connections():
    """Function to explicitly test all connections."""
    logging.info("Testing connections...")
    try:
        # Test OpenAI (simple list models call)
        client.models.list()
        logging.info("OpenAI connection successful (API key valid for listing models).")
    except Exception as e:
        logging.error(f"OpenAI connection test failed: {e}")

    try:
        # Test Pinecone
        index = get_pinecone_index()
        # index.describe_index_stats() # Already logged in get_pinecone_index
        logging.info("Pinecone connection test successful.")
    except Exception as e:
        logging.error(f"Pinecone connection test failed: {e}")

    try:
        # Test Neo4j
        driver = get_neo4j_driver_local()
        driver.close()
        logging.info("Neo4j connection test successful.")
    except Exception as e:
        logging.error(f"Neo4j connection test failed: {e}")

    logging.info("Connection tests finished.")

# If you want to run tests when this module is executed directly:
# if __name__ == "__main__":
#     test_connections() 