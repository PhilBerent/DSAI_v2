#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loads configuration parameters and initializes clients."""

import os
import logging
# Assuming running from DSAI_v2_Scripts or parent
import sys
from DSAIParams import *  # Import all parameters from DSAIParams

# Adjust path if necessary - assuming this file is now in DSAI_v2_Scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = script_dir # Or os.path.dirname(script_dir) if structure changes
sys.path.insert(0, parent_dir)

# Imports for parameters
from DSAIParams import *
import google.generativeai as genai # For Gemini configuration

# --- Load Basic Parameters --- #
# Example: Keep existing config loading if any
DocToAddPath = r"C:\Users\Phil\Documents\DataSphere AI\DataSphere AI Not Code\Inputs And Outputs\Texts For Testing\Pride and PrejudiceDingo.txt"
Chunk_size = 1000  # Target chunk size in tokens (example)
Chunk_overlap = 100 # Overlap in tokens (example)
LLM_model = LLM_model # Use the model name from DSAIParams

# Expose Pinecone index name for other modules
# Assuming PINECONE_INDEX_NAME is defined in DSAIParams.py
PINECONE_INDEX_NAME_LOADED = Pinecone_Index_name

# Expose Neo4j connection details for other modules
# Assuming NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD are defined in DSAIParams.py
NEO4J_URI = NEO4J_URI
NEO4J_USERNAME = NEO4J_USERNAME
NEO4J_PASSWORD = NEO4J_PASSWORD

# --- LLM Client Configuration --- #

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Gemini client if selected
AIPlatform_upper = AIPlatform.upper()
if AIPlatform_upper == "GEMINI":
    try:
        genai.configure(api_key=LLMAPI_KEY)
        logging.info("Gemini client configured via config_loader.")
    except Exception as e:
        logging.error(f"Failed to configure Gemini client in config_loader: {e}")
        raise
elif AIPlatform_upper == "OPENAI":
    # OpenAI client assumed to be configured via db_connections import
    # No specific configuration needed here as the client is imported elsewhere
    logging.info("OpenAI platform selected. Client configuration handled by importer.")
else:
    raise ValueError(f"Unsupported AIPlatform specified in DSAIParams: {AIPlatform}")

# --- Other Configurations --- #
# e.g., Database connection strings if not handled elsewhere

logging.info(f"Config loaded: AIPlatform={AIPlatform}, Model={LLM_model}, DocPath={DocToAddPath}") 