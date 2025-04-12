#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loads configuration parameters required for the storage pipeline."""

import sys
import os

# Adjust the path to include the parent directory (DSAI_v2_Scripts)
# This allows importing DSAIParams directly
# Note: Consider more robust path management for larger projects
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

# Keep the rest of the specific parameter loading logic
try:
    # --- Define Neo4j Params (Replace with actual values in DSAIParams.py) ---
    # TODO: Add NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD to DSAIParams.py
    # Use existing variables if they are already defined in DSAIParams
    NEO4J_URI = NEO4J_URI if 'NEO4J_URI' in locals() else "bolt://localhost:7687" # Example
    NEO4J_USERNAME = NEO4J_USERNAME if 'NEO4J_USERNAME' in locals() else "neo4j" # Example
    NEO4J_PASSWORD = NEO4J_PASSWORD if 'NEO4J_PASSWORD' in locals() else "your_neo4j_password" # Example - GET FROM DSAIParams

    # Rename for clarity within this module/package
    PINECONE_INDEX_NAME_LOADED = Index_name

    # Assign GPT-4o if Glt_model isn't already set to it
    CHAT_MODEL_NAME = "gpt-4o" if Glt_model != "gpt-4o" else Glt_model

except NameError as e:
    print(f"Missing required parameter in DSAIParams.py: {e}")
    # Handle missing specific parameters
    raise

# Optional: Validate parameters (e.g., check if keys are set)
if not OAI_API_KEY:
    raise ValueError("OAI_API_KEY is not set in DSAIParams.py or environment variables.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in DSAIParams.py or environment variables.")
# Add validation for Neo4j params once added

print("Configuration loaded successfully.") 