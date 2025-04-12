#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Constants for the storage utility modules."""

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

# Add any storage-specific constants below if needed
# Example:
# DEFAULT_PINECONE_NAMESPACE = "novels" 