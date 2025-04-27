import spacy
import re
from typing import List, Dict, Tuple, Any
from globals import *
from UtilityFunctions import *
from DSAIParams import *
from DSAIUtilities import *

import spacy
import re
from typing import List, Dict, Tuple, Any
import os
import sys

# Get the directory where the script is
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to storage_pipeline inside DSAI_v2_Scripts
storage_pipeline_path = os.path.join(script_dir, 'storage_pipeline')

# Add to sys.path
sys.path.insert(0, storage_pipeline_path)

# Now import
from storage_pipeline.alias_resolution import *



def pathTest():
    a=3
    # # Add the path to sys.path
    # sys.path.append(storage_pipeline_path)

    # # Now you can import directly
    # try:
    #     from alias_resolution import *
    #     print("Successfully imported alias_resolution from storage_pipeline.")
    # except ImportError as e:
    #     print(f"Error importing alias_resolution: {e}")
pathTest()