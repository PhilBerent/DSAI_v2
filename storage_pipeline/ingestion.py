#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles Step 1: Document Ingestion and Initial Parsing."""

import os
import logging
import sys
import re

# Adjust path to import from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import required global modules
try:
    from globals import *
    from UtilityFunctions import *
    from DSAIParams import *
    from DSAIUtilities import *  # Import utility functions if needed
except ImportError as e:
    print(f"Error importing core modules (globals, UtilityFunctions, DSAIParams): {e}")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_document(file_path: str) -> str:
    """Reads the text content from a given file path.

    Args:
        file_path: The path to the document file.

    Returns:
        The raw text content of the document as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
        UnicodeDecodeError: If the file encoding is not standard (e.g., UTF-8).
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The specified document file was not found: {file_path}")

    try:
        # Attempt to read with UTF-8 encoding first
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        logging.info(f"Successfully ingested document: {os.path.basename(file_path)}")

        # Basic Cleaning (optional, can be expanded)
        # Replace multiple spaces/tabs within lines with a single space
        cleaned_text = re.sub(r'[ \t]+', ' ', raw_text)
        # Replace multiple consecutive newlines (and surrounding space) with a single newline
        # This preserves paragraph breaks but collapses excessive blank lines
        # cleaned_text = re.sub(r'\s*\n\s*', '\n', cleaned_text).strip()
        cleaned_text= cleanText(cleaned_text)  # Assuming cleanText is a function defined in UtilityFunctions or similar

        # TODO: Add more sophisticated cleaning if needed based on document types
        # e.g., handling specific artifacts from PDF/DOCX conversion if applicable

        return cleaned_text

    except UnicodeDecodeError:
        logging.warning(f"UTF-8 decoding failed for {file_path}. Trying 'latin-1'.")
        try:
            # Fallback to latin-1 or another common encoding if needed
            with open(file_path, 'r', encoding='latin-1') as f:
                raw_text = f.read()
            logging.info(f"Successfully ingested document {os.path.basename(file_path)} using latin-1 encoding.")
            cleaned_text = ' '.join(raw_text.split())
            return cleaned_text
        except Exception as e:
            logging.error(f"Error reading file {file_path} even with fallback encoding: {e}")
            raise IOError(f"Could not read file {file_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading file {file_path}: {e}")
        raise IOError(f"Could not read file {file_path}: {e}")

# Example Usage (can be removed or put under if __name__ == "__main__")
# if __name__ == "__main__":
#     from config_loader import DocToAddPath # Assumes config_loader is in the same dir or path is set
#     try:
#         document_text = ingest_document(DocToAddPath)
#         print(f"Successfully read document. Length: {len(document_text)} characters.")
#         # print(f"First 500 chars:\n{document_text[:500]}")
#     except Exception as e:
#         print(f"Failed to ingest document: {e}") 