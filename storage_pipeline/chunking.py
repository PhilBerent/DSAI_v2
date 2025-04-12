#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles Step 3: Adaptive Text Chunking."""

import logging
import re
from typing import List, Dict, Any, Optional
import uuid
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

# Attempt to import tiktoken, fall back if unavailable (though it should be installed)
try:
    import tiktoken
    # Use the tokenizer appropriate for the embedding model
    encoding = tiktoken.get_encoding("cl100k_base")
except ImportError:
    logging.warning("tiktoken library not found. Chunk size estimations might be less accurate.")
    encoding = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration --- (Could move to constants.py or config_loader)
# Define patterns for potential structural breaks (simple examples)
# More robust parsing might involve NLP libraries or specific format parsers
CHAPTER_PATTERNS = [
    re.compile(r"^\s*Chapter\s+[IVXLCDM\d]+\s*[:.]?\s*", re.IGNORECASE),
    re.compile(r"^\s*Part\s+[IVXLCDM\d]+\s*[:.]?\s*", re.IGNORECASE),
    re.compile(r"^\s*Book\s+[IVXLCDM\d]+\s*[:.]?\s*", re.IGNORECASE)
]
SCENE_BREAK_PATTERN = re.compile(r"\n\s*\n\s*\n+") # Example: 3+ newlines
MIN_CHUNK_SIZE_CHARS = 100 # Avoid overly small chunks
DEFAULT_TARGET_CHUNK_SIZE_TOKENS = 400 # Target size (align with config_loader?)
DEFAULT_CHUNK_OVERLAP_TOKENS = 50 # Overlap (align with config_loader?)

def _get_token_count(text: str) -> int:
    """Estimates token count using tiktoken if available, otherwise uses rough char count."""
    if encoding:
        return len(encoding.encode(text))
    else:
        return len(text) // 4 # Rough approximation

def adaptive_chunking(
    full_text: str,
    document_structure: Optional[Dict[str, Any]] = None,
    target_chunk_size: int = DEFAULT_TARGET_CHUNK_SIZE_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS
) -> List[Dict[str, Any]]:
    """Performs adaptive chunking based on structure, paragraphs, and size.

    Args:
        full_text: The entire document text.
        document_structure: Optional pre-analyzed structure (list of chapters/sections).
        target_chunk_size: Aimed token size for chunks.
        chunk_overlap: Token overlap between consecutive chunks.

    Returns:
        A list of chunk dictionaries, each containing:
            'chunk_id': Unique identifier.
            'text': The chunk text content.
            'metadata': {
                'document_id': Placeholder (to be added later),
                'source_location': {
                    'structure_type': 'Chapter/Section/Paragraph/Split',
                    'structure_ref': 'Chapter X / Paragraph Y',
                    'sequence': chunk_index
                }
            }
    """
    logging.info("Starting adaptive chunking...")
    chunks = []
    chunk_index = 0
    current_position = 0

    # --- Initial Split by Chapters/Sections (if available) ---
    structural_units = []
    if document_structure and document_structure.get('structure'):
        # TODO: Improve logic to find exact start/end positions of chapters
        # This simple version just uses the titles as markers
        # A more robust method would parse start/end indices during analysis
        logging.warning("Chunking by provided structure is basic. Consider enhancing structure analysis.")
        last_pos = 0
        for i, struct in enumerate(document_structure['structure']):
            title = struct.get('title', f"Unit {struct.get('number', i+1)}")
            # Very basic search for the title - prone to errors
            match = re.search(re.escape(title), full_text[last_pos:], re.IGNORECASE)
            if match:
                start_pos = last_pos + match.start()
                if start_pos > last_pos:
                    structural_units.append({
                        'text': full_text[last_pos:start_pos],
                        'type': 'Inter-structure',
                        'ref': f"Before {title}"
                    })
                structural_units.append({
                    'text': "", # Placeholder, actual text is from start_pos
                    'type': struct.get('type', 'Chapter/Section'),
                    'ref': title,
                    'start_pos': start_pos
                })
                last_pos = start_pos
            else:
                logging.warning(f"Could not find structure title '{title}' in text. Chunking might be less accurate.")
        structural_units.append({
            'text': full_text[last_pos:],
            'type': 'Final Section',
            'ref': 'After last structure'
        })
        # Assign text content based on start positions
        for i in range(len(structural_units) - 1):
            if 'start_pos' in structural_units[i]:
                structural_units[i]['text'] = full_text[structural_units[i]['start_pos']:structural_units[i+1].get('start_pos', len(full_text))]

    else:
        # No structure provided, treat whole doc as one unit
        structural_units.append({'text': full_text, 'type': 'Document', 'ref': 'Full'})

    # --- Process each structural unit --- #
    for unit in structural_units:
        unit_text = unit['text']
        unit_type = unit['type']
        unit_ref = unit['ref']
        unit_pos = 0

        # Split unit by scene breaks (e.g., multiple newlines)
        scene_texts = SCENE_BREAK_PATTERN.split(unit_text)

        for scene_index, scene_text in enumerate(scene_texts):
            if not scene_text.strip():
                continue
            scene_ref = f"{unit_ref} / Scene {scene_index + 1}"

            # Split scene by paragraphs
            paragraphs = [p.strip() for p in scene_text.split('\n') if p.strip()]

            current_chunk_text = ""
            current_chunk_token_count = 0
            para_start_index = 0

            for para_index, paragraph in enumerate(paragraphs):
                para_token_count = _get_token_count(paragraph)

                # If adding this paragraph exceeds target size significantly, finalize the current chunk
                if current_chunk_token_count > 0 and (current_chunk_token_count + para_token_count > target_chunk_size * 1.2): # Allow some overshoot
                    chunk_id = str(uuid.uuid4())
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': current_chunk_text.strip(),
                        'metadata': {
                            'document_id': None, # Add later
                            'source_location': {
                                'structure_type': unit_type,
                                'structure_ref': f"{scene_ref} / Paras {para_start_index+1}-{para_index}",
                                'sequence': chunk_index
                            }
                        }
                    })
                    chunk_index += 1
                    # Start new chunk with overlap (simple overlap for now)
                    overlap_text = " ".join(current_chunk_text.split()[-chunk_overlap:]) # Rough word overlap
                    current_chunk_text = overlap_text + " " + paragraph
                    current_chunk_token_count = _get_token_count(current_chunk_text)
                    para_start_index = para_index
                else:
                    # Add paragraph to current chunk
                    separator = " " if current_chunk_text else ""
                    current_chunk_text += separator + paragraph
                    current_chunk_token_count = _get_token_count(current_chunk_text)

                    # If current chunk meets target size, finalize it
                    if current_chunk_token_count >= target_chunk_size:
                        chunk_id = str(uuid.uuid4())
                        chunks.append({
                            'chunk_id': chunk_id,
                            'text': current_chunk_text.strip(),
                            'metadata': {
                                'document_id': None,
                                'source_location': {
                                    'structure_type': unit_type,
                                    'structure_ref': f"{scene_ref} / Paras {para_start_index+1}-{para_index+1}",
                                    'sequence': chunk_index
                                }
                            }
                        })
                        chunk_index += 1
                        # Start new chunk with overlap
                        overlap_text = " ".join(current_chunk_text.split()[-chunk_overlap:]) # Rough word overlap
                        current_chunk_text = overlap_text
                        current_chunk_token_count = _get_token_count(current_chunk_text)
                        para_start_index = para_index + 1 # Next para starts the potential new chunk

            # Add any remaining text as the last chunk for this scene
            if current_chunk_text.strip() and len(current_chunk_text) > MIN_CHUNK_SIZE_CHARS:
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk_text.strip(),
                    'metadata': {
                        'document_id': None,
                        'source_location': {
                            'structure_type': unit_type,
                            'structure_ref': f"{scene_ref} / Paras {para_start_index+1}-{len(paragraphs)}",
                            'sequence': chunk_index
                        }
                    }
                })
                chunk_index += 1

    logging.info(f"Adaptive chunking complete. Generated {len(chunks)} chunks.")
    # Post-processing: Ensure sequence is correct if chunking logic has issues
    for i, chunk in enumerate(chunks):
        chunk['metadata']['source_location']['sequence'] = i

    return chunks

# TODO: Implement more robust token-based overlap logic if needed.
# TODO: Consider using LangChain's text splitters (RecursiveCharacterTextSplitter with length_function=len(encoding.encode)) for potentially simpler implementation. 