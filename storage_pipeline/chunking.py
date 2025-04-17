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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration --- (Adjusted)
CHAPTER_PATTERNS = [
    # Find anywhere: CHAPTER, spaces, Roman OR Arabic numerals. Capture type and numeral.
    re.compile(r"(CHAPTER|PART|BOOK)\s+([IVXLCDM\d]+)", re.IGNORECASE),
]
# Keywords to ignore if a potential chapter line starts with them
IGNORE_PREFIXES = ("to ", "heading to ", "contents:", "illustrations:")

SCENE_BREAK_PATTERN = re.compile(r"\n\s*\n\s*\n+") # Example: 3+ newlines
MIN_CHUNK_SIZE_CHARS = 100 # Avoid overly small chunks
DEFAULT_TARGET_CHUNK_SIZE_TOKENS = Chunk_Size # Use from DSAIParams
DEFAULT_CHUNK_OVERLAP_TOKENS = Chunk_overlap # Use from DSAIParams
# Fallback coarse chunk size (if no structure found) - make this configurable?
FALLBACK_COARSE_CHUNK_TARGET_TOKENS = 50000
MIN_COARSE_CHUNK_TOKENS = 10000 # Ensure coarse chunks aren't tiny

def _get_token_count(text: str) -> int:
    """Estimates token count using tiktoken if available, otherwise uses rough char count."""
    if encoding:
        return len(encoding.encode(text))
    else:
        return len(text) // 4 # Rough approximation

# --- NEW: Step 2 - Initial Structural Scan & Coarse Chunking ---
def coarse_chunk_by_structure(full_text: str) -> List[Dict[str, Any]]:
    """
    Splits the document into large structural units (e.g., chapters)
    based on reliable regex patterns. Falls back to large token-based chunks
    if no structure is found.

    Args:
        full_text: The entire document text.

    Returns:
        A list of dictionaries, each representing a large structural unit/block:
            {'text': str, 'type': 'Chapter/Part/Book/FallbackBlock', 'ref': 'Chapter X / Block Y'}
    """
    logging.info("Starting Step 2: Initial structural scan and coarse chunking...")
    coarse_chunks = []
    split_points = [0] # Start at the beginning
    marker_refs = {}
    marker_types = {}

    # Find all matches for chapter patterns
    for pattern_idx, pattern in enumerate(CHAPTER_PATTERNS):
        matches = list(pattern.finditer(full_text))
        # --- IMPORTANT --- #
        # DO NOT iterate over 'matches' here (e.g., for printing) before the main loop below.
        # 'finditer' returns a one-time-use iterator.

        # Iterate through the matches iterator ONLY ONCE here:
        for match in matches: # Removed extra parentheses
            # --- Debugging Print (Optional) ---
            # print(f"Processing match: {match}") # Add print inside the loop if needed
            # --------------------------------

            start_index = match.start()
            # Get context around the match for logging
            context_start = max(0, start_index - 20)
            context_end = min(len(full_text), start_index + 50)
            logging.debug(f"Regex matched potential marker starting at index {start_index}: ...'{full_text[context_start:context_end]}'...")

            # Get the full line containing the match start for filtering
            line_start = full_text.rfind('\n', 0, start_index) + 1
            line_end = full_text.find('\n', start_index)
            if line_end == -1: line_end = len(full_text)
            line_text = full_text[line_start:line_end].strip()
            logging.debug(f"  >> Line for filtering: '{line_text}'")

            # --- Filtering Logic --- #
            is_false_positive = False
            # Check if the line text (not just the match) starts with ignore prefix
            for prefix in IGNORE_PREFIXES:
                if line_text.lower().startswith(prefix):
                    is_false_positive = True
                    logging.debug(f"Ignoring potential marker (prefix match: '{prefix}'): {line_text}")
                    break
            if is_false_positive:
                continue
            # --- End Filtering Logic --- #

            # Condition: Simply check if it's a new split point
            if start_index not in split_points:
                # Check if preceded immediately by non-space (might indicate false positive like "see chapter X")
                if start_index > 0 and not full_text[start_index-1].isspace():
                    logging.debug(f"  >> Match at {start_index} rejected. Preceded by non-space: '{full_text[start_index-1]}'")
                    continue # Skip this match, likely embedded in text

                logging.debug(f"  >> Adding split point: {start_index}")
                split_points.append(start_index)
                # Store the captured numeral (group 2)
                numeral = match.group(2).strip() if len(match.groups()) >= 2 else "UnknownNum"
                marker_refs[start_index] = numeral
                # Store the captured type (group 1)
                marker_types[start_index] = match.group(1).strip().capitalize()

    split_points = sorted(list(set(split_points)))

    if len(split_points) > 1: # Found structural markers
        logging.info(f"Found {len(split_points)-1} potential structural markers after filtering.")
        for i in range(len(split_points)):
            start_pos = split_points[i]
            end_pos = split_points[i+1] if (i+1) < len(split_points) else len(full_text)
            text = full_text[start_pos:end_pos].strip()
            if not text: continue

            # Determine type and reference using stored info
            unit_type = marker_types.get(start_pos, "DetectedStructure") # Use stored type
            ref = marker_refs.get(start_pos, f"Unit {i+1}")

            coarse_chunks.append({'text': text, 'type': unit_type, 'ref': ref})
        logging.info(f"Coarse chunking resulted in {len(coarse_chunks)} structure-based blocks.")

    else: # No reliable structure found, use fallback
        logging.warning("No reliable structural markers found after filtering (using revised regex). Using fallback token-based coarse chunking.")
        # Simple token-based splitting (can use LangChain's splitter for more robustness)
        current_pos = 0
        block_index = 1
        while current_pos < len(full_text):
            # Estimate end position based on tokens
            estimated_end_pos = current_pos + FALLBACK_COARSE_CHUNK_TARGET_TOKENS * 4 # Rough char estimate
            # Find a better split point (e.g., end of sentence/paragraph near estimate)
            split_pos = full_text.find('. ', estimated_end_pos)
            if split_pos == -1: split_pos = full_text.find('\n', estimated_end_pos)
            if split_pos == -1 or split_pos < current_pos + MIN_COARSE_CHUNK_TOKENS * 4:
                split_pos = min(len(full_text), current_pos + FALLBACK_COARSE_CHUNK_TARGET_TOKENS * 5) # Fallback further out

            text = full_text[current_pos:split_pos].strip()
            if text:
                 coarse_chunks.append({
                     'text': text,
                     'type': 'FallbackBlock',
                     'ref': f"Block {block_index}"
                 })
                 block_index += 1
            current_pos = split_pos + 1 # Move past the split point

        logging.info(f"Coarse chunking resulted in {len(coarse_chunks)} fallback blocks.")

    return coarse_chunks


# --- REVISED: Step 4 - Adaptive Text Chunking (Fine-Grained) ---
def adaptive_chunking(
    structural_units: List[Dict[str, Any]], # Now takes coarse chunks/units
    validated_structure: Optional[Dict[str, Any]] = None, # Takes structure from iterative analysis
    target_chunk_size: int = DEFAULT_TARGET_CHUNK_SIZE_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS
) -> List[Dict[str, Any]]:
    """Performs adaptive chunking based on validated structure, paragraphs, and size
       within the provided structural units (coarse chunks).

    Args:
        structural_units: List of coarse chunks/blocks from Step 2.
        validated_structure: Structure list validated/generated in Step 3.
        target_chunk_size: Aimed token size for final chunks.
        chunk_overlap: Token overlap between consecutive final chunks.

    Returns:
        A list of final chunk dictionaries (same format as before).
    """
    logging.info("Starting Step 4: Adaptive fine-grained chunking...")
    final_chunks = []
    chunk_index = 0

    # Map validated structure titles/numbers to their sequence for reference
    structure_map = {}
    val_struct_list = validated_structure.get('structure', []) if validated_structure else []

    for i, struct in enumerate(val_struct_list):
        key = struct.get('title') or struct.get('number') or f"Struct_{i}"
        structure_map[key] = {'order': i, 'type': struct.get('type', 'Structure')}

    # Process each structural unit (coarse chunk)
    for unit_idx, unit in enumerate(structural_units):
        unit_text = unit['text']
        unit_type = unit['type'] # Type from coarse chunking
        unit_ref = unit['ref']   # Ref from coarse chunking (e.g., "Chapter X", "Block Y")

        # Attempt to link this unit to the validated structure for better metadata
        validated_struct_info = structure_map.get(unit_ref) # Try matching ref
        if validated_struct_info:
             struct_meta_type = validated_struct_info['type']
             struct_meta_ref = unit_ref # Use the matched ref
        else:
             # Could try fuzzy matching or use coarse chunk info as fallback
             struct_meta_type = unit_type
             struct_meta_ref = unit_ref

        # Split unit by scene breaks (e.g., multiple newlines)
        scene_texts = SCENE_BREAK_PATTERN.split(unit_text)

        for scene_index, scene_text in enumerate(scene_texts):
            if not scene_text.strip():
                continue
            # Use combined reference for source location
            scene_ref_combined = f"{struct_meta_ref} / Scene {scene_index + 1}"

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
                    final_chunks.append({
                        'chunk_id': chunk_id,
                        'text': current_chunk_text.strip(),
                        'metadata': {
                            'document_id': None, # Add later in run_pipeline
                            'source_location': {
                                # Use validated structure type/ref if available
                                'structure_type': struct_meta_type,
                                'structure_ref': f"{scene_ref_combined} / Paras {para_start_index+1}-{para_index}",
                                'sequence': chunk_index # Overall sequence across doc
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
                        final_chunks.append({
                            'chunk_id': chunk_id,
                            'text': current_chunk_text.strip(),
                            'metadata': {
                                'document_id': None,
                                'source_location': {
                                    'structure_type': struct_meta_type,
                                    'structure_ref': f"{scene_ref_combined} / Paras {para_start_index+1}-{para_index+1}",
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
                final_chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk_text.strip(),
                    'metadata': {
                        'document_id': None,
                        'source_location': {
                            'structure_type': struct_meta_type,
                            'structure_ref': f"{scene_ref_combined} / Paras {para_start_index+1}-{len(paragraphs)}",
                            'sequence': chunk_index
                        }
                    }
                })
                chunk_index += 1

    logging.info(f"Adaptive fine-grained chunking complete. Generated {len(final_chunks)} final chunks.")
    # Post-processing: Ensure sequence is correct
    for i, chunk in enumerate(final_chunks):
        chunk['metadata']['source_location']['sequence'] = i

    return final_chunks

# TODO: Implement more robust token-based overlap logic if needed.
# TODO: Consider using LangChain's text splitters (RecursiveCharacterTextSplitter with length_function=len(encoding.encode)) for potentially simpler implementation. 