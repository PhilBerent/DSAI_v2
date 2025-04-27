#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles Step 3: Adaptive Text Chunking."""

import logging
import re
from typing import List, Dict, Any, Optional
import uuid
import sys
import os
import traceback

# Adjust path to import from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import required global modules
try:
    from globals import *
    from UtilityFunctions import *
    from DSAIParams import *
    from DSAIUtilities import *    
except ImportError as e:
    print(f"Error importing core modules (globals, UtilityFunctions, DSAIParams): {e}")
    raise

# Attempt to import tiktoken, fall back if unavailable (though it should be installed)
try:
    import tiktoken
    # Use the tokenizer appropriate for the embedding model
    encoding = tiktoken.get_encoding("cl100k_base")
except ImportError:
    logging.warning("tiktoken library not found. Chunk size estimations might be less accurate.", exc_info=True)
    encoding = None

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration --- (Adjusted)
CHAPTER_PATTERNS = [
    # Find anywhere: CHAPTER, spaces, Roman OR Arabic numerals. Capture type and numeral.
    re.compile(r"(CHAPTER|PART|BOOK|PREFACE|APPENDIX)\s+([IVXLCDM\d]+)", re.IGNORECASE),
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
    if no structure is found. Cleans common non-standard characters from text.

    Args:
        full_text: The entire document text.

    Returns:
        A list of dictionaries, each representing a large structural unit/block:
            {'text': str, 'type': str, 'ref': str, 'structural_marker': str}
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
            text_slice = full_text[start_pos:end_pos].strip()
            if not text_slice: continue

            # --- ADD CLEANING HERE --- #
            cleaned_text = cleanText(text_slice) # Use the cleanText function
            # --- END CLEANING --- #

            # Determine type and reference using stored info
            unit_type = marker_types.get(start_pos, "DetectedStructure") # Use stored type
            marker = marker_refs.get(start_pos, f"Unit {i+1}")
            structural_marker = f"{unit_type} {marker}"
            ref = f"{i}_{structural_marker}"

            coarse_chunks.append({
                'text': cleaned_text, # Use cleaned text
                'type': unit_type,
                'ref': ref,
                'structural_marker': structural_marker,
                'block_number': i, # 1-based index
                'start_char': start_pos
            })
            aa=1
        logging.info(f"Coarse chunking resulted in {len(coarse_chunks)} structure-based blocks.")

    else: # No reliable structure found, use fallback
        logging.warning("No reliable structural markers found after filtering (using revised regex). Using fallback token-based coarse chunking.")
        current_pos = 0
        block_index = 1
        while current_pos < len(full_text):
            # Estimate end position based on tokens
            estimated_end_pos = current_pos + FALLBACK_COARSE_CHUNK_TARGET_TOKENS * 4
            split_pos = full_text.find('. ', estimated_end_pos)
            if split_pos == -1: split_pos = full_text.find('\n', estimated_end_pos)
            if split_pos == -1 or split_pos < current_pos + MIN_COARSE_CHUNK_TOKENS * 4:
                split_pos = min(len(full_text), current_pos + FALLBACK_COARSE_CHUNK_TARGET_TOKENS * 5)

            text_slice = full_text[current_pos:split_pos].strip()
            if text_slice:
                    # --- ADD CLEANING HERE --- #
                    cleaned_text = cleanText(text_slice) # Use the cleanText function
                    # --- END CLEANING --- #

                    unit_type = 'FallbackBlock'
                    marker = f"Block {block_index}"
                    structural_marker = f"{unit_type} {marker}"
                    ref = f"{i}_{structural_marker}"

                    coarse_chunks.append({
                        'text': cleaned_text, # Use cleaned text
                        'type': unit_type,
                        'structural_marker': structural_marker,
                'ref': ref,                
                        'block_number': i, # 1-based index                        
                        'start_char': current_pos
                    })
                    block_index += 1
            current_pos = split_pos + 1

        logging.info(f"Coarse chunking resulted in {len(coarse_chunks)} fallback blocks.")

    return coarse_chunks

# --- Helper: Split block into paragraph/scene pieces ---
def split_by_paragraph_and_scene(text: str, start_offset: int, structure_ref: str) -> List[Dict[str, Any]]:
    """Splits a block of text into smaller pieces based on paragraphs and scene breaks.

    Args:
        text: The text content of the larger structural block.
        start_offset: The starting character offset of this block within the original document.
        structure_ref: A reference string for the structural block (e.g., "Chapter 1").

    Returns:
        A list of dictionaries, each representing a piece with text and character offsets.
    """
    pieces = []
    current_char_offset = 0
    # Split by scene breaks first (multiple newlines)
    potential_pieces = SCENE_BREAK_PATTERN.split(text)

    for piece_text in potential_pieces:
        trimmed_piece = piece_text.strip()
        if not trimmed_piece:
            # Update offset even for empty lines caused by split to maintain position
            # Find the original text corresponding to this split part to get its length
            original_part_start = text.find(piece_text, current_char_offset)
            if original_part_start != -1:
                 current_char_offset = original_part_start + len(piece_text)
            # else: couldn't find it, offset might drift slightly. Best effort.
            continue

        # Calculate the start/end offsets relative to the original document
        # Find where this trimmed piece starts within the original *block* text
        piece_start_in_block = text.find(trimmed_piece, current_char_offset)
        if piece_start_in_block == -1:
             logging.warning(f"Could not reliably find start offset for piece within block '{structure_ref}'. Offsets may be approximate.")
             # Fallback: use current_char_offset, might not be perfect
             piece_start_in_block = current_char_offset

        piece_original_start = start_offset + piece_start_in_block
        piece_original_end = piece_original_start + len(trimmed_piece)

        pieces.append({
            'text': trimmed_piece,
            'start_char': piece_original_start,
            'end_char': piece_original_end,
            'source_location': {'structure_ref': structure_ref} # Initial source location
        })

        # Update the current character offset within the block text for the next search
        current_char_offset = piece_start_in_block + len(trimmed_piece)

    # Further split by single newlines (paragraphs) if necessary? - Current logic handles paragraphs implicitly
    # The SCENE_BREAK_PATTERN split handles major breaks. If finer paragraph splitting is needed,
    # it could be added here, but often splitting by multiple newlines is sufficient.

    logging.debug(f"Split block '{structure_ref}' into {len(pieces)} pieces.")
    return pieces

import re

import re
import logging
from typing import List, Dict, Any

def align_start_to_boundary(text: str) -> str:
    """
    Skips forward to the first likely sentence or word boundary.
    Removes leading partial words or punctuation fragments.
    """
    # Try skipping to the next sentence boundary
    sentence_start = re.search(r'(?<=[.?!])["’”\']?\s+[A-Z]', text)
    if sentence_start:
        return text[sentence_start.start() + 1:].lstrip()

    # If no sentence start, skip to first full word (space followed by letter)
    word_start = re.search(r'\s+[A-Za-z]', text)
    if word_start:
        return text[word_start.start() + 1:].lstrip()

    # Fallback: return as-is
    return text

def safe_truncate_text(text: str, max_length: int) -> str:
    """
    Truncates text to the nearest sentence or word boundary within max_length.
    """
    if len(text) <= max_length:
        return text.strip()

    # Try to find a sentence boundary
    sentence_endings = list(re.finditer(r'[.!?]["’”\']?\s', text[:max_length + 1]))
    if sentence_endings:
        return text[:sentence_endings[-1].end()].strip()

    # If no sentence boundary, try to find a word boundary
    word_boundary = text[:max_length].rstrip()
    last_space = word_boundary.rfind(' ')
    if last_space != -1 and last_space > max_length * 0.5:
        return word_boundary[:last_space].strip()

    # Fallback: truncate at max_length
    return word_boundary

# --- REVISED: Step 4 - Adaptive Text Chunking (Fine-Grained) ---
def adaptive_chunking(
    structural_units: List[Dict[str, Any]],  # Coarse chunks/units
    block_info_list: List[Dict[str, Any]],       # Corresponding analysis results from map phase
    target_chunk_size: int = DEFAULT_TARGET_CHUNK_SIZE_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS
) -> List[Dict[str, Any]]:
    """Performs adaptive chunking based on paragraphs and size,
       using metadata from the map phase analysis.

    Args:
        structural_units: List of coarse chunks/blocks from Step 2.
        block_info_list: List of analysis results from Step 3 Map phase,
                     corresponding 1:1 with structural_units.
        target_chunk_size: Aimed token size for final chunks.
        chunk_overlap: Token overlap between consecutive final chunks.

    Returns:
        A list of final chunk dictionaries.
    """
    logging.info("Starting Step 4: Adaptive fine-grained chunking using map results...")
    final_chunks = []
    chunk_index = 0

    max_chunk_size = int(target_chunk_size * 2.5)

    # Ensure block_info_list align with structural_units
    if len(structural_units) != len(block_info_list):
        logging.error(f"Mismatch between structural_units ({len(structural_units)}) and block_info_list ({len(block_info_list)}) count. Cannot reliably link metadata.")
        # Decide how to handle: fallback or raise error? Raising for now.
        raise ValueError("Structural units and map results count mismatch in adaptive_chunking.")

    # Process each structural unit (coarse chunk)
    for unit_idx, unit in enumerate(structural_units):
        unit_text = unit.get('text', '')
        # --- Get type from block_info_list instead of the coarse unit --- #
        # original_unit_type = unit.get('type', 'UnknownType') # Type from coarse chunking
        original_unit_ref = unit.get('ref', f'Unit_{unit_idx+1}')   # Ref from coarse chunking (e.g., "Unit 1")

        # Get the corresponding analysis from the map phase
        map_analysis = block_info_list[unit_idx]
        unit_type = map_analysis.get('unit_type', 'UnknownType')  # <-- Use type from map_analysis

        # Use the structural marker found during map phase for better reference
        # Fallback to original ref if marker wasn't found or is empty
        struct_meta_ref = map_analysis.get('structural_marker_found')
        if not struct_meta_ref:  # Handle None or empty string
            struct_meta_ref = original_unit_ref

        # Use the type determined by the map phase analysis
        struct_meta_type = unit_type

        # Split unit by scene breaks (e.g., multiple newlines)
        scene_texts = SCENE_BREAK_PATTERN.split(unit_text)

        for scene_index, scene_text in enumerate(scene_texts):
            if not scene_text.strip():
                continue
            # Use combined reference for source location
            scene_ref_combined = f"{struct_meta_ref} / Scene {scene_index + 1}"

            paragraphs = [p.strip() for p in scene_text.split('\n') if p.strip()]

            current_chunk_text = ""
            current_chunk_token_count = 0
            para_start_index = 0
            para_index = 0

            while para_index < len(paragraphs):
                paragraph = paragraphs[para_index]
                para_token_count = _get_token_count(paragraph)

                # --- NEW: Enforce hard maximum chunk size ---
                if current_chunk_token_count + para_token_count > max_chunk_size:
                    # Force break current_chunk_text if it’s already too big
                    if current_chunk_token_count > 0:
                        chunk_id = str(uuid.uuid4())
                        final_chunks.append({
                            'chunk_id': chunk_id,
                            'text': current_chunk_text.strip(),
                            'metadata': {
                                'document_id': None,
                                'source_location': {
                                    'structure_type': struct_meta_type,
                                    'structure_ref': f"{scene_ref_combined} / Paras {para_start_index+1}-{para_index}",
                                    'sequence': chunk_index
                                }
                            }
                        })
                        chunk_index += 1
                        overlap_text = " ".join(current_chunk_text.split()[-chunk_overlap:])
                        current_chunk_text = overlap_text
                        current_chunk_token_count = _get_token_count(current_chunk_text)
                        para_start_index = para_index
                        continue

                    # If the paragraph *alone* is too big, break it mid-paragraph at word boundary
                    words = paragraph.split()
                    truncated_paragraph = []
                    token_total = 0
                    for word in words:
                        word_token_count = _get_token_count(word)
                        if token_total + word_token_count > max_chunk_size:
                            break
                        truncated_paragraph.append(word)
                        token_total += word_token_count

                    chunk_text = " ".join(truncated_paragraph)
                    chunk_id = str(uuid.uuid4())
                    final_chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk_text.strip(),
                        'metadata': {
                            'document_id': None,
                            'source_location': {
                                'structure_type': struct_meta_type,
                                'structure_ref': f"{scene_ref_combined} / Paras {para_start_index+1}-{para_index+1} (split)",
                                'sequence': chunk_index
                            }
                        }
                    })
                    chunk_index += 1

                    # Put the remaining part of the paragraph back for future processing
                    remaining_text = " ".join(words[len(truncated_paragraph):])
                    if remaining_text:
                        paragraphs.insert(para_index + 1, remaining_text)
                    para_index += 1
                    current_chunk_text = ""
                    current_chunk_token_count = 0
                    para_start_index = para_index
                    continue

                # If adding this paragraph exceeds target size significantly, finalize the current chunk
                if current_chunk_token_count > 0 and (current_chunk_token_count + para_token_count > target_chunk_size * 1.2):  # Allow some overshoot
                    chunk_id = str(uuid.uuid4())
                    final_chunks.append({
                        'chunk_id': chunk_id,
                        'text': current_chunk_text.strip(),
                        'metadata': {
                            'document_id': None, # Add later in run_pipeline
                            'source_location': {
                                'structure_type': struct_meta_type, # Use derived type
                                'structure_ref': f"{scene_ref_combined} / Paras {para_start_index+1}-{para_index}", # Use derived ref
                                'sequence': chunk_index
                            }
                        }
                    })
                    chunk_index += 1
                    # Start new chunk with overlap
                    overlap_text = " ".join(current_chunk_text.split()[-chunk_overlap:])
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
                                    'structure_type': struct_meta_type, # Use derived type
                                    'structure_ref': f"{scene_ref_combined} / Paras {para_start_index+1}-{para_index+1}", # Use derived ref
                                    'sequence': chunk_index
                                }
                            }
                        })
                        chunk_index += 1
                        # Start new chunk with overlap
                        overlap_text = " ".join(current_chunk_text.split()[-chunk_overlap:])
                        current_chunk_text = overlap_text
                        current_chunk_token_count = _get_token_count(current_chunk_text)
                        para_start_index = para_index + 1

                para_index += 1

            # Add any remaining text as the last chunk for this scene
            if current_chunk_text.strip() and len(current_chunk_text) > MIN_CHUNK_SIZE_CHARS:
                chunk_id = str(uuid.uuid4())
                final_chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk_text.strip(),
                    'metadata': {
                        'document_id': None,
                        'source_location': {
                            'structure_type': struct_meta_type, # Use derived type
                            'structure_ref': f"{scene_ref_combined} / Paras {para_start_index+1}-{len(paragraphs)}", # Use derived ref
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