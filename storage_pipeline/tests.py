#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Standalone test script for storage pipeline components, e.g., chunking."""

import sys
import os
import logging
import pprint
import re

# Adjust path to import from parent directory (DSAI_v2_Scripts)
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

# Import the function and patterns to test
try:
    from storage_pipeline.chunking import coarse_chunk_by_structure, CHAPTER_PATTERNS
except ImportError as e:
    print(f"Error importing from storage_pipeline.chunking: {e}")
    raise

# Configure logging for the test
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Sample Test Data --- #
def chapterExtractionTest():
    SAMPLE_TEXT = """
    This is the beginning of our story, a preface perhaps, discussing the nature of things
    and setting the scene. It goes on for a while, describing the world and the general
    atmosphere before we get to the main action. Lorem ipsum dolor sit amet, consectetur
    adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip
    ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit
    esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non
    proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

    CHAPTER 1

    Here begins the first real chapter. Our hero, Alex, wakes up to a sunny morning.
    Birds are singing, the coffee is brewing. Alex stretches and thinks about the day ahead.
    Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque
    laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi
    architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas
    sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione
    voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit
    amet, consectetur, adipisci velit.

    CHAPTER 3.

    Now things get a bit more exciting. Alex meets a mysterious stranger named Zara.
    They have a cryptic conversation by the old oak tree. What could it mean? The plot thickens.
    Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae
    consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur? At vero eos et
    accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti
    atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident. CHAPTER XXXV. A challenge arises. Alex must solve a riddle posed by an ancient guardian. Failure means
    doom, success means progress. The pressure is on. It involves llamas and maybe a rubber chicken.
    Similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga.
    Et harum quidem rerum facilis est et expedita distinctio. Nam libero tempore, cum soluta
    nobis est eligendi optio cumque nihil impedit quo minus id quod maxime placeat facere possimus.

    CHAPTER 27

    The riddle is solved! Alex moves on, feeling accomplished but wary. Zara appears again,
    offering a helpful clue, or perhaps a misleading one? Trust is a fragile thing. We also
    see a brief mention of illustrations: to Chapter IV. This line should be ignored.
    Omnis voluptas assumenda est, omnis dolor repellendus. Temporibus autem quibusdam et aut
    officiis debitis aut rerum necessitatibus saepe eveniet ut et voluptates repudiandae sint
    et molestiae non recusandae. Itaque earum rerum hic tenetur a sapiente delectus.

    CHAPTER  18

    The final confrontation looms. Alex prepares for the ultimate test. The fate of the story
    hangs in the balance. Will Alex succeed? Will Zara's true intentions be revealed?
    Ut aut reiciendis voluptatibus maiores alias consequatur aut perferendis doloribus asperiores
    repellat. This marks the end of the main narrative part.
    """

    # --- Test Execution --- #

    print("--- Running Coarse Chunking Test (Logic inside tests.py) ---")

    # --- Logic copied and adapted from chunking.py --- #
    # Define the regex directly here for testing
    # Find anywhere: CHAPTER, spaces, Roman OR Arabic numerals
    test_chapter_pattern = re.compile(r"(CHAPTER)\s+([IVXLCDM\d]+)", re.IGNORECASE)
    test_ignore_prefixes = ("to ", "heading to ", "contents:", "illustrations:")

    print(f"Using Test Regex Pattern: {test_chapter_pattern.pattern}")
    print("-------------------------------------")

    test_split_points = [0] # Start at the beginning
    test_marker_refs = {}

    # Find all matches for chapter patterns in SAMPLE_TEXT
    matches = test_chapter_pattern.finditer(SAMPLE_TEXT)
    for match in (matches):
        start_index = match.start() # Index of 'C' in CHAPTER
        # Log the raw matched line for debugging
        # Get context around the match for logging
        context_start = max(0, start_index - 20)
        context_end = min(len(SAMPLE_TEXT), start_index + 50)
        logging.debug(f"Regex matched potential marker starting at index {start_index}: ...'{SAMPLE_TEXT[context_start:context_end]}'...")

        # Get the full line containing the match start for filtering
        line_start = SAMPLE_TEXT.rfind('\n', 0, start_index) + 1
        line_end = SAMPLE_TEXT.find('\n', start_index)
        if line_end == -1: line_end = len(SAMPLE_TEXT)
        line_text = SAMPLE_TEXT[line_start:line_end].strip()
        logging.debug(f"  >> Line for filtering: '{line_text}'")

        # --- Filtering Logic --- #
        is_false_positive = False
        # Check if the line text (not just the match) starts with ignore prefix
        # This helps filter out mentions in TOC-like lines
        for prefix in test_ignore_prefixes:
            if line_text.lower().startswith(prefix):
                is_false_positive = True
                logging.debug(f"Ignoring potential marker (prefix match: '{prefix}'): {line_text}")
                break
        if is_false_positive:
            continue
        # Add more sophisticated filtering? E.g., check if preceded by non-space?
        # --- End Filtering Logic --- #

        # Condition: Simply check if it's a new split point
        if start_index not in test_split_points:
            # Check if preceded immediately by non-space (might indicate false positive like "see chapter X")
            if start_index > 0 and not SAMPLE_TEXT[start_index-1].isspace():
                 logging.debug(f"  >> Match at {start_index} rejected. Preceded by non-space: '{SAMPLE_TEXT[start_index-1]}'")
                 continue # Skip this match, likely embedded in text
            
            logging.debug(f"  >> Adding split point: {start_index}")
            test_split_points.append(start_index)
            # Store the captured numeral (group 2 in this test regex)
            numeral = match.group(2).strip() if len(match.groups()) >= 2 else "UnknownNum"
            test_marker_refs[start_index] = numeral

    test_split_points = sorted(list(set(test_split_points)))

    print(f"\nFinal test_split_points: {test_split_points}")
    print(f"Final test_marker_refs: {test_marker_refs}")

    # --- Now, create the coarse chunks based on these points --- #
    test_coarse_chunks = []
    if len(test_split_points) > 1: # Found structural markers
        print(f"\nGenerating coarse chunks based on {len(test_split_points)} split points...")
        for i in range(len(test_split_points)):
            start_pos = test_split_points[i]
            end_pos = test_split_points[i+1] if (i+1) < len(test_split_points) else len(SAMPLE_TEXT)
            text = SAMPLE_TEXT[start_pos:end_pos].strip()
            if not text: continue

            # Determine type and reference using stored info
            unit_type = "Chapter" # Assumed Chapter type based on simplified regex
            ref = test_marker_refs.get(start_pos, f"Preface/Unit {i+1}") # Adjust default ref

            test_coarse_chunks.append({'text': f"{text[:100]}..." if len(text)>100 else text, 'type': unit_type, 'ref': ref}) # Store truncated text for brevity

    elif len(SAMPLE_TEXT) > 0:
         # Fallback if no splits found
         print("\nNo structure found, creating single fallback chunk.")
         test_coarse_chunks.append({'text': f"{SAMPLE_TEXT[:100]}...", 'type': 'FallbackBlock', 'ref': 'Block 1'})
    # --- End of copied logic --- #

    print("\n--- Test Coarse Chunking Results ---")
    if test_coarse_chunks:
        print(f"Successfully generated {len(test_coarse_chunks)} coarse chunks:")
        pprint.pprint(test_coarse_chunks)
    else:
        print("Failed to generate any coarse chunks.")

    print("-------------------------------------") 
    
def pathTest():
    # Test the path to the storage pipeline directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one level and then to storage_pipeline directory (using relative path)
    storage_pipeline_path = os.path.join(os.path.dirname(current_dir), 'storage_pipeline')
    sys.path.insert(0, storage_pipeline_path)
    

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