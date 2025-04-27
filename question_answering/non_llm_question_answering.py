import logging
import time
import os
import uuid
import sys
from typing import List, Dict, Any, Optional

# Adjust path to import from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up one level and then to storage_pipeline directory (using relative path)
storage_pipeline_path = os.path.join(os.path.dirname(current_dir), 'storage_pipeline')

# Add the path to sys.path
sys.path.append(storage_pipeline_path)

# Import required global modules first
from globals import *
from UtilityFunctions import *
from DSAIParams import * # Imports RunCodeFrom, StateStorageList, DocToAddPath etc.
# Import enums for state management and the list of stages
from enums_constants_and_classes import CodeStages, StateStoragePoints, Code_Stages_List

def answer_first_meeting_question(char1, char2, graph, timeline):
    """Answer where two characters first met."""
    if not graph.has_edge(char1, char2):
        return f"{char1} and {char2} never meet in the text."
    
    # Get first meeting info
    first_block = graph[char1][char2]['first_meeting']
    locations = graph[char1][char2]['meeting_locations']
    
    # Get context from timeline
    block_info = timeline.get(first_block, {})
    
    return {
        'answer': f"{char1} and {char2} first met in {block_info.get('title', f'Chapter {first_block}')}",
        'block_number': first_block,
        'locations': locations,
        'context': block_info.get('summary', '')
    }

def count_interactions(char1, char2, graph):
    """Count interactions between characters."""
    if not graph.has_edge(char1, char2):
        return 0
    
    return graph[char1][char2]['count']

def get_meeting_details(char1, char2, graph, timeline):
    """Get details of all meetings between characters."""
    if not graph.has_edge(char1, char2):
        return []
    
    meetings = []
    for block_num in graph[char1][char2]['meetings']:
        block_info = timeline.get(block_num, {})
        
        # Find other characters present at this meeting
        other_chars = []
        for entity in block_info.get('entities', []):
            if (entity['entity_type'] == 'characters' and 
                entity['canonical_name'] not in [char1, char2]):
                other_chars.append(entity['canonical_name'])
        
        # Find locations
        locations = []
        for entity in block_info.get('entities', []):
            if entity['entity_type'] == 'locations':
                locations.append(entity['canonical_name'])
        
        meetings.append({
            'block_number': block_num,
            'title': block_info.get('title', ''),
            'summary': block_info.get('summary', ''),
            'other_characters': other_chars,
            'locations': locations
        })
    
    return meetings

def get_relationship_development(char1, char2, graph, timeline):
    """Analyze how a relationship develops over time."""
    if not graph.has_edge(char1, char2):
        return f"{char1} and {char2} have no relationship in the text."
    
    meetings = get_meeting_details(char1, char2, graph, timeline)
    
    # Create a chronological view of the relationship
    development = []
    for i, meeting in enumerate(meetings):
        stage = "first meeting" if i == 0 else f"meeting {i+1}"
        development.append({
            'stage': stage,
            'block': meeting['block_number'],
            'title': meeting['title'],
            'summary': meeting['summary'],
            'location': meeting['locations'][0] if meeting['locations'] else "unknown location"
        })
    
    return development