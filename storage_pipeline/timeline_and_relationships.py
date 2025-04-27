import logging
import time
import os
import uuid
import sys
from typing import List, Dict, Any, Optional

# Adjust path to import from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import required global modules first
from globals import *
from UtilityFunctions import *
from DSAIParams import * # Imports RunCodeFrom, StateStorageList, DocToAddPath etc.
# Import enums for state management and the list of stages
from enums_constants_and_classes import CodeStages, StateStoragePoints, Code_Stages_List
from primary_analysis_stages import *
from alias_resolution import *


def build_entity_timeline(resolved_registry, block_info_list):
    """
    Build a timeline of entity appearances throughout the text.
    """
    timeline = {}
    
    # Process each block chronologically
    for block_info in sorted(block_info_list, key=lambda x: x.get('block_number', 0)):
        block_num = block_info.get('block_number')
        
        # Skip blocks without numbers
        if block_num is None:
            continue
            
        block_entries = []
        
        # Go through each entity type and find who appears in this block
        for entity_type in ['characters', 'locations', 'organizations']:
            for entity in block_info['key_entities_in_block'].get(entity_type, []):
                entity_name = entity['name']
                
                # Map to canonical name
                canonical_name = map_to_canonical_name(entity_name, resolved_registry[entity_type])
                
                if canonical_name:
                    block_entries.append({
                        'canonical_name': canonical_name,
                        'used_name': entity_name,
                        'entity_type': entity_type,
                        'description': entity.get('description', '')
                    })
        
        # Store in timeline
        timeline[block_num] = {
            'title': block_info.get('title', f'Block {block_num}'),
            'summary': block_info.get('block_summary', ''),
            'entities': block_entries
        }
    
    return timeline

def build_relationship_graph(resolved_registry, entity_timeline):
    """
    Build a graph of relationships between entities.
    """
    import networkx as nx
    
    G = nx.Graph()
    
    # Add all entities as nodes
    for entity_type in resolved_registry:
        for canonical_name, entity_data in resolved_registry[entity_type].items():
            G.add_node(
                canonical_name, 
                type=entity_type,
                alternate_names=list(entity_data['all_alternate_names']),
                first_appearance=min(entity_data['block_occurrences']) if entity_data['block_occurrences'] else None
            )
    
    # Add edges based on co-occurrence in blocks
    for block_num, block_data in entity_timeline.items():
        entities_in_block = [(e['canonical_name'], e['entity_type']) for e in block_data['entities']]
        
        # Characters present at a location
        characters = [e[0] for e in entities_in_block if e[1] == 'characters']
        locations = [e[0] for e in entities_in_block if e[1] == 'locations']
        
        # Connect characters to locations
        for char in characters:
            for loc in locations:
                if G.has_edge(char, loc):
                    # Update existing relationship
                    G[char][loc]['meetings'].append(block_num)
                    G[char][loc]['count'] += 1
                else:
                    # Create new relationship
                    G.add_edge(
                        char, loc,
                        relationship_type='visited',
                        meetings=[block_num],
                        count=1,
                        first_meeting=block_num
                    )
        
        # Character interactions with each other
        for i, char1 in enumerate(characters):
            for char2 in characters[i+1:]:
                if G.has_edge(char1, char2):
                    # Update existing relationship
                    G[char1][char2]['meetings'].append(block_num)
                    G[char1][char2]['count'] += 1
                    
                    # Update locations of meetings
                    G[char1][char2]['meeting_locations'].extend(locations)
                else:
                    # Create new relationship
                    G.add_edge(
                        char1, char2,
                        relationship_type='interacted',
                        meetings=[block_num],
                        count=1,
                        first_meeting=block_num,
                        meeting_locations=locations.copy()
                    )
    
    return G

def map_to_canonical_name(name, entity_registry):
    """Map a name to its canonical form using the registry."""
    # Direct match with a canonical name
    if name in entity_registry:
        return name
        
    # Check if it's an alternate name of any canonical entity
    for canonical_name, entity_data in entity_registry.items():
        if name in entity_data['all_alternate_names']:
            return canonical_name
    
    # Try a more flexible match
    for canonical_name, entity_data in entity_registry.items():
        # Check if name is similar to canonical name
        if is_name_match(name, canonical_name):
            return canonical_name
            
        # Check if name is similar to any alternate name
        for alt_name in entity_data['all_alternate_names']:
            if is_name_match(name, alt_name):
                return canonical_name
    
    return None  # No match found