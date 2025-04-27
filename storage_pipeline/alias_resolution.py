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

def is_name_match(name1, name2):
    """
    Determine if two names likely refer to the same entity using various heuristics.
    """
    # Convert to lowercase for comparison
    name1 = name1.lower().strip()
    name2 = name2.lower().strip()
    
    # Direct equality
    if name1 == name2:
        return True
    
    # Extract name components
    name1_parts = name1.split()
    name2_parts = name2.split()
    
    # Check for subname relationships
    if (name1 in name2) or (name2 in name1):
        # If one is contained in the other, they might be the same entity
        # But check some special cases first
        
        # Check for title differences that indicate different people
        titles1 = extract_titles(name1)
        titles2 = extract_titles(name2)
        
        if titles1 and titles2 and titles1 != titles2:
            # Different titles might indicate different people
            # E.g., "Mrs. Smith" vs "Miss Smith" are likely different people
            if ('mrs' in titles1 and 'miss' in titles2) or ('mrs' in titles2 and 'miss' in titles1):
                return False
        
        # Check last name match (common in literature)
        if len(name1_parts) > 0 and len(name2_parts) > 0:
            if name1_parts[-1] == name2_parts[-1]:
                # Same last name, might be the same person or related
                return True
    
    # Check for nickname matches
    if (len(name1) <= 5 and len(name1) >= 2) or (len(name2) <= 5 and len(name2) >= 2):
        # One might be a nickname
        longer = name1 if len(name1) > len(name2) else name2
        shorter = name2 if len(name1) > len(name2) else name1
        
        # Check if the shorter one is at the beginning of the longer one
        if longer.startswith(shorter):
            return True
        
        # Check for common nickname patterns
        common_nicknames = {
            'will': ['william'], 'bill': ['william'], 'liz': ['elizabeth'],
            'beth': ['elizabeth'], 'lizzy': ['elizabeth'], 'tom': ['thomas'],
            'alex': ['alexander'], 'kate': ['katherine', 'catherine'],
            'katie': ['katherine', 'catherine'], 'jen': ['jennifer', 'jenny'],
            'jim': ['james'], 'bob': ['robert'], 'rob': ['robert'],
            'dick': ['richard'], 'rick': ['richard'], 'rich': ['richard'],
            'meg': ['margaret'], 'maggie': ['margaret'], 'peggy': ['margaret']
        }
        
        for nick, formals in common_nicknames.items():
            if shorter == nick and any(formal in longer for formal in formals):
                return True
    
    return False

def extract_titles(name):
    """Extract titles from a name."""
    titles = set()
    title_patterns = ['mr', 'mrs', 'miss', 'dr', 'prof', 'sir', 'lady', 'rev']
    
    name_lower = name.lower()
    for title in title_patterns:
        if title in name_lower.split() or f"{title}." in name_lower:
            titles.add(title)
    
    return titles

def refine_clusters_with_context(potential_clusters, entity_type):
    """
    Refine entity clusters using contextual information like block co-occurrence,
    description similarity, etc.
    """
    refined_clusters = []
    
    for cluster in potential_clusters:
        # For small clusters or unambiguous matches, accept as is
        if len(cluster) <= 2:
            refined_clusters.append(cluster)
            continue
        
        # For larger clusters, split if contextual evidence suggests
        # they are different entities
        sub_clusters = {}
        
        for name, data in cluster.items():
            assigned = False
            
            # Try to assign to an existing sub-cluster
            for sub_id, sub_cluster in sub_clusters.items():
                sample_name = next(iter(sub_cluster.keys()))
                sample_data = sub_cluster[sample_name]
                
                # Check contextual compatibility
                if are_contextually_compatible(name, data, sample_name, sample_data, entity_type):
                    sub_cluster[name] = data
                    assigned = True
                    break
            
            # If not assigned, create a new sub-cluster
            if not assigned:
                sub_clusters[len(sub_clusters)] = {name: data}
        
        # Add all sub-clusters to refined results
        refined_clusters.extend(sub_clusters.values())
    
    return refined_clusters

def are_contextually_compatible(name1, data1, name2, data2, entity_type):
    """
    Determine if two entities are contextually compatible 
    based on descriptions, block patterns, etc.
    """
    # Check description similarity
    desc_compat = evaluate_description_compatibility(data1['descriptions'], data2['descriptions'])
    if desc_compat is False:  # Explicitly incompatible
        return False
    
    # Check block overlap
    blocks1 = set(data1['block_list'])
    blocks2 = set(data2['block_list'])
    
    # If they appear in the same block but with different names, they might be different entities
    # (Unless one is an alternate name of the other)
    common_blocks = blocks1.intersection(blocks2)
    if common_blocks and not is_alternate_relationship(name1, data1, name2, data2):
        # If they're both mentioned by different primary names in the same block,
        # they're probably different entities
        return False
    
    # For characters, check for special cases like "Mrs. X" and "Miss X"
    if entity_type == 'characters':
        title1 = extract_titles(name1)
        title2 = extract_titles(name2)
        
        if title1 and title2 and title1 != title2:
            # Different titles often indicate different people
            # But need to be careful - "Mr. X" and "Dr. X" could be the same person
            conflicting_titles = {
                frozenset(['mrs', 'miss']),  # Mrs. X vs Miss X (different women)
                frozenset(['mr', 'mrs'])      # Mr. X vs Mrs. X (husband and wife)
            }
            
            if any(title1.intersection(pair) and title2.intersection(pair) 
                   and title1.intersection(pair) != title2.intersection(pair) 
                   for pair in conflicting_titles):
                return False
    
    # Default to compatible if no evidence of incompatibility
    return True

def evaluate_description_compatibility(desc_list1, desc_list2):
    """
    Evaluate if two sets of descriptions are compatible.
    Returns: True (compatible), False (incompatible), or None (inconclusive)
    """
    if not desc_list1 or not desc_list2:
        return None  # Inconclusive
    
    # Extract description texts
    desc_texts1 = [desc['description'].lower() for desc in desc_list1]
    desc_texts2 = [desc['description'].lower() for desc in desc_list2]
    
    # Look for contradicting information
    contradiction_pairs = [
        ('young', 'old'),
        ('married', 'unmarried'),
        ('tall', 'short'),
        ('wealthy', 'poor'),
        ('male', 'female'),
        ('father', 'son'),
        ('mother', 'daughter')
    ]
    
    for desc1 in desc_texts1:
        for desc2 in desc_texts2:
            for term1, term2 in contradiction_pairs:
                if term1 in desc1 and term2 in desc2:
                    return False  # Contradiction found
                if term2 in desc1 and term1 in desc2:
                    return False  # Contradiction found
    
    # Check for significant similarity
    similar_count = 0
    for desc1 in desc_texts1:
        for desc2 in desc_texts2:
            # Simple word overlap measure
            words1 = set(desc1.split())
            words2 = set(desc2.split())
            overlap = len(words1.intersection(words2))
            
            if overlap > 3:  # Arbitrary threshold
                similar_count += 1
    
    if similar_count > 0:
        return True  # Descriptions seem compatible
    
    return None  # Inconclusive

def is_alternate_relationship(name1, data1, name2, data2):
    """Check if one name might be an alternate of the other."""
    # Check if name2 appears as an alternate of name1
    for alt in data1['alternate_names']:
        if alt['alternate_name'] == name2:
            return True
    
    # Check if name1 appears as an alternate of name2
    for alt in data2['alternate_names']:
        if alt['alternate_name'] == name1:
            return True
    
    return False

def select_canonical_name(cluster):
    """
    Choose the best canonical name from a cluster of entity names.
    """
    # Scoring system for name quality
    scores = {}
    
    for name in cluster.keys():
        score = 0
        
        # Prefer longer names (likely more informative)
        score += len(name) * 0.1
        
        # Prefer names with titles (for characters)
        if any(title in name.lower() for title in ['mr', 'mrs', 'miss', 'dr', 'prof', 'sir', 'lady']):
            score += 2
        
        # Prefer names with first and last components
        parts = name.split()
        if len(parts) >= 2:
            score += 3
        
        # Prefer names that appear in more blocks (more common)
        score += len(cluster[name]['block_list']) * 0.5
        
        # Store score
        scores[name] = score
    
    # Return the name with the highest score
    return max(scores.items(), key=lambda x: x[1])[0]

def resolve_entity_aliases(prelim_entity_data):
    """
    Group entities that likely refer to the same character/location/organization
    based on name similarity and context.
    """
    resolved_registry = {
        'characters': {},
        'locations': {},
        'organizations': {}
    }
    
    for entity_type in prelim_entity_data:
        # Step 1: Build potential clusters based on name similarity
        potential_clusters = []
        processed_entities = set()
        
        for entity_name, entity_data in prelim_entity_data[entity_type].items():
            if entity_name in processed_entities:
                continue
                
            # Start a new cluster with this entity
            current_cluster = {
                entity_name: entity_data
            }
            processed_entities.add(entity_name)
            
            # Collect all alternate names for this entity
            all_alternates = {alt_data['alternate_name'] for alt_data in entity_data['alternate_names']}
            
            # Find other potential matches
            for other_name, other_data in prelim_entity_data[entity_type].items():
                if other_name in processed_entities:
                    continue
                
                # Collect all alternate names for the other entity
                other_alternates = {alt_data['alternate_name'] for alt_data in other_data['alternate_names']}
                
                # Check various matching criteria
                if is_name_match(entity_name, other_name) or \
                   any(is_name_match(entity_name, alt) for alt in other_alternates) or \
                   any(is_name_match(alt, other_name) for alt in all_alternates) or \
                   any(is_name_match(alt1, alt2) for alt1 in all_alternates for alt2 in other_alternates):
                    
                    current_cluster[other_name] = other_data
                    processed_entities.add(other_name)
            
            potential_clusters.append(current_cluster)
        
        # Step 2: Refine clusters using contextual information
        refined_clusters = refine_clusters_with_context(potential_clusters, entity_type)
        
        # Step 3: Create final resolved entities
        for cluster in refined_clusters:
            # Choose the best canonical name
            canonical_name = select_canonical_name(cluster)
            
            # Merge all information
            merged_entity = {
                'canonical_name': canonical_name,
                'all_alternate_names': set(),
                'all_descriptions': [],
                'block_occurrences': set()
            }
            
            # Collect all information from the cluster
            for name, data in cluster.items():
                # Add primary name blocks
                merged_entity['block_occurrences'].update(data['block_list'])
                
                # Add main name as an alternate (except the canonical one)
                if name != canonical_name:
                    merged_entity['all_alternate_names'].add(name)
                
                # Add all alternate names
                for alt_data in data['alternate_names']:
                    merged_entity['all_alternate_names'].add(alt_data['alternate_name'])
                    merged_entity['block_occurrences'].update(alt_data['block_list'])
                
                # Add all descriptions
                for desc_data in data['descriptions']:
                    merged_entity['all_descriptions'].append({
                        'description': desc_data['description'],
                        'blocks': desc_data['block_list']
                    })
            
            # Store the resolved entity
            resolved_registry[entity_type][canonical_name] = merged_entity
    
    return resolved_registry
