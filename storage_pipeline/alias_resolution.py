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

from globals import *
from UtilityFunctions import *
from DSAIParams import * # Imports RunCodeFrom, StateStorageList, DocToAddPath etc.
# Import enums for state management and the list of stages
from enums_constants_and_classes import CodeStages, StateStoragePoints, Code_Stages_List
from primary_analysis_stages import *

def generate_comparison_pairs(prelim_primary_names, primary_name_dict, alt_names_dict):
    
    # Prefixes to ignore when matching
    PREFIXES = {"Mr", "Mrs", "Miss", "Ms", "Dr", "Sir", "Lady", "Captain", "Colonel", "General", "Reverend"}

    def strip_prefix(name):
        """Remove prefix if it exists."""
        parts = name.split()
        if parts and parts[0] in PREFIXES:
            return " ".join(parts[1:])
        return name

    def names_match(name1, name2, entity_type):
        """
        Determine if two names potentially refer to the same entity based on rules.
        """
        if name1 == name2:
            return True

        if entity_type == 'characters':
            stripped1 = strip_prefix(name1)
            stripped2 = strip_prefix(name2)

            if stripped1 == stripped2:
                return True

            parts1 = stripped1.split()
            parts2 = stripped2.split()

            if len(parts1) > 1 and len(parts2) == 1:
                if parts2[0] in parts1:
                    return True
            elif len(parts2) > 1 and len(parts1) == 1:
                if parts1[0] in parts2:
                    return True

        return False

    comparison_pairs_set = set()
    indices_already_matched = set()

    # Main loop over entity types (characters, locations, organizations)
    for entity_type in ['characters', 'locations', 'organizations']:
        entity_list = prelim_primary_names[entity_type]
        num_entities = len(entity_list)

        for i in range(num_entities):
            if i in indices_already_matched:
                continue

            primary_name_i = entity_list[i][0]
            to_explore = [i]
            cluster_indices = set([i])

            while to_explore:
                current = to_explore.pop()
                current_name = entity_list[current][0]

                # Explore all indices > i (our starting point)
                for j in range(i+1, num_entities):
                    if j in cluster_indices:
                        continue  # Already added

                    candidate_name = entity_list[j][0]

                    # Check via alt_names_dict first
                    matches_via_alt = False
                    if candidate_name in alt_names_dict[entity_type]:
                        if current in alt_names_dict[entity_type][candidate_name]:
                            matches_via_alt = True
                    if current_name in alt_names_dict[entity_type]:
                        if j in alt_names_dict[entity_type][current_name]:
                            matches_via_alt = True

                    # Or direct name match
                    matches_via_name = names_match(current_name, candidate_name, entity_type)

                    if matches_via_alt or matches_via_name:
                        cluster_indices.add(j)
                        to_explore.append(j)

            # Build all pairwise combinations inside the cluster
            cluster_list = sorted(cluster_indices)
            for idx1 in range(len(cluster_list)):
                for idx2 in range(idx1 + 1, len(cluster_list)):
                    pair = (cluster_list[idx1], cluster_list[idx2])
                    comparison_pairs_set.add(pair)

            # Mark all cluster indices as matched
            indices_already_matched.update(cluster_indices)

    # Final output list
    comparison_pairs_list = sorted(list(comparison_pairs_set))
    return comparison_pairs_list

# === Example Usage ===
# comparison_pairs_list = generate_comparison_pairs(prelim_primary_names, primary_name_dict, alt_names_dict)