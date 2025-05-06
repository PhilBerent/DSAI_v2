import logging
import time
import os
import uuid
import sys
from typing import List, Dict, Any, Optional
import traceback

# Adjust path to import from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from globals import *
from UtilityFunctions import *
from DSAIParams import * 
from DSAIUtilities import *
from enums_constants_and_classes import CodeStages, StateStoragePoints, Code_Stages_List
from primary_analysis_stages import *
from itertools import combinations  
from nameFunctions import *

def generate_comparison_pairsOld(prelim_primary_names, primary_name_dict, alt_names_dict):
    
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

def get_alias_comparison_pairsold2(prelim_primary_names, primary_name_dict, is_an_alt_name_of_dict, has_alt_names_dict,
    character_name_match_dict):

    comparison_pairs = {'characters': [], 'locations': [], 'organizations': []}
    comp_pair_names = {'characters': [], 'locations': [], 'organizations': []}
    full_names_dict = character_name_match_dict['full_names_dict']
    name_no_title_dict = character_name_match_dict['name_no_title_dict']
    first_names_dict = character_name_match_dict['first_names_dict']
    last_names_dict = character_name_match_dict['last_names_dict']
    first_and_last_names_dict = character_name_match_dict['first_and_last_names_dict']
    #sb
    debug = True
    compStop1 = "Mr. Darcy"
    compStop2 = "Babe"
    #ed
    
    for entity_type in ['characters', 'locations', 'organizations']:
        checked_elements = set()
        checked_element_names = set()
        added_pairs = set()
        #db
        added_pair_names = set()
        #ed
        entity_list = prelim_primary_names[entity_type]
        name_to_index = primary_name_dict[entity_type]
        # create name_list_this_type for this entity type
        name_list_this_type = [name[0] for name in entity_list]
        type_has_alt_names_dict = has_alt_names_dict.get(entity_type, {})
        type_is_an_alt_name_of_dict = is_an_alt_name_of_dict.get(entity_type, {})
        # Memoize parsed names for characters
        parsed_names = {}
        if entity_type == 'characters':
            for idx, (name, _) in enumerate(entity_list):
                parsed_names[idx] = parse_name(name)

        for i in range(len(entity_list)):
            if i in checked_elements:
                continue
            #db 
            # i=55
            # i=112
            # #ed
            name1 = entity_list[i][0]
            cluster = set([i])
            cluster_names = set([name1])
            to_explore_list = [i]
            to_explore_set = set([i])
            to_explore_name_list = [name1]
            to_explore_name_set = set([name1])

            while to_explore_list:
                current = to_explore_list.pop()
                #db
                curr_name = to_explore_name_list.pop()
                #ed
                matches = set()
                #db
                matches_names = set()
                #ed
                
                curr_name = entity_list[current][0]
                added_pair_names_this_name = set()
                added_pairs_this_name = set()
                #db
                if current ==9:
                    aa=4
                #ed

                # === 1. Has Alt Names Match ===
                alt_names_index_list = type_has_alt_names_dict.get(curr_name, [])
                alt_names_list = []
                for idx in alt_names_index_list:
                    alt_names_list.append(entity_list[idx][0])
                for alt_idx in alt_names_index_list:
                    if alt_idx not in matches and alt_idx != current:
                        matches.add(alt_idx)
                        name_to_add = entity_list[alt_idx][0]                        
                        #db
                        if (name_to_add == compStop1 and curr_name == compStop2) or (curr_name == compStop1 and name_to_add == compStop2):
                            aa=4 
                        #ed
                        matches_names.add(name_to_add)
                        name_details = parsed_names[alt_idx]
                        if (addToCombos(name_details)):
                            cluster.add(alt_idx)
                            #db
                            cluster_names.add(name_to_add)
                            i=5
                            #ed
                            if alt_idx not in to_explore_set:
                                to_explore_set.add(alt_idx)
                                to_explore_list.append(alt_idx)
                                #db
                                to_explore_name_list.append(name_to_add)
                                to_explore_name_set.add(name_to_add)
                                #ed

                # === 2. Is An Alt Name Of Match ===
                if curr_name in type_is_an_alt_name_of_dict:
                    is_an_alt_name_of_indices = type_is_an_alt_name_of_dict[curr_name]
                    is_an_alt_name_of_list = []
                    for idx in is_an_alt_name_of_indices:
                        is_an_alt_name_of_list.append(entity_list[idx][0])
                    for idx in is_an_alt_name_of_indices:
                        if idx not in matches and idx != current:
                            matches.add(idx)
                            #db
                            name_to_add = entity_list[idx][0]                        
                            matches_names.add(name_to_add)
                            #ed
                            #db
                            if (name_to_add == compStop1 and curr_name == compStop2) or (curr_name == compStop1 and name_to_add == compStop2):
                                aa=4 
                            #ed
                            name_details = parsed_names[idx]
                            if (addToCombos(name_details)):
                                cluster.add(idx)
                                #db
                                cluster_names.add(name_to_add)
                                i=5
                                #ed
                                if alt_idx not in to_explore_set:
                                    to_explore_set.add(idx)
                                    to_explore_list.append(idx)
                                    #db
                                    to_explore_name_list.append(name_to_add)
                                    to_explore_name_set.add(name_to_add)
                                    #ed


                # === 3. Character Name Heuristics ===
                if entity_type == 'characters':
                    parsed_name = parsed_names[current]
                    name_no_title = parsed_name['name_no_title']
                    first_name = parsed_name['first_name']
                    last_name = parsed_name['last_name']
                    first_and_last_name = parsed_name['first_and_last_name']
                    title = parsed_name['title']


                    # Rule 1: name1 and name2 are the same if you remove the titles and suffixes from both and both names have a first and last name 
                    if first_name and last_name:
                        name_no_title_match = name_no_title_dict.get(name_no_title, [])
                        name_no_title_match_names = []
                        #db
                        for idx in name_no_title_match:
                            name_no_title_match_names.append(entity_list[idx][0])
                        #ed
                        for idx in name_no_title_match:
                            matched_name_details = parsed_names[idx]
                            current_name_details = parsed_names[current]
                            if matched_name_details['title'] == "" or current_name_details['title'] == "":
                                if idx != current and idx not in matches:
                                    matches.add(idx)
                                    #db
                                    name_to_add = entity_list[idx][0]
                                    matches_names.add(name_to_add)
                                    #ed
                                    #db
                                    if (name_to_add == compStop1 and curr_name == compStop2) or (curr_name == compStop1 and name_to_add == compStop2):
                                        aa=4 
                                    #ed

                            

                    # Rule 2: full first+last name match
                    if first_and_last_name:
                        if first_and_last_name in first_and_last_names_dict:
                            for idx in first_and_last_names_dict[first_and_last_name]:
                                if idx != current and idx not in matches:
                                    matches.add(idx)
                                    #db
                                    name_to_add = entity_list[idx][0]
                                    matches_names.add(name_to_add)
                                    #db
                                    if (name_to_add == compStop1 and curr_name == compStop2) or (curr_name == compStop1 and name_to_add == compStop2):
                                        aa=4 
                                    #ed

                    # Rule 3: full name == first_name or last_name and one of the names has no title
                    # e.g. "Mr. John Doe" and "John" or "John Doe" and "Doe"
                    #db 
                    first_name_match = first_names_dict.get(curr_name, [])
                    first_name_match_names = []
                    for idx in first_name_match:
                        first_name_match_names.append(entity_list[idx][0])
                    #ed
                    
                    for idx in first_name_match:
                        if idx != current and idx not in matches:
                            other_name_parsed = parsed_names[idx]
                            matches.add(idx)
                            #db
                            name_to_add = entity_list[idx][0]
                            matches_names.add(name_to_add)
                            #ed
                            #db
                            if (name_to_add == compStop1 and curr_name == compStop2) or (curr_name == compStop1 and name_to_add == compStop2):
                                aa=4 
                            #ed

                    last_name_match = last_names_dict.get(curr_name, [])
                    #db
                    last_name_match_names = []
                    for idx in last_name_match:
                        last_name_match_names.append(entity_list[idx][0])
                    #ed
                    for idx in last_name_match:
                        if idx != current and idx not in matches:
                            other_name_parsed = parsed_names[idx]
                            matches.add(idx)
                            #db
                            name_to_add = entity_list[idx][0]
                            matches_names.add(name_to_add)
                            #db
                            if (name_to_add == compStop1 and curr_name == compStop2) or (curr_name == compStop1 and name_to_add == compStop2):
                                aa=4 
                            #ed

                            #ed
                #db
                i=3
                #ed
                # create pairs for all name matches
                for idx in matches:
                    idxName = entity_list[idx][0]
                    lowindex = min(current, idx)
                    highindex = max(current, idx)
                    pair = (lowindex, highindex)
                    if pair not in added_pairs:
                        comparison_pairs[entity_type].append(pair)
                        pairName1 = entity_list[lowindex][0]
                        pairName2 = entity_list[highindex][0]
                        pair_names = (pairName1, pairName2)                      
                        comp_pair_names[entity_type].append(pair_names)
                        added_pairs.add(pair)
                        #db
                        if debug:
                            if compStop1 in pair_names and compStop2 in pair_names:
                                j=4
                            print(f"Added pair: {pair} for {entity_type} with names {pair_names}")
                        #ed
                        added_pair_names.add(pair_names)
                        added_pairs_this_name.add(pair)
                        added_pair_names_this_name.add(pair_names)
                    if idx not in checked_elements:
                        if idx not in to_explore_set:
                            to_explore_set.add(idx)
                            to_explore_list.append(idx)
                            #db
                            to_explore_name_list.append(idxName)
                            to_explore_name_set.add(idxName)
                        #ed
                    
            
                checked_elements.add(current)
                checked_element_names.add(curr_name)
                #db
                i=3
                #ed

            # Record all unique pairwise combinations in the cluster
            cluster_list = sorted(cluster)
            for a, b in combinations(cluster_list, 2):
                lowindex = min(a, b)
                highindex = max(a, b)
                pair = (lowindex, highindex)
                if pair not in added_pairs:
                    pairName1 = entity_list[lowindex][0]
                    pairName2 = entity_list[highindex][0]
                    comparison_pairs[entity_type].append(pair)
                    pair_names = (pairName1, pairName2)
                    comp_pair_names[entity_type].append(pair_names)
                    added_pairs.add(pair)
                    added_pair_names.add(pair_names)
                    if debug:
                        if compStop1 in pair_names and compStop2 in pair_names:
                            j=4
                        print(f"Added pair: {pair} for {entity_type} with names {pair_names}")
                    #ed
                    
                    added_pairs_this_name.add(pair)
                    added_pair_names_this_name.add(pair_names)

            
            i=4

    return comparison_pairs, comp_pair_names

def addPairs(entity_list: list, parsed_names: list , current: int, matches: set[int], 
             added_pairs: set[int], added_pair_names: set[str], added_pair_names_this_name: set[str], 
             addCombos = False, debug_name1="", debug_name2="", debug=False):
    for i in matches:
        lowindex = min(current, i)
        highindex = max(current, i)
        pair = (lowindex, highindex)
        if pair not in added_pairs:
            pairName1 = entity_list[lowindex][0]
            pairName2 = entity_list[highindex][0]
            pair_names = (pairName1, pairName2)                      
            added_pairs.add(pair)
            added_pair_names.add(pair_names)
            added_pair_names_this_name.add(pair_names)                        
            #db
            if debug:
                if debug_name1 in pair_names and debug_name2 in pair_names:
                    j=4
                print(f"Added pair: {pair} with names {pair_names}")
            #ed
    if addCombos:
        # Record all unique pairwise combinations in matches
        matches_list = sorted(matches)
        combo_list = []
        for i in matches_list:
            parsed_name = parsed_names[i]
            if (addToCombos(parsed_name)):
                combo_list.append(i)
        for a, b in combinations(combo_list, 2):
            lowindex = min(a, b)
            highindex = max(a, b)
            pair = (lowindex, highindex)
            if pair not in added_pairs:
                pairName1 = entity_list[lowindex][0]
                pairName2 = entity_list[highindex][0]
                pair_names = (pairName1, pairName2)
                added_pairs.add(pair)
                added_pair_names.add(pair_names)
                #db
                if debug:
                    if debug_name1 in pair_names and debug_name2 in pair_names:
                        j=4
                    print(f"Added pair: {pair} with names {pair_names}")
                #ed
    return added_pairs, added_pair_names, added_pair_names_this_name
    
    
def get_comparison_pairs(prelim_primary_names, parsed_char_names, primary_name_dict, 
        is_an_alt_name_of_dict, has_alt_names_dict, character_name_match_dict):

    comparison_pairs = {'characters': [], 'locations': [], 'organizations': []}
    comp_pair_names = {'characters': [], 'locations': [], 'organizations': []}
    full_names_dict = character_name_match_dict['full_names_dict']
    name_no_title_dict = character_name_match_dict['name_no_title_dict']
    first_names_dict = character_name_match_dict['first_names_dict']
    last_names_dict = character_name_match_dict['last_names_dict']
    first_and_last_names_dict = character_name_match_dict['first_and_last_names_dict']
    #sb
    debug = True
    compStop1 = "Mr. Darcy"
    compStop2 = "Babe"
    #ed
    
    for entity_type in ['characters', 'locations', 'organizations']:
        checked_elements = set()
        checked_element_names = set()
        added_pairs = set()
        #db
        added_pair_names = set()
        #ed
        entity_list = prelim_primary_names[entity_type]
        name_to_index = primary_name_dict[entity_type]
        # create name_list_this_type for this entity type
        name_list_this_type = [name[0] for name in entity_list]
        type_has_alt_names_dict = has_alt_names_dict.get(entity_type, {})
        type_is_an_alt_name_of_dict = is_an_alt_name_of_dict.get(entity_type, {})

        for current in range(len(entity_list)):
            if current in checked_elements:
                continue
            #db 
            # i=55
            # i=112
            # #ed

            matches = set()
            #db
            matches_names = set()
            #ed
            curr_name = entity_list[current][0]
            added_pair_names_this_name = set()
            added_pairs_this_name = set()
            #db
            if current ==9:
                aa=4
            #ed

            # === 1. Has Alt Names Match ===
            alt_names_index_list = type_has_alt_names_dict.get(curr_name, [])
            alt_names_list = []
            has_alt_names_matches = []
            has_alt_names_match_names = []
            for idx in alt_names_index_list:
                alt_names_list.append(entity_list[idx][0])
            for alt_idx in alt_names_index_list:
                if alt_idx not in matches and alt_idx != current:
                    has_alt_names_matches.append(alt_idx)
                    name_to_add = entity_list[alt_idx][0]                        
                    has_alt_names_match_names.append(name_to_add)                        
                    #db
                    if (name_to_add == compStop1 and curr_name == compStop2) or (curr_name == compStop1 and name_to_add == compStop2):
                        aa=4 
                    #ed
            
            added_pairs, added_pair_names, added_pair_names_this_name = \
                addPairs(entity_list, parsed_char_names, current, has_alt_names_matches, added_pairs, 
                            added_pair_names, added_pair_names_this_name, addCombos=True, debug_name1=compStop1, debug_name2=compStop2, debug=debug)
            matches.update(has_alt_names_matches)
            matches_names.update(has_alt_names_match_names)
            

            # === 2. Is An Alt Name Of Match ===
            is_alt_names_matches = []
            is_alt_names_match_names = []
            if curr_name in type_is_an_alt_name_of_dict:
                is_an_alt_name_of_indices = type_is_an_alt_name_of_dict[curr_name]
                is_an_alt_name_of_list = []
                for idx in is_an_alt_name_of_indices:
                    is_an_alt_name_of_list.append(entity_list[idx][0])
                for idx in is_an_alt_name_of_indices:
                    if idx not in matches and idx != current:
                        is_alt_names_matches.append(idx)
                        name_to_add = entity_list[idx][0]                        
                        is_alt_names_match_names.append(name_to_add)
                        #db
                        if (name_to_add == compStop1 and curr_name == compStop2) or (curr_name == compStop1 and name_to_add == compStop2):
                            aa=4 
                        #ed
            added_pairs, added_pair_names, added_pair_names_this_name = \
                addPairs(entity_list, parsed_char_names, current, is_alt_names_matches, added_pairs, 
                            added_pair_names, added_pair_names_this_name, addCombos=True, debug_name1=compStop1, debug_name2=compStop2, debug=debug)
            matches.update(is_alt_names_matches)
            matches_names.update(is_alt_names_match_names)
            

            # === 3. Character Name Heuristics ===
            if entity_type == 'characters':
                parsed_name = parsed_char_names[current]
                name_no_title = parsed_name['name_no_title']
                first_name = parsed_name['first_name']
                last_name = parsed_name['last_name']
                first_and_last_name = parsed_name['first_and_last_name']
                title = parsed_name['title']


                # Rule 1: name1 and name2 are the same if you remove the titles and suffixes from both and both names have a first and last name 
                if first_name and last_name:
                    names_to_pair = name_no_title_dict.get(name_no_title, [])
                    names_to_pair_names = []
                    first_and_last_matches = []
                    first_and_last_match_names = []
                    
                    for idx in names_to_pair:
                        names_to_pair_names.append(entity_list[idx][0])
                    #ed
                    for idx in names_to_pair:
                        if idx != current and idx not in matches:
                            first_and_last_matches.append(idx)
                            name_to_add = entity_list[idx][0]
                            first_and_last_match_names.append(name_to_add)
                            #db
                            if (name_to_add == compStop1 and curr_name == compStop2) or (curr_name == compStop1 and name_to_add == compStop2):
                                aa=4 
                            #ed
                    
                    added_pairs, added_pair_names, added_pair_names_this_name = \
                    addPairs(entity_list, parsed_char_names, current, first_and_last_matches, added_pairs, 
                                added_pair_names, added_pair_names_this_name, addCombos=True, debug_name1=compStop1, debug_name2=compStop2, debug=debug)
                    matches.update(first_and_last_matches)
                    matches_names.update(first_and_last_match_names)


                # Rule 3: Name without title is just one word and this matches a first name or a last name of another name and if both have titles the titles are the same
                wordcount = len(name_no_title.split())
                if wordcount == 1:
                    first_name_match = first_names_dict.get(name_no_title, [])                        
                    #db 
                    first_name_match_names = []
                    for idx in first_name_match:
                        first_name_match_names.append(entity_list[idx][0])
                    #ed
                    last_name_match = last_names_dict.get(name_no_title, [])
                    names_to_check = set(first_name_match)
                    names_to_check.update(last_name_match)
                    # remove names that are already in matches or the current name
                    names_to_pair = []
                    names_to_pair_names = []
                    for idx in names_to_check:
                        if idx != current and idx not in matches:
                            if title:
                                other_name_title = parsed_char_names[idx]['title']
                                if other_name_title and other_name_title != title:
                                    continue
                            names_to_pair.append(idx)
                            names_to_pair_names.append(entity_list[idx][0])

                    added_pairs, added_pair_names, added_pair_names_this_name = \
                    addPairs(entity_list, parsed_char_names, current, names_to_pair, added_pairs, 
                            added_pair_names, added_pair_names_this_name, addCombos=True, debug_name1=compStop1, debug_name2=compStop2, debug=debug)
                    matches.update(names_to_pair)
                    matches_names.update(names_to_pair_names)
                    
    return comparison_pairs, comp_pair_names


def combinePrelimCharNames(prelim_enity_data, primary_names_entity_dict, entity_type, 
        name1Details: NameDetails, name2Details: NameDetails, elementsToRemove:dict[str, list[int]]):

    entity_dict = primary_names_entity_dict[entity_type]
    entity_data = prelim_enity_data[entity_type]
    name1 = name1Details.name
    name2 = name2Details.name
    index1 = entity_dict[name1]
    index2 = entity_dict[name2]
    if index1 == index2 or \
        (index1 in elementsToRemove[entity_type] or index2 in elementsToRemove[entity_type]):
        return prelim_enity_data, elementsToRemove
    name1EntityData = entity_data[index1]
    name2EntityData = entity_data[index2]
    numEntries1 = len(name1EntityData['block_list'])
    numEntries2 = len(name2EntityData['block_list'])
    combinedNameEntry = {}
    if numEntries1 > numEntries2:
        nameUsed = name1
        indexUsed = index1
        indexNotUsed = index2
        nameNotUsed = name2
    else:
        nameUsed = name2
        indexUsed = index2
        indexNotUsed = index1
        nameNotUsed = name1
    combinedNameEntry['name'] = nameUsed
    block_list1 = name1EntityData['block_list']
    block_list2 = name2EntityData['block_list']
    newBlockSet = set(block_list1)
    newBlockSet.update(block_list2)
    alternateNamesList1 = name1EntityData['alternate_names']
    alternateNamesList2 = name2EntityData['alternate_names']
    alternateNameDict1 = {}
    alternateNameDict2 = {}
    alt_name_list1 = [x['alternate_name'] for x in alternateNamesList1]
    alt_name_list2 = [x['alternate_name'] for x in alternateNamesList2]

    for alt_name_and_blocks in alternateNamesList1:
        thisAltName = alt_name_and_blocks['alternate_name']
        alternateNameDict1[thisAltName] = alt_name_and_blocks
    for alt_name_and_blocks in alternateNamesList2:
        thisAltName = alt_name_and_blocks['alternate_name']
        alternateNameDict2[thisAltName] = alt_name_and_blocks
    combined_alt_name_set = set(alt_name_list1)
    combined_alt_name_set.update(alt_name_list2)
    combined_alt_name_list = list(combined_alt_name_set)
    new_alt_names = [] 
    for alt_name in combined_alt_name_list:
        new_alt_name_and_blocks = {}
        new_alt_name_and_blocks['alternate_name'] = alt_name
        altNameBlockSet = set()
        if alt_name in alternateNameDict1:
            if alt_name == nameUsed:
                newBlockSet.update(altName1Data['block_list'])
            else:
                altName1Data = alternateNameDict1[alt_name]
                altNameBlockSet.update(altName1Data['block_list'])
        elif alt_name in alternateNameDict2:
            if alt_name == nameUsed:
                newBlockSet.update(altName2Data['block_list'])
            else:
                altName2Data = alternateNameDict2[alt_name]
                altNameBlockSet.update(altName2Data['block_list'])
        
        new_alt_name_and_blocks['block_list'] = list(altNameBlockSet)
        new_alt_names.append(new_alt_name_and_blocks)
    
    combinedNameEntry['alternate_names'] = new_alt_names
    newBlockList = sorted(list(newBlockSet))
    combinedNameEntry['block_list'] = newBlockList

    descriptionList1 = name1EntityData['descriptions']
    descriptionList2 = name2EntityData['descriptions']
    descriptionDict1 = {}
    descriptionDict2 = {}
    for desc in descriptionList1:
        descriptionDict1[desc['description']] = desc
    for desc in descriptionList2:
        descriptionDict2[desc['description']] = desc
    desc_list1 = [x['description'] for x in descriptionList1]
    desc_list2 = [x['description'] for x in descriptionList2]
    new_descriptions = set(desc_list1)
    new_descriptions.update(desc_list2)
    new_descriptionsList = list(new_descriptions)
    new_descriptions = [] 
    for desc in new_descriptionsList:
        new_description = {}
        new_description['description'] = desc
        altNameBlockSet = set()
        if desc in descriptionDict1:
            name1DescData = descriptionDict1[desc]
            altNameBlockSet.update(name1DescData['block_list'])
        elif desc in descriptionDict2:
            name2DescData = descriptionDict2[desc]
            altNameBlockSet.update(name2DescData['block_list'])
        
        new_description['block_list'] = list(altNameBlockSet)
        new_descriptions.append(new_description)
    
    combinedNameEntry['descriptions'] = new_descriptions

    entity_data[indexUsed] = combinedNameEntry
    del entity_dict[nameNotUsed]
    
    return prelim_enity_data, elementsToRemove, primary_names_entity_dict, \
        nameUsed, nameNotUsed, indexNotUsed
    
    
def removePrelimEntDataElements(prelim_entity_data, primary_names_entity_dict, 
                                elementsToRemove:dict[str, list[int]]):    
    new_entity_data = {}
    new_entity_dict = {}
    for entity_type in prelim_entity_data:
        entity_data_this_type = prelim_entity_data[entity_type]
        entity_dict_this_type = primary_names_entity_dict[entity_type]
        new_entity_dict_this_type = {}
        elThisTypeToRemove = elementsToRemove[entity_type]
        if len(elThisTypeToRemove) == 0:
            new_entity_data[entity_type] = entity_data_this_type
            new_entity_dict[entity_type] = entity_dict_this_type
            continue
        new_entity_data_this_type = []
        for i in range(len(entity_data_this_type)):
            if i in elThisTypeToRemove:
                continue
            entity = entity_data_this_type[i]
            new_entity_data_this_type.append(entity)
            entity_name = entity['name']
            new_entity_data_this_type[entity_name] = i
            
        new_entity_data[entity_type] = new_entity_data_this_type
        new_entity_dict[entity_type] = new_entity_dict_this_type
    
    new_entData_alt_names_dict = getIsAnAltNameDict(new_entity_data, new_entity_dict)
    cmd = CharacterMatchData(prelim_entity_data)
    return new_entity_data, new_entity_dict, new_entData_alt_names_dict, cmd

def clean_prelim_entity_data_char(prelim_entity_data, primary_names_entity_dict, 
        entityData_alt_names_dict, cmd: CharacterMatchData):
    
    try:
        char_dict = primary_names_entity_dict['characters']
        char_alt_names_dict = entityData_alt_names_dict['characters']
        char_names_list = cmd.full_name_list
        name_details_list = cmd.name_details_list
        numNames = len(char_names_list)
        cleaned_primary_names = {}

        prelim_names_char_new = {}
        elementsToRemove = {}
        elementsToRemove['characters'] = set()
        elementsToRemove['locations'] = set()
        elementsToRemove['organizations'] = set()
        #db
        matchesFound = []
        namesRemoved = []
        #ed
        
        # go through each element of char_names and if there is an entry with the same  first and last name but where one has a title and the other does not
        for name1Index in range(numNames):
            lastRemovedName = ""
            if name1Index in elementsToRemove['characters']:
                continue
            name_details1 = cmd.name_details_list[name1Index]
            name1 = name_details1.name
            lastName1 = name_details1.last_name
            #  get all names with the same last name
            lastNameMatchIndexes = cmd.last_names_dict.get(lastName1, [])
            num_matches = len(lastNameMatchIndexes)
            # db
            lastNameMatchList = []
            for idx in lastNameMatchIndexes:
                lastNameMatchList.append(name_details_list[idx].name)
            # ed
            if num_matches == 0:
                continue
            # Get a list of all last names from all the indexes of lastNameMatchIndexes in cmd.name_details_list
            for j in range(num_matches):
                name2Index = lastNameMatchIndexes[j]
                if name2Index == name1Index or name2Index in elementsToRemove['characters']:
                    continue
                name_details2 = name_details_list[name2Index]
                name2 = name_details_list[name2Index].name
                namesMatch = names_match(name_details1, name_details2)
                if namesMatch == MatchTest.MATCH:
                    #db
                    matchesFound.append((name1, name2))
                    #ed
                    (prelim_enity_data, elementsToRemove, primary_names_entity_dict, nameUsed, \
                        nameNotUsed, indexNotUsed) = combinePrelimCharNames(prelim_entity_data, 
                            primary_names_entity_dict, 'characters', name_details1, name_details2, elementsToRemove)
                    
                    elementsToRemove['characters'].add(indexNotUsed)
                    # db
                    # cmd.remove_name(nameNotUsed)
                    namesRemoved.append(nameNotUsed)
                    #ed
                    lastRemovedName = nameNotUsed
                    if lastRemovedName == name1:
                        break
            
            if lastRemovedName == name1:
                continue
            
        
            # get all names for which this is an alt name
            if name1 in char_alt_names_dict:
                alt_names = char_alt_names_dict.get(name1, [])
                # alt_names is a list of indexes in char_names get combo list from the list of parsed names for these indexes
                comboIndexes = alt_names['indexes']
                comboNames = alt_names['primary_names']
                numComboNames = len(comboIndexes)
                if numComboNames >= 2:
                    # create combinations for all indexes in comboNames
                    thisRemoved = False
                    for i in range(numComboNames):
                        if thisRemoved:
                            break
                        alt_name1_index = comboIndexes[i]
                        if alt_name1_index == name1Index or alt_name1_index in elementsToRemove['characters']:
                            continue
                        altName1Details = name_details_list[name1Index]
                        altName1 = comboNames[i]
                        for j in range(i+1, numComboNames):
                            alt_name2_index = comboIndexes[j]
                            if alt_name2_index == name1Index or alt_name2_index in elementsToRemove['characters']:
                                continue
                            altName2Details = name_details_list[alt_name2_index]
                            matchTest = names_match(altName1Details, altName2Details)
                            if matchTest == MatchTest.NO_MATCH:
                                elementsToRemove['characters'].add(name1Index)
                                namesRemoved.append(name1)
                                thisRemoved = True
                                break

        new_entity_data, new_entity_dict, new_entData_alt_names_dict, cmd = \
            removePrelimEntDataElements(prelim_entity_data, primary_names_entity_dict, elementsToRemove)
    except Exception as e:
        print(f"Error in clean_prelim_entity_data_char: {e}")
        errorMessage = traceback.format_exc()
        name1Index=2
        # Handle the exception as needed
        # For example, you might want to log the error or re-raise it
        raise e
            
    return new_entity_data, new_entity_dict, new_entData_alt_names_dict, cmd, matchesFound, namesRemoved
        
    def get_alias_comparison_pairs(prelim_primary_names, primary_name_dict, is_an_alt_name_of_dict, has_alt_names_dict,
        character_name_match_dict):

        char_names = prelim_primary_names['characters']
        parsed_char_names = {}
        # for idx, (name, _) in enumerate(char_names):
        #     parsed_char_names[idx] = parse_name(name)

        
        comparison_pairs, comp_pair_names = get_comparison_pairs(prelim_primary_names, parsed_char_names, primary_name_dict, is_an_alt_name_of_dict, has_alt_names_dict,
            character_name_match_dict)

    return comparison_pairs, comp_pair_names


