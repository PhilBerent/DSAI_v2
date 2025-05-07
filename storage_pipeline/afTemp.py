#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Handles Steps 3 & 4: LLM-based document and chunk analysis."""

import logging
import json
from typing import List, Dict, Any, Tuple, Optional
import sys
import os
# Removed concurrent.futures, tiktoken, time, openai imports as they are handled elsewhere
from enum import Enum
import traceback
import collections
from collections import *

# import prompts # Import the whole module

# Adjust path to import from parent directory AND sibling directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir) # Add parent DSAI_v2_Scripts

from globals import *
from UtilityFunctions import *
from DSAIParams import *
from enums_constants_and_classes import *
from llm_calls import *
from prompts import *
from DSAIUtilities import *
from nameFunctions import *
from alias_resolution import *
from analysis_functions import *
# Import required global modules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Schemas moved to prompts.py

# --- REVISED: Step 3.1 - Map Phase Helper ---

def getIsAnAltNameDict(prelim_entity_data, primary_names_dict):
    """
    Returns a dictionary where the keys are the entity types and the values are dictionaries
    where the keys are the alternate names and the values are dictionaries with keys 'primary_names' and 'indexes'.
    The value of 'primary_names' field is a list of names.
    """
    is_an_alt_name_of_dict = {}
    for entity_type in prelim_entity_data:
        entity_data = prelim_entity_data[entity_type]
        entityDict = primary_names_dict[entity_type]
        thisEntAltNameDict = is_an_alt_name_of_dict[entity_type] = {}
        numEntities = len(entity_data)
        for i in range(numEntities):
            entity = entity_data[i]
            name = entity['name']            
            alternateNameList = entity['alternate_names']
            numAltNames = len(alternateNameList)
            for j in range(numAltNames):
                alt_name = alternateNameList[j].get('alternate_name', '')
                if alt_name not in thisEntAltNameDict:
                    thisEntAltNameDict[alt_name] = {"primary_names": [], "indexes": []}
                thisEntAltNameDict[alt_name]['primary_names'].append(name)
                thisEntAltNameDict[alt_name]['indexes'].append(i)

    return is_an_alt_name_of_dict


def consolidate_entity_information(block_info_list):
    """
    Consolidates entity information from a list of block analyses.

    Returns:
        - formatted_entities_data: a dict where 'characters', 'locations', and 'organizations'
          are lists of entity dicts (each with 'name', 'block_list', 'alternate_names', 'descriptions').
        - primary_name_dict: maps each primary name to its index in the list.
        - alternate_names_dict: maps each alternate name to a list of indices where it appears.
    """

    def consolidate_entity_data(entity_list, entity_data_by_name, block_index):
        """Updates the entity dictionary keyed by primary name."""
        for entity in entity_list:
            name = entity["name"].strip()
            if not name:
                continue

            alt_names = entity.get("alternate_names", [])
            desc = entity.get("description", "")
            if desc:
                desc = desc.strip()

            if name not in entity_data_by_name:
                entity_data_by_name[name] = {
                    "primary_name_blocks": set(),
                    "alt_names_map": collections.defaultdict(set),
                    "descriptions_map": collections.defaultdict(set)
                }

            entity_data_by_name[name]["primary_name_blocks"].add(block_index)

            for alt_name in alt_names:
                cleaned_alt_name = alt_name.strip()
                if cleaned_alt_name:
                    entity_data_by_name[name]["alt_names_map"][cleaned_alt_name].add(block_index)

            if desc:
                entity_data_by_name[name]["descriptions_map"][desc].add(block_index)

    # --- Prepare raw structures ---
    raw_entities_data = {
        "characters": {},
        "locations": {},
        "organizations": {}
    }

    for block_index, block in enumerate(block_info_list):
        key_entities = block.get("key_entities_in_block", {})
        consolidate_entity_data(key_entities.get("characters", []), raw_entities_data["characters"], block_index)
        consolidate_entity_data(key_entities.get("locations", []), raw_entities_data["locations"], block_index)
        consolidate_entity_data(key_entities.get("organizations", []), raw_entities_data["organizations"], block_index)

    # --- Now transform into list-based structures ---
    prelim_entity_data = {
        "characters": [],
        "locations": [],
        "organizations": []
    }
    primary_name_dict = {
        "characters": {},
        "locations": {},
        "organizations": {}
    }

    for category in raw_entities_data:
        name_to_entity_data = raw_entities_data[category]
        for name, data in name_to_entity_data.items():
            # Format alternate names
            formatted_alt_names = []
            for alt_name, block_indices in data["alt_names_map"].items():
                formatted_alt_names.append({
                    "alternate_name": alt_name,
                    "block_list": sorted(list(block_indices))
                })
            formatted_alt_names.sort(key=lambda x: x["alternate_name"])

            # Format descriptions
            formatted_descriptions = []
            for description, block_indices in data["descriptions_map"].items():
                formatted_descriptions.append({
                    "description": description,
                    "block_list": sorted(list(block_indices))
                })
            formatted_descriptions.sort(key=lambda x: x["description"])

            primary_block_list = sorted(list(data["primary_name_blocks"]))

            # --- Create final entity dictionary ---
            entity_entry = {
                "name": name,
                "block_list": primary_block_list,
                "alternate_names": formatted_alt_names,
                "descriptions": formatted_descriptions
            }

            # --- Update main list ---
            list_index = len(prelim_entity_data[category])
            prelim_entity_data[category].append(entity_entry)

            # --- Update primary name dictionary ---
            primary_name_dict[category][name] = list_index

    alt_names_dict = getIsAnAltNameDict(prelim_entity_data, primary_name_dict)
    cmd = CharacterMatchData(prelim_entity_data)
    
    # a = ObjToMindMap(prelim_entity_data)
    # (prelim_entity_data, primary_names_entity_dict, entityData_alt_names_dict, char_match_data, \
    # matchesFound, namesRemoved) = clean_prelim_entity_data_char(prelim_entity_data, primary_names_entity_dict, entityData_alt_names_dict, char_match_data)


    return prelim_entity_data, primary_name_dict, alt_names_dict, cmd

def get_primary_entity_namesOld(prelim_entity_data):
    # prelim_entity_data is a dictionary with keys 'characters', 'locations', and 'organizations' the values of each of these are also dictionaries for which the keys are the names of the entities. Return a dictionary 'prelim_primary_names' where the keys are 'characters', 'locations', and 'organizations' and the values are a list of tuples with the first element being the keys of each of the elements in the corresponding dictionaries in 'prelim_entity_data', and the second element being the length of the block_list coreresponding to the keys in 'prelim_entity_data'. The results should be sorted in descending order of the length of the block_list. 
    i=1
    prelim_primary_names = {}
    for entity_type in prelim_entity_data:
        prelim_primary_names[entity_type] = sorted(
            prelim_entity_data[entity_type].items(),
            key=lambda x: len(x[1]['block_list']),
            reverse=True
        )
        # Convert to list of tuples (name, block_list_length)
        prelim_primary_names[entity_type] = [(name, len(data['block_list'])) for name, data in prelim_primary_names[entity_type]]
        # Sort by block_list length in descending order
        prelim_primary_names[entity_type].sort(key=lambda x: x[1], reverse=True)
    return prelim_primary_names

def get_primary_entity_names(prelim_entity_data, is_alt_name_dict_in):
    prelim_primary_names = {}
    for entity_type in prelim_entity_data:
        entities = prelim_entity_data[entity_type]

        prelim_primary_names[entity_type] = sorted(
            [
                (entity['name'], len(entity['block_list']))
                for entity in entities
            ],
            key=lambda x: x[1],
            reverse=True
        )
    # create a dictionary 'prim_names_dict' where the keys are the names of the entities and the are the index of where that name appears in prelim_primary_names
    primary_names_dict = {}
    for entity_type in prelim_primary_names:
            primary_names_dict[entity_type] = {}
            for index, (name, _) in enumerate(prelim_primary_names[entity_type]):
                primary_names_dict[entity_type][name] = index
    
    
    # alt_names_dict_in is a dictionary where the keys are the entity types and the values are dictionaries where the keys are the alternate names and the values are dictionaries with keys 'primary_names' and 'indexes'. The value of 'primary_names' field is a list of names. Create a dictionary 'alt_names_dict_out' where the keys are the keys for the corresponding type, and the values are lists containing the indexes of the elements of the 'primary_names' field in the prelim_primary_names list. 
    is_an_alt_name_of_dict = {}
    for entity_type in is_alt_name_dict_in:
        alt_name_dict = is_alt_name_dict_in[entity_type]
        is_an_alt_name_of_dict[entity_type] = {}
        ent_prim_names_dict = primary_names_dict[entity_type]
        
        for alt_name, data in alt_name_dict.items():
            is_an_alt_name_of_dict[entity_type][alt_name] = []
            primary_names_list = data['primary_names']
            num_primary_names = len(primary_names_list) 
            for i in range(num_primary_names):
                name = primary_names_list[i]
                index_in_prelim_primary_names = ent_prim_names_dict[name]
                is_an_alt_name_of_dict[entity_type][alt_name].append(index_in_prelim_primary_names)

    has_alt_names_dict = {}
    # create has_alt_names_dict = {}. For every name in the prelim_primary_names list where 
    # (1) alternate_names[] is not empty & (2) at least one of the alternate_names in in primary_name_dict then the key is the name field in prelim_primary_names and the value is a list of the numbers resulting from looking up the alternate_names in primary_names_dict
    for entity_type in prelim_entity_data:
        has_alt_names_dict[entity_type] = {}
        for entity in prelim_entity_data[entity_type]:
            name_dict = primary_names_dict[entity_type]
            alt_name_index_list = []
            name = entity['name']
            alternate_names_list = entity['alternate_names']
            num_alt_names = len(alternate_names_list)
            if num_alt_names == 0:
                continue
            # Check if any of the alternate names are in primary_names_dict
            for i in range(num_alt_names):
                alt_name = alternate_names_list[i]['alternate_name']
                if alt_name in name_dict:
                    index_in_primary_names = name_dict[alt_name]
                    # If it does, check if any of the alternate names are in primary_names_dict
                    alt_name_index_list.append(index_in_primary_names)    
            has_alt_names_dict[entity_type][name] = alt_name_index_list
    
    return prelim_primary_names, primary_names_dict, is_an_alt_name_of_dict, has_alt_names_dict


class CharacterMatchData:
    def __init__(self, primary_entity_data):
        self.full_names_dict = {}
        self.name_no_title_dict = {}
        self.first_names_dict = {}
        self.last_names_dict = {}
        self.first_and_last_names_dict = {}
        self.name_details_dict = {}
        self.name_details_list = []
        self.full_name_list = []

        self.full_name_list = [item['name'] for item in primary_entity_data['characters']]

        num_names = len(self.full_name_list)

        for i in range(num_names):
            name = self.full_name_list[i]
            name_details = NameDetails(name)

            self.name_details_dict[name] = name_details
            self.name_details_list.append(name_details)

            first_name = name_details.first_name
            last_name = name_details.last_name
            name_no_title = name_details.name_no_title
            first_and_last_name = name_details.first_and_last_name

            self.full_names_dict[name] = i
            if first_name:
                self.first_names_dict.setdefault(first_name, []).append(i)
            if last_name:
                self.last_names_dict.setdefault(last_name, []).append(i)
            if name_no_title:
                self.name_no_title_dict.setdefault(name_no_title, []).append(i)
            if first_and_last_name:
                self.first_and_last_names_dict.setdefault(first_and_last_name, []).append(i)
    
    # db delete this function
    def remove_name(self, name: str):
        # Remove the name itself from all direct mappings
        self.full_names_dict.pop(name, None)
        self.name_details_dict.pop(name, None)

        # Retrieve the NameDetails instance to access its components
        name_details = NameDetails(name)

        self.name_no_title_dict.pop(name_details.name_no_title, None)
        self.first_names_dict.pop(name_details.first_name, None)
        self.last_names_dict.pop(name_details.last_name, None)
        self.first_and_last_names_dict.pop(name_details.first_and_last_name, None)

        # Remove from name_details_list and full_name_list
        self.name_details_list = [nd for nd in self.name_details_list if nd.input_name != name]
        self.full_name_list = [n for n in self.full_name_list if n != name]
    # ed
    def as_dict(self):
        return {
            "full_names_dict": self.full_names_dict,
            "name_no_title_dict": self.name_no_title_dict,
            "first_names_dict": self.first_names_dict,
            "last_names_dict": self.last_names_dict,
            "first_and_last_names_dict": self.first_and_last_names_dict,
            "full_name_list": self.full_name_list,
            "name_details_list": self.name_details_list,
            "name_details_dict": self.name_details_dict
        }
# db
# def getPrimaryCharsMatchDict(primary_entity_data):
#     # create a dictionary 'character_name_match_dict' where the keys are the names of the characters and the values are the indexes of the elements in the prelim_primary_names list. The results should be sorted in descending order of the length of the block_list. 
#     character_name_match_dict = {}
#     full_names_dict = {}
#     name_no_title_dict = {}
#     first_names_dict = {}    
#     last_names_dict = {}
#     first_and_last_names_dict = {}
#     nameDetailsDict = {}
#     nameDetailsList = []
#     fullNameList = [name for name, _ in primary_entity_data['characters']]
#     num_names = len(fullNameList)
#     for i in range(num_names):
#         name = fullNameList[i]
#         name_details = NameDetails(name)
#         nameDetailsDict[name] = name_details
#         nameDetailsList.append(name_details)
#         first_name = name_details.first_name
#         last_name = name_details.last_name
#         name_no_title = name_details.name_no_title
#         first_and_last_name = name_details.first_and_last_name
#         full_names_dict[name] = i
#         if first_name:
#             # Check if first name is already in the dictionary
#             if first_name in first_names_dict:
#                 # If it is, append the index to the list
#                 first_names_dict[first_name].append(i)
#             else:
#                 # If it isn't, create a new list with the index
#                 first_names_dict[first_name] = [i]
#         if last_name:
#             # Check if last name is already in the dictionary
#             if last_name in last_names_dict:
#                 # If it is, append the index to the list
#                 last_names_dict[last_name].append(i)
#             else:
#                 # If it isn't, create a new list with the index
#                 last_names_dict[last_name] = [i]
#         if name_no_title:
#             # Check if name without title is already in the dictionary
#             if name_no_title in name_no_title_dict:
#                 # If it is, append the index to the list
#                 name_no_title_dict[name_no_title].append(i)
#             else:
#                 # If it isn't, create a new list with the index
#                 name_no_title_dict[name_no_title] = [i]
#         if first_and_last_name:
#             # Check if name without title is already in the dictionary
#             if first_and_last_name in first_and_last_names_dict:
#                 # If it is, append the index to the list
#                 first_and_last_names_dict[first_and_last_name].append(i)
#             else:
#                 # If it isn't, create a new list with the index
#                 first_and_last_names_dict[first_and_last_name] = [i]
        
#         # Add the name to the character_name_match_dict
#         character_name_match_dict['full_names_dict'] = full_names_dict
#         character_name_match_dict['name_no_title_dict'] = name_no_title_dict
#         character_name_match_dict['first_names_dict'] = first_names_dict
#         character_name_match_dict['last_names_dict'] = last_names_dict
#         character_name_match_dict['first_and_last_names_dict'] = first_and_last_names_dict
#         character_name_match_dict['full_name_list'] = fullNameList
#         character_name_match_dict['name_details_list'] = nameDetailsList
#         character_name_match_dict['name_details_dict'] = nameDetailsDict

#     return character_name_match_dict
# ed
def removeFromCharNameMatchDict(character_name_match_dict, name):
    # remove the name from the character_name_match_dict
    if name in character_name_match_dict['full_names_dict']:
        del character_name_match_dict['full_names_dict'][name]
    if name in character_name_match_dict['name_no_title_dict']:
        del character_name_match_dict['name_no_title_dict'][name]
    if name in character_name_match_dict['first_names_dict']:
        del character_name_match_dict['first_names_dict'][name]
    if name in character_name_match_dict['last_names_dict']:
        del character_name_match_dict['last_names_dict'][name]
    if name in character_name_match_dict['first_and_last_names_dict']:
        del character_name_match_dict['first_and_last_names_dict'][name]
    if name in character_name_match_dict['parsed_names_dict']:
        del character_name_match_dict['parsed_names_dict'][name]
        
    full_names_dict = character_name_match_dict['full_names_dict']
    name_no_title_dict = character_name_match_dict['name_no_title_dict']
    first_names_dict = character_name_match_dict['first_names_dict']
    last_names_dict = character_name_match_dict['last_names_dict']
    first_and_last_names_dict = character_name_match_dict['first_and_last_names_dict']
    parsed_names_dict = character_name_match_dict['parsed_names_dict']

    return character_name_match_dict, full_names_dict, name_no_title_dict, first_names_dict, \
        last_names_dict, first_and_last_names_dict, parsed_names_dict  