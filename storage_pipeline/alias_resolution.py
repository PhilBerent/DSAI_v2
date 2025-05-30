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
from itertools import combinations  
from nameFunctions import *
import globals as g
import traceback

class CharacterMatchData:
    def __init__(self, prelim_entity_data):
        try:
            entityList = prelim_entity_data['characters']
            numEntities = len(entityList)
            full_name_list = [item['name'] for item in entityList]
            gender_list = [item.get('gender', None) for item in entityList]
            self.common_init_code(full_name_list, gender_list)    
        except Exception as e:
            print(f"Error in CharacterMatchData init: {e}")
            errorMessage = traceback.format_exc()
            # Handle the exception as needed
            # For example, you might want to log the error or re-raise it
            raise e
    
    @classmethod
    def from_prelim_primary_names(cls,  prelim_primary_names, prelim_entity_data, char_entity_dict):
        full_name_list = [item[0] for item in prelim_primary_names['characters']]
        numFullNames = len(full_name_list)
        gender_list = []
        for i in range(numFullNames):
            name = full_name_list[i]
            nameEntityIndex = char_entity_dict[name]
            nameData = prelim_entity_data['characters'][nameEntityIndex]
            gender = nameData.get('gender', None)
            gender_list.append(gender)

        instance = cls.__new__(cls)
        instance.common_init_code(full_name_list, gender_list)
        return instance
    
    def common_init_code(self, full_name_list, gender_list=None):
        self.full_name_list = full_name_list
        self.full_names_dict = {}
        self.name_no_title_dict = {}
        self.first_names_dict = {}
        self.last_names_dict = {}
        self.first_and_last_names_dict = {}
        self.name_details_dict = {}
        self.name_details_list = []

        num_names = len(self.full_name_list)

        for i in range(num_names):
            name = self.full_name_list[i]
            name_details = NameDetails(name)
            if gender_list is not None:
                gender = gender_list[i]
                name_details.addGender(gender)

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

    def replace_element(self, index_to_replace, new_name_details: NameDetails):
        # db
        old_name_details = self.name_details_list[index_to_replace]
        old_name = old_name_details.name
        # ed
        self.remove_element_by_index(index_to_replace)
        
        self.name_details_list[index_to_replace] = new_name_details
        new_name = new_name_details.name
        self.full_name_list[index_to_replace] = new_name
        self.full_names_dict[new_name] = index_to_replace
        self.name_details_dict[new_name] = new_name_details
        self.name_details_list[index_to_replace] = new_name_details
        first_name = new_name_details.first_name
        last_name = new_name_details.last_name
        name_no_title = new_name_details.name_no_title
        first_and_last_name = new_name_details.first_and_last_name
        if first_name:
            self.first_names_dict.setdefault(first_name, []).append(index_to_replace)
        if last_name:
            self.last_names_dict.setdefault(last_name, []).append(index_to_replace)
        if name_no_title:
            self.name_no_title_dict.setdefault(name_no_title, []).append(index_to_replace)
        if first_and_last_name:
            self.first_and_last_names_dict.setdefault(first_and_last_name, []).append(index_to_replace)
    
    def remove_element_by_index(self, index_to_remove):
        name_details = self.name_details_list[index_to_remove]
        name = self.full_name_list[index_to_remove]
        first_name = name_details.first_name
        last_name = name_details.last_name
        name_no_title = name_details.name_no_title
        first_and_last_name = name_details.first_and_last_name
        
        del self.full_names_dict[name]
        del self.name_details_dict[name]
        self.full_name_list[index_to_remove] = None
        self.name_details_list[index_to_remove] = None
        if index_to_remove in self.first_names_dict.get(first_name, []):
            self.first_names_dict[first_name].remove(index_to_remove)
        if index_to_remove in self.last_names_dict.get(last_name, []):
            self.last_names_dict[last_name].remove(index_to_remove)
        if index_to_remove in self.name_no_title_dict.get(name_no_title, []):
            self.name_no_title_dict[name_no_title].remove(index_to_remove)
        if index_to_remove in self.first_and_last_names_dict.get(first_and_last_name, []):
            self.first_and_last_names_dict[first_and_last_name].remove(index_to_remove)
        del self.first_names_dict[first_name]
    
    def WriteNameAndGenderToFile(self, file_path = g.tempOutputFile):
        output = ""
        numNames = len(self.full_name_list)
        for i in range(numNames):
            name_details = self.name_details_list[i]
            name = name_details.name
            gender = name_details.gender
            output += f"{name} - {gender}\n"
        WriteToFile(output, file_path)

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

def addPairs(entity_list: list, entity_Type, char_name_details_list: list[NameDetails] , current: int, matches: set[int], 
             added_pairs: set[int], added_pair_names: set[str], added_pair_names_this_name: set[str], 
             addCombos = False, debug_name1="", debug_name2="", debug=False):
    for i in matches:
        if i == current:
            continue
        lowindex = min(current, i)
        highindex = max(current, i)
        pair = (lowindex, highindex)
        if pair not in added_pairs:
            if entity_Type == 'characters':
                nameDetails1 = char_name_details_list[lowindex]
                nameDetails2 = char_name_details_list[highindex]
                if names_match(nameDetails1, nameDetails2) == MatchTest.NO_MATCH:
                    continue 
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
            nameDetails = char_name_details_list[i]
            if (addCharNameToCombos(nameDetails)):
                combo_list.append(i)
        for a, b in combinations(combo_list, 2):
            if a == b:
                continue
            lowindex = min(a, b)
            highindex = max(a, b)
            pair = (lowindex, highindex)
            if pair not in added_pairs:
                if entity_Type == 'characters':
                    nameDetails1 = char_name_details_list[lowindex]
                    nameDetails2 = char_name_details_list[highindex]
                    if names_match(nameDetails1, nameDetails2) == MatchTest.NO_MATCH:
                        continue
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
    
    
def get_comparison_pairs(prelim_primary_names, primary_name_dict, is_an_alt_name_of_dict, 
    has_alt_names_dict, char_match_data: CharacterMatchData):

    comparison_pairs = {'characters': [], 'locations': [], 'organizations': []}
    comp_pair_names = {'characters': [], 'locations': [], 'organizations': []}
    full_names_dict = char_match_data.full_names_dict
    name_no_title_dict = char_match_data.name_no_title_dict
    first_names_dict = char_match_data.first_names_dict
    last_names_dict = char_match_data.last_names_dict
    first_and_last_names_dict = char_match_data.first_and_last_names_dict
    char_name_details_list =  char_match_data.name_details_list
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
        num_elemts = len(entity_list)
        for current in range(num_elemts):
            if current in checked_elements:
                continue
            matches = set()
            matches_names = set()
            curr_name = entity_list[current][0]
            added_pair_names_this_name = set()
            curr_name_details = char_name_details_list[current]
            #db
            if current ==4:
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
                    alt_name_details = char_name_details_list[alt_idx]
                    if names_match(curr_name_details, alt_name_details) != MatchTest.NO_MATCH:
                        has_alt_names_matches.append(alt_idx)
                        name_to_add = entity_list[alt_idx][0]                        
                        has_alt_names_match_names.append(name_to_add)                        
                        #db
                        if (name_to_add == compStop1 and curr_name == compStop2) or (curr_name == compStop1 and name_to_add == compStop2):
                            aa=4 
                        #ed
            
            added_pairs, added_pair_names, added_pair_names_this_name = \
                addPairs(entity_list, entity_type, char_name_details_list, current, has_alt_names_matches, added_pairs, 
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
                addPairs(entity_list, entity_type, char_name_details_list, current, is_alt_names_matches, added_pairs, 
                            added_pair_names, added_pair_names_this_name, addCombos=True, debug_name1=compStop1, debug_name2=compStop2, debug=debug)
            matches.update(is_alt_names_matches)
            matches_names.update(is_alt_names_match_names)
            

            # === 3. Character Name Heuristics ===
            if entity_type == 'characters':
                name_details = char_name_details_list[current]
                name_no_title = name_details.name_no_title
                first_name = name_details.first_name
                last_name = name_details.last_name
                first_and_last_name = name_details.first_and_last_name
                title = name_details.title
                gender = name_details.gender


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
                    addPairs(entity_list, entity_type, char_name_details_list, current, 
                             first_and_last_matches, added_pairs, added_pair_names, added_pair_names_this_name, addCombos=True, debug_name1=compStop1, debug_name2=compStop2, debug=debug)
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
                            other_name_details = char_name_details_list[idx]
                            names_to_pair.append(idx)
                            names_to_pair_names.append(entity_list[idx][0])

                    added_pairs, added_pair_names, added_pair_names_this_name = \
                    addPairs(entity_list, entity_type, char_name_details_list, current, names_to_pair, added_pairs, 
                            added_pair_names, added_pair_names_this_name, addCombos=True, debug_name1=compStop1, debug_name2=compStop2, debug=debug)
                    matches.update(names_to_pair)
                    matches_names.update(names_to_pair_names)
                    # db
                    a=3
                    # ed
                    
        comparison_pairs[entity_type] = list(added_pairs)
        comp_pair_names[entity_type] = list(added_pair_names)
        
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
    _, best = selectBestName(name1Details, name2Details, numEntries1, numEntries2)
    if best == 1:
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
            altName1Data = alternateNameDict1[alt_name]
            if alt_name == nameUsed:
                newBlockSet.update(altName1Data['block_list'])
            else:
                altNameBlockSet.update(altName1Data['block_list'])
        elif alt_name in alternateNameDict2:
            altName2Data = alternateNameDict2[alt_name]
            if alt_name == nameUsed:
                newBlockSet.update(altName2Data['block_list'])
            else:
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
        nameUsed, nameNotUsed, indexNotUsed, indexUsed
    
    
def removePrelimEntDataElements(prelim_entity_data, primary_names_entity_dict, 
                                elementsToRemove:dict[str, list[int]]):    
    new_entity_data = {}
    new_entity_dict = {}
    for entity_type in prelim_entity_data:
        entity_data_this_type = prelim_entity_data[entity_type]
        entity_dict_this_type = primary_names_entity_dict[entity_type]
        elThisTypeToRemove = elementsToRemove[entity_type]
        if len(elThisTypeToRemove) == 0:
            new_entity_data[entity_type] = entity_data_this_type
            new_entity_dict[entity_type] = entity_dict_this_type
            continue
        new_entity_data_this_type = []
        new_entity_dict_this_type = {}
        addIndex = 0
        for i in range(len(entity_data_this_type)):
            if i in elThisTypeToRemove:
                continue
            entity = entity_data_this_type[i]
            new_entity_data_this_type.append(entity)
            entity_name = entity['name']
            new_entity_dict_this_type[entity_name] = addIndex
            addIndex += 1
            
        new_entity_data[entity_type] = new_entity_data_this_type
        new_entity_dict[entity_type] = new_entity_dict_this_type
    
    new_entData_alt_names_dict = getIsAnAltNameDict(new_entity_data, new_entity_dict)
    cmd = CharacterMatchData(new_entity_data)
    return new_entity_data, new_entity_dict, new_entData_alt_names_dict, cmd

def adjust_bad_first_word_entities(prelim_entity_data, primary_names_entity_dict, elementsToRemove, 
                                    alt_name_of_entity_dict, cmd: CharacterMatchData):

        matchesFound = []
        namesRemoved = set()
        for entityType in ['characters', 'locations', 'organizations']:
            entityData = prelim_entity_data[entityType]
            entityDict = primary_names_entity_dict[entityType]
            alt_name_of_this_entity_dict = alt_name_of_entity_dict[entityType]
            numNames = len(entityData)
            for name1Index in range(numNames):
                name1Data = entityData[name1Index]
                if name1Index in elementsToRemove[entityType]:
                    continue
                orig_name = cmd.full_name_list[name1Index]
                firstWord = orig_name.split()[0]
                if firstWord in FirstWordsToRemoveFromNames:
                    name1Details = cmd.name_details_list[name1Index]
                    if firstWord.lower() == "the" and entityType == "characters" and name1Details.isFamily:
                        continue
                    bad_name = orig_name
                    nameNoFirstWord = bad_name.replace(firstWord, "", 1).strip()
                    adjustedNameDetails = NameDetails(nameNoFirstWord)
                    if adjustedNameDetails.gender is None:
                        adjustedNameDetails.addGender(name1Data['gender'])
                    nameCombined = False
                    if nameNoFirstWord in entityDict:
                        name2Index = entityDict[nameNoFirstWord]
                        if name2Index not in elementsToRemove[entityType]:
                            name2Data = entityData[name2Index] 
                            if name1Index != name2Index:
                                (prelim_enity_data, elementsToRemove, primary_names_entity_dict, nameUsed, \
                                    nameNotUsed, indexNotUsed, indexUsed) = combinePrelimCharNames(prelim_entity_data, 
                                    primary_names_entity_dict, entityType, name1Data, name2Data, elementsToRemove)
                                elementsToRemove[entityType].add(indexNotUsed)
                                entityData[indexUsed]['name'] = nameNoFirstWord
                                entityDict[nameNoFirstWord] = indexUsed
                                del entityDict[nameNotUsed]
                                cmd.replace_element(indexUsed, adjustedNameDetails)
                                namesRemoved.add(nameNotUsed)
                                matchesFound.append((bad_name, nameNoFirstWord))
                                nameCombined = True

                    if not nameCombined:
                        entityData[name1Index]['name'] = nameNoFirstWord
                        cmd.replace_element(name1Index, adjustedNameDetails)
                        entityDict[nameNoFirstWord] = name1Index
                        del entityDict[bad_name]
                        indexUsed = name1Index

                    if bad_name in alt_name_of_this_entity_dict:
                        alt_name_data = alt_name_of_this_entity_dict[bad_name]
                        alt_name_primary_names = alt_name_data['primary_names']
                        num_alt_names = len(alt_name_primary_names)
                        for i in range(num_alt_names):
                            alt_name = alt_name_primary_names[i]
                            if alt_name == bad_name:
                                alt_name_primary_names[i] = nameNoFirstWord
                                break
                    
                altNamesThisEntity = entityData[name1Index]['alternate_names']
                numAltNames = len(altNamesThisEntity)
                for i in range(numAltNames):
                    altNameData = altNamesThisEntity[i]
                    altName = altNameData['alternate_name']
                    firstWord = altName.split()[0]
                    if firstWord in FirstWordsToRemoveFromNames:
                        if firstWord.lower() == "the" and entityType == "characters":
                            continue
                        altNameNoFirstWord = altName.replace(firstWord, "", 1).strip()
                        altNamesThisEntity[i]['alternate_name'] = altNameNoFirstWord
                        
        return prelim_entity_data, primary_names_entity_dict, elementsToRemove, alt_name_of_this_entity_dict, cmd, matchesFound, namesRemoved

def clean_prelim_entity_data_char(prelim_entity_data, primary_names_entity_dict, 
        alt_name_of_entity_dict, cmd: CharacterMatchData):
    
    try:
        char_dict = primary_names_entity_dict['characters']
        charEntityList = prelim_entity_data['characters']
        char_alt_names_dict = alt_name_of_entity_dict['characters']
        char_names_list = cmd.full_name_list
        name_details_list = cmd.name_details_list

        elementsToRemove = {}
        elementsToRemove['characters'] = set()
        elementsToRemove['locations'] = set()
        elementsToRemove['organizations'] = set()
        #db
        matchesFound = []
        namesRemoved = set()
        #ed
        # db
        debugstr = ""
        # ed
        (prelim_entity_data, primary_names_entity_dict, elementsToRemove, alt_name_of_entity_dict, 
            cmd, matchesFound, namesRemoved) = \
                adjust_bad_first_word_entities(prelim_entity_data, primary_names_entity_dict, 
                        elementsToRemove, alt_name_of_entity_dict, cmd)
        # go through each element of char_names and if there is an entry with the same  first and last name but where one has a title and the other does not
        num_names = len(char_names_list)
        for name1Index in range(num_names):
            # db
            if name1Index == 41:
                aaz=4
            # ed
            if name1Index in elementsToRemove['characters']:
                continue
            name_details1 = cmd.name_details_list[name1Index]
            name1 = name_details1.name
            # db
            if name1 == "Miss. Dingo":
                aaz=4
            #ed

            charData = charEntityList[name1Index]
            alternate_names = charData['alternate_names']
            hasAltNameList = [x['alternate_name'] for x in alternate_names]
            isAltName = char_alt_names_dict.get(name1, [])
            if isAltName:
                isAltNameList = isAltName['primary_names']
            else:
                isAltNameList = []
            altNameSet = set(hasAltNameList)
            altNameSet.update(isAltNameList)
            allAltNameList = list(altNameSet)
            numAltNames = len(allAltNameList)
            name1Removed = False
            for i in range(numAltNames):
                if name1Removed:
                    break
                altName1 = allAltNameList[i]
                altName1Details = NameDetails(altName1)
                for j in range(i+1, numAltNames):
                    altName2 = allAltNameList[j]
                    altName2Details = NameDetails(altName2)
                    altNameMatch = names_match(altName1Details, altName2Details)
                    if altNameMatch == MatchTest.NO_MATCH:
                        elementsToRemove['characters'].add(name1Index)
                        namesRemoved.add(name1)
                        name1Removed = True
                        break
        
            if name1Removed:
                continue
            
        
            # # get all names for which this is an alt name
            # if name1 in char_alt_names_dict:
            #     alt_names = char_alt_names_dict.get(name1, [])
            #     # alt_names is a list of indexes in char_names get combo list from the list of parsed names for these indexes
            #     comboIndexes = alt_names['indexes']
            #     comboNames = alt_names['primary_names']
            #     numComboNames = len(comboIndexes)
            #     if numComboNames >= 2:
            #         # create combinations for all indexes in comboNames
            #         thisRemoved = False
            #         for i in range(numComboNames):
            #             if thisRemoved:
            #                 break
            #             alt_name1_index = comboIndexes[i]
            #             altName1Details = name_details_list[alt_name1_index]
            #             altName1 = comboNames[i]
            #             if not can_reject_match(altName1Details):
            #                 continue
            #             for j in range(i+1, numComboNames):
            #                 alt_name2_index = comboIndexes[j]
            #                 if alt_name2_index == name1Index or alt_name2_index in elementsToRemove['characters']:
            #                     continue
            #                 altName2Details = name_details_list[alt_name2_index]
            #                 altName2 = comboNames[j]
            #                 matchTest = names_match(altName1Details, altName2Details)
            #                 if matchTest == MatchTest.NO_MATCH:
            #                     elementsToRemove['characters'].add(name1Index)
            #                     namesRemoved.add(name1)
            #                     thisRemoved = True
            #                     break

        for name1Index in range(num_names):
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
            if num_matches != 0:
                # Get a list of all last names from all the indexes of lastNameMatchIndexes in cmd.name_details_list
                for j in range(num_matches):
                    name2Index = lastNameMatchIndexes[j]
                    if name2Index == name1Index or name2Index in elementsToRemove['characters']:
                        continue
                    name_details2 = name_details_list[name2Index]
                    name2 = name_details_list[name2Index].name
                    namesMatch = names_match(name_details1, name_details2)
                    # db
                    debugstr += f"Comparing {name1} with {name2} - Match: {namesMatch}\n"
                    # ed
                    if namesMatch == MatchTest.MATCH:
                        #db
                        matchesFound.append((name1, name2))
                        #ed
                        (prelim_enity_data, elementsToRemove, primary_names_entity_dict, nameUsed, \
                            nameNotUsed, indexNotUsed, indexUsed) = combinePrelimCharNames(prelim_entity_data, 
                                primary_names_entity_dict, 'characters', name_details1, name_details2, elementsToRemove)
                        
                        elementsToRemove['characters'].add(indexNotUsed)
                        # db
                        # cmd.remove_name(nameNotUsed)
                        namesRemoved.add(nameNotUsed)
                        #ed
                        lastRemovedName = nameNotUsed
                        if lastRemovedName == name1:
                            break
    
        new_entity_data, new_entity_dict, new_alt_name_of_entity_dict, cmd = \
            removePrelimEntDataElements(prelim_entity_data, primary_names_entity_dict, elementsToRemove)
        namesRemovedList = list(namesRemoved)        
    except Exception as e:
        print(f"Error in clean_prelim_entity_data_char: {e}")
        errorMessage = traceback.format_exc()
        name1Index=2
        # Handle the exception as needed
        # For example, you might want to log the error or re-raise it
        raise e
    
    
    return new_entity_data, new_entity_dict, new_alt_name_of_entity_dict, cmd, matchesFound, namesRemovedList
        
def get_alias_comparison_pairs(prelim_primary_names, prelim_entity_data, primary_name_dict, 
        is_an_alt_name_of_dict, has_alt_names_dict, char_entity_dict):
    
    char_names = prelim_primary_names['characters']
    char_match_data = CharacterMatchData.from_prelim_primary_names(prelim_primary_names, 
                                                    prelim_entity_data, char_entity_dict)
    parsed_char_names = {}
    # for idx, (name, _) in enumerate(char_names):
    #     parsed_char_names[idx] = parse_name(name)

    
    comparison_pairs, comp_pair_names = get_comparison_pairs(prelim_primary_names, 
        primary_name_dict, is_an_alt_name_of_dict, has_alt_names_dict, char_match_data)

    return comparison_pairs, comp_pair_names


