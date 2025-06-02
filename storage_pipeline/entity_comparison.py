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
from alias_resolution import *
from llm_calls import *
from entity_data import Entity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def getComparisonPairScores(comparison_pairs: Dict[str, Any], sorted_entity_data):
    maxTokens = MAX_TPM
    maxRequests = MAX_RPM
    start_time = time.time()
    entityDataLists = getEntityDataLists(sorted_entity_data)
    prompt_Lists = getPromptLists(comparison_pairs, entityDataLists)
    end_time = time.time()
    time_taken = end_time - start_time
    a=4

    numCompPairs = len(comparison_pairs)
    entityDataLists = {}
    total_description_tokens = 0
    for entity_type in sorted_entity_data:
        entityDataLists[entity_type] = []
        entityDataInput = sorted_entity_data[entity_type]
        numEntities = len(entityDataInput)
        for i in range(numEntities):
            thisEntityData = entityDataInput[i]
            entity = Entity(thisEntityData, entity_type)
            entityDataLists[entity_type].append(entity)
            total_description_tokens += entity.get_description_tokens()

def getPromptLists(comparison_pairs: Dict[str, Any], entityDataLists) -> Dict[str, Any]:
    prompt_lists = {}
    for entity_type in comparison_pairs:
        prompt_list_thisET = []
        entityDataInput = entityDataLists[entity_type]
        numPairsThisET = len(comparison_pairs[entity_type])
        comparisonPairsThisET = comparison_pairs[entity_type]
        for i in range(numPairsThisET):
            pair1, pair2 = comparisonPairsThisET[i]
            entity1 = entityDataInput[pair1]
            entity2 = entityDataInput[pair2]
            
            prompt = get_prompt(entity1, entity2)
            description_tokens = entity1.description_tokens + entity2.description_tokens + 143
            prompt_and_tokens = (prompt, description_tokens)
            prompt_list_thisET.append(prompt_and_tokens)
        
        prompt_lists[entity_type] = prompt_list_thisET   
    return prompt_lists

def getEntityDataLists(sorted_entity_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Entity]]:
    entity_data_lists = {}
    for entity_type, entity_data_list in sorted_entity_data.items():
        entity_data_lists[entity_type] = [Entity(data, entity_type) for data in entity_data_list]
    return entity_data_lists

def compare_entities(entity1: Entity, entity2: Entity) -> float:
    if entity1.entity_type != entity2.entity_type:
        return 0
    if entity1.entity_type == 'character':
        if names_match(entity1.name, entity2.name) == MatchTest.NO_MATCH:
            return 0
    prompt = get_prompt(entity1, entity2)
    response = retry_function(llm_call, prompt=prompt, numRetries=7, add_initial_prompt=False)
    return float(response.strip())

def get_prompt(entity1: Entity, entity2: Entity) -> str:
    descriptionList1 = entity1.sorted_descriptions
    descriptionList2 = entity2.sorted_descriptions
    alt_nameList1 = entity1.alternate_name_list
    alt_nameList2 = entity2.alternate_name_list
    entity_type = entity1.entity_type  # Assuming both entities are of the same type
    name1 = entity1.name
    name2 = entity2.name
    if entity_type == 'character':
        gender1 = entity1.name_details.gender if entity1.name_details.gender else "unknown"
        gender2 = entity2.name_details.gender if entity2.name_details.gender else "unknown"
        if gender1 == 'unknown':
            gendertext1 = "It was not posible to positively determine the gender of the first entity from the text."
        else:
            gendertext1 = f"The first entity is referred to as a {gender1}."
        if gender2 == 'unknown':
            gendertext2 = "It was not posible to positively determine  gender of the second entity from the text."
        else:
            gendertext2 = f"The second entity is referred to as a {gender2}."
        gendertext = f"{gendertext1} {gendertext2}"
    else:
        gendertext = "" 

    if entity1.num_alternate_names > 0:
        if entity1.num_alternate_names == 1:
            name1_text = f"The first entity is mainly referred to as '{name1}' but has also been referred to as '{entity1.alt_name_string}' at various places in the text."
        else:
            name1_text = f"The first entity is mainly referred to as '{name1}' but has also been referred to by the following alternate names at various places in the text: '{entity1.alt_name_string}'."
    else:
        name1_text = f"The first entity is referred to as '{name1}' in the text."
    
    if entity2.num_alternate_names > 0:
        if entity2.num_alternate_names == 1:
            name2_text = f"The second entity is mainly referred to as '{name2}' but has also been referred to as '{entity2.alt_name_string}' at various places in the text."
        else:
            name2_text = f"The second entity is mainly referred to as '{name2}' but has also been referred to by the following alternate names at various places in the text: '{entity2.alt_name_string}'."
    else:
        name2_text = f"The second entity is referred to as '{name2}' in the text."
    
    prompt = f"""
    The two lists below contain descriptions of entities which have been extracted by summarizing information about the entity from different points in a text. The descriptions in each list are in the order that they appear in the text. 
    The type of the entity is {entity_type}
    {name1_text} {name2_text}
    {gendertext}
    
    List 1:
    {descriptionList1}
    
    List 2:
    {descriptionList2}
    
    Provide a score between 0 and 1 indicating whether you how strongly you think these two lists of descriptions refer to the same entity.
    """
    return prompt.strip()
    
