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
from nameFunctions import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Entity:
    def __init__(self, data: dict[str, Any], entity_type: str):
        self.entity_type = entity_type
        self.name = data.get('name', '')
        self.block_list = data.get('block_list', [])
        self.num_blocks = len(self.block_list)
        self.alternate_names = data.get('alternate_names', [])
        self.num_alternate_names = len(self.alternate_names)
        self.descriptions = data.get('descriptions', [])
        self.num_descriptions = len(self.descriptions)
        self._internal_sorted_descriptions = None 
        self._internal_sorted_block_list = None
        self._description_tokens_internal = -1
        self._internal_alt_names_list = "xx"
        self._alt_name_tokens_internal = -1
        self.gender = None
        self.name_details = None
        self._alt_name_string_internal = "xx"

        # self.internal_sorted_descriptions is self._internal_sorted_descriptions if it exists, otherwise it calls get_sorted_descriptions()
        
        if entity_type == 'character':
            self.gender = data.get('gender', None)
            self.name_details = NameDetails(self.name)
            if self.gender:
                self.name_details.addGender(self.gender)
            
    def get_sorted_descriptions(self):
        # Sort the list of dicts by the first element in 'block_list' (or float('inf') if empty)
        sorted_items = sorted(
            self.descriptions,
            key=lambda x: x['block_list'][0] if x['block_list'] else float('inf')
        )

        # Extract the 'description' and 'block_list' from sorted_items
        sorted_descriptions = [item['description'] for item in sorted_items]
        sorted_block_list = [item['block_list'][0] for item in sorted_items]

        # Cache the sorted results
        self._internal_sorted_descriptions = sorted_descriptions
        self._internal_sorted_block_list = sorted_block_list

    
    # a function get_description_tokens that returns the number of tokens in the descriptions
    def get_description_tokens(self):
        # Assuming descriptions is a list of strings
        if len(self.sorted_descriptions) == 0:
            self._description_tokens_internal =  0
        else:
            self._description_tokens_internal = sum(len(encoding.encode(desc)) for desc in self.sorted_descriptions)

    def get_alt_names_tokens(self):
        # Assuming descriptions is a list of strings
        alt_name_list = self.alternate_name_list
        if len(self.alternate_name_list) == 0:
            self._internal_alt_names_list =  0
        else:
            self._internal_alt_names_list = sum(len(encoding.encode(alt_name)) for alt_name in alt_name_list)

    def get_alternate_names_list(self):
        alt_name_list = [item['alternate_name'] for item in self.alternate_names]
        self._internal_alt_names_list = alt_name_list
    
    def get_alternate_name_string(self):
        if len(self.alternate_name_list) == 0:
            self._alt_name_string_internal = ""
        elif len(self.alternate_name_list) == 1:
            self._alt_name_string_internal = self.alternate_name_list[0]
        else:
            self._alt_name_string_internal = ', '.join(f"'{item}'" for item in self.alternate_name_list)

    @property
    def alternate_name_list(self):
        if self._internal_alt_names_list == "xx":
            self.get_alternate_names_list()
        return self._internal_alt_names_list
    
    @property
    def alt_name_string(self):
        if self._alt_name_string_internal == "xx":
            self.get_alternate_name_string()
        return self._alt_name_string_internal

    @property
    def has_alternate_names(self):
        return len(self.alternate_name_list) > 0

    @property
    def sorted_descriptions(self):
        if self._internal_sorted_descriptions is None:
            self.get_sorted_descriptions()
        return self._internal_sorted_descriptions

    @property
    def sortred_block_list(self):
        if self._internal_sorted_descriptions is None:
            self.get_sorted_descriptions()
        return self._internal_sorted_block_list       
    
    @property
    def description_tokens(self):
        if self._description_tokens_internal == -1:
            self.get_description_tokens()
        return self._description_tokens_internal       
    
    @property
    def alt_name_tokens(self):
        if self._alt_name_tokens_internal == -1:
            self.get_alt_names_tokens()
        return self._alt_name_tokens_internal       