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

# Now you can import directly
from alias_resolution import *


# Import required global modules first
from globals import *
from UtilityFunctions import *
from DSAIParams import * # Imports RunCodeFrom, StateStorageList, DocToAddPath etc.
# Import enums for state management and the list of stages
from enums_constants_and_classes import CodeStages, StateStoragePoints, Code_Stages_List


def extract_entity_names_from_question(question, resolved_registry):
    """Extract entity names mentioned in a question."""
    entities = []
    
    # Combine all entity names (canonical and alternates)
    all_names = {}
    for entity_type in resolved_registry:
        for canonical_name, entity_data in resolved_registry[entity_type].items():
            all_names[canonical_name] = (canonical_name, entity_type)
            for alt_name in entity_data['all_alternate_names']:
                all_names[alt_name] = (canonical_name, entity_type)
    
    # Sort names by length (longer first to avoid substring matches)
    sorted_names = sorted(all_names.keys(), key=len, reverse=True)
    
    # Find names in the question
    question_lower = question.lower()
    for name in sorted_names:
        if name.lower() in question_lower:
            canonical, entity_type = all_names[name]
            entities.append((name, entity_type))
    
    return entities

def is_first_meeting_question(question):
    """Check if the question is about a first meeting."""
    keywords = ['first meet', 'first encounter', 'first interaction', 'met first', 
                'initially meet', 'where did they meet', 'when did they meet']
    
    return any(kw in question.lower() for kw in keywords)

def is_interaction_count_question(question):
    """Check if the question is about counting interactions."""
    keywords = ['how many times', 'number of times', 'frequency', 'often', 
                'count of', 'total meetings', 'interact how many']
    
    return any(kw in question.lower() for kw in keywords)

def is_meeting_details_question(question):
    """Check if the question is about meeting details."""
    keywords = ['where did they meet', 'locations', 'where were', 'present at', 
                'who else', 'others at', 'attend', 'meeting details']
    
    return any(kw in question.lower() for kw in keywords)

def is_relationship_development_question(question):
    """Check if the question is about relationship development."""
    keywords = ['relationship develop', 'how did they', 'evolution of', 
                'change over', 'progression', 'develop over']
    
    return any(kw in question.lower() for kw in keywords)

def generate_natural_answer(context, large_blocks, llm_client):
    """
    Generate natural language answers using the LLM.
    """
    question_type = context['question_type']
    
    if question_type == 'first_meeting':
        char1 = context['char1']
        char2 = context['char2']
        result = context['result']
        
        # If no meeting found
        if isinstance(result, str):
            return result
            
        block_num = result['block_number']
        block_text = large_blocks.get(block_num, {}).get('text', '')
        
        prompt = f"""
        Please answer the following question about the relationship between {char1} and {char2}:
        
        {context['original_question']}
        
        Based on the text, they first met in {result['block_number']}: "{result['context']}"
        The meeting took place at: {', '.join(result['locations']) if result['locations'] else 'unspecified location'}
        
        Please provide a concise answer focusing specifically on where and when they first met.
        """
        
        # Call the LLM with the prompt
        response = llm_client.generate_text(prompt)
        return response
    
    elif question_type == 'interaction_count':
        # Similar implementation...
        pass
    
    # Implement other question types...
    
    return "I'm not sure how to answer that question about the text."