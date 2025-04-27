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
from storage_pipeline.alias_resolution import *
from storage_pipeline.timeline_and_relationships import *


def answer_relationship_question(question, graph, timeline, resolved_registry, large_blocks, llm_client=None):
    """
    Generate natural language answers to questions about character relationships.
    """
    # Extract entity names from question
    entity_names = extract_entity_names_from_question(question, resolved_registry)
    
    # Convert to canonical names
    canonical_entities = {}
    for name, entity_type in entity_names:
        canonical_name = map_to_canonical_name(name, resolved_registry[entity_type])
        if canonical_name:
            canonical_entities[name] = {
                'canonical_name': canonical_name,
                'type': entity_type
            }
    
    # Handle different question types
    if is_first_meeting_question(question):
        # Get the two main characters
        chars = [data['canonical_name'] for data in canonical_entities.values() 
                if data['type'] == 'characters']
        
        if len(chars) >= 2:
            char1, char2 = chars[0], chars[1]
            meeting_info = answer_first_meeting_question(char1, char2, graph, timeline)
            
            # Generate natural language answer
            context = {
                'question_type': 'first_meeting',
                'char1': char1,
                'char2': char2,
                'result': meeting_info,
                'original_question': question
            }
            
            return generate_natural_answer(context, large_blocks)
    
    elif is_interaction_count_question(question):
        chars = [data['canonical_name'] for data in canonical_entities.values() 
                if data['type'] == 'characters']
        
        if len(chars) >= 2:
            char1, char2 = chars[0], chars[1]
            count = count_interactions(char1, char2, graph)
            
            context = {
                'question_type': 'interaction_count',
                'char1': char1,
                'char2': char2,
                'count': count,
                'original_question': question
            }
            
            return generate_natural_answer(context, large_blocks)
    
    elif is_meeting_details_question(question):
        chars = [data['canonical_name'] for data in canonical_entities.values() 
                if data['type'] == 'characters']
        
        if len(chars) >= 2:
            char1, char2 = chars[0], chars[1]
            meetings = get_meeting_details(char1, char2, graph, timeline)
            
            context = {
                'question_type': 'meeting_details',
                'char1': char1,
                'char2': char2,
                'meetings': meetings,
                'original_question': question
            }
            
            return generate_natural_answer(context, large_blocks)
    
    elif is_relationship_development_question(question):
        chars = [data['canonical_name'] for data in canonical_entities.values() 
                if data['type'] == 'characters']
        
        if len(chars) >= 2:
            char1, char2 = chars[0], chars[1]
            development = get_relationship_development(char1, char2, graph, timeline)
            
            context = {
                'question_type': 'relationship_development',
                'char1': char1,
                'char2': char2,
                'development': development,
                'original_question': question
            }
            
            return generate_natural_answer(context, large_blocks)
    
    # For questions that don't fit specific patterns, use general question answering
    return answer_general_question(question, graph, timeline, resolved_registry, large_blocks)


def generate_natural_answer(context, large_blocks):
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
        
        # Prepare the relevant context text (limited portion to avoid token overload)
        context_text = block_text[:2000] + "..." if len(block_text) > 2000 else block_text
        
        system_message = """
        You are an expert literature analyst. Your task is to provide specific, accurate answers to questions 
        about character relationships in a novel. Base your answer solely on the information provided, not on 
        any prior knowledge of literature. Be concise but thorough, focusing precisely on the question asked.
        """
        
        prompt = f"""
        Question: {context['original_question']}
        
        Based on the text analysis:
        - Character 1: {char1}
        - Character 2: {char2}
        - First meeting occurred in block/chapter number: {result['block_number']}
        - Meeting context: "{result['context']}"
        - Meeting location(s): {', '.join(result['locations']) if result['locations'] else 'unspecified location'}
        
        Relevant text excerpt:
        {context_text}
        
        Please provide a concise answer focusing specifically on where and when these characters first met, 
        using only the information provided above.
        """
        
        # Call the LLM with the prompt
        response = llm_call(system_message=system_message, prompt=prompt, temperature=0.3)
        return response
    
    elif question_type == 'interaction_count':
        char1 = context['char1']
        char2 = context['char2']
        count = context['count']
        
        system_message = """
        You are an expert literature analyst. Your task is to provide specific, accurate answers to questions 
        about character relationships in a novel. Base your answer solely on the information provided, not on 
        any prior knowledge of literature. Be concise and direct.
        """
        
        prompt = f"""
        Question: {context['original_question']}
        
        Based on the text analysis:
        - Character 1: {char1}
        - Character 2: {char2}
        - Number of interactions: {count}
        
        Please provide a concise answer about how many times these characters interact 
        throughout the novel, using only the information provided above.
        """
        
        response = llm_call(system_message=system_message, prompt=prompt, temperature=0.3)
        return response
    
    elif question_type == 'meeting_details':
        char1 = context['char1']
        char2 = context['char2']
        meetings = context['meetings']
        
        if not meetings:
            return f"{char1} and {char2} never meet in the text."
        
        # Format meeting details for the prompt
        meeting_details = []
        for i, meeting in enumerate(meetings):
            meeting_text = (
                f"Meeting {i+1} (Chapter {meeting['block_number']}):\n"
                f"- Title: {meeting['title']}\n"
                f"- Summary: {meeting['summary']}\n"
                f"- Location(s): {', '.join(meeting['locations']) if meeting['locations'] else 'unspecified'}\n"
                f"- Other characters present: {', '.join(meeting['other_characters']) if meeting['other_characters'] else 'none'}"
            )
            meeting_details.append(meeting_text)
        
        system_message = """
        You are an expert literature analyst. Your task is to provide specific, accurate answers to questions 
        about character relationships in a novel. Base your answer solely on the information provided, not on 
        any prior knowledge of literature. Organize your answer clearly.
        """
        
        prompt = f"""
        Question: {context['original_question']}
        
        Based on the text analysis, here are the details of all meetings between {char1} and {char2}:
        
        {"\n\n".join(meeting_details)}
        
        Please provide a comprehensive answer about where and when these characters meet, who else was present,
        and any other relevant details. Focus specifically on answering the original question.
        """
        
        response = llm_call(system_message=system_message, prompt=prompt, temperature=0.3)
        return response
    
    elif question_type == 'relationship_development':
        char1 = context['char1']
        char2 = context['char2']
        development = context['development']
        
        if isinstance(development, str):
            return development
        
        # Format relationship development for the prompt
        development_points = []
        for stage in development:
            point = (
                f"{stage['stage']} (Chapter {stage['block']}):\n"
                f"- Title: {stage['title']}\n"
                f"- Summary: {stage['summary']}\n"
                f"- Location: {stage['location']}"
            )
            development_points.append(point)
        
        system_message = """
        You are an expert literature analyst. Your task is to provide insightful analysis of character 
        relationships in a novel. Base your answer solely on the information provided, not on 
        any prior knowledge of literature. Focus on the development and changes in the relationship over time.
        """
        
        prompt = f"""
        Question: {context['original_question']}
        
        Based on the text analysis, here is the development of the relationship between {char1} and {char2}
        throughout the novel:
        
        {"\n\n".join(development_points)}
        
        Please provide an analysis of how their relationship develops over the course of the novel,
        focusing on any shifts, turning points, or changes in their interactions. Use only the information provided above.
        """
        
        response = llm_call(system_message=system_message, prompt=prompt, temperature=0.3)
        return response
    
    # Default response
    return "I'm not sure how to answer that question about the text."


def answer_general_question(question, graph, timeline, resolved_registry, large_blocks):
    """
    Answer general questions about the text that don't fit into specific relationship patterns.
    """
    # Extract entity names from question
    entity_names = extract_entity_names_from_question(question, resolved_registry)
    
    # Convert to canonical names
    canonical_entities = {}
    for name, entity_type in entity_names:
        canonical_name = map_to_canonical_name(name, resolved_registry[entity_type])
        if canonical_name:
            canonical_entities[name] = {
                'canonical_name': canonical_name,
                'type': entity_type,
                'original_name': name
            }
    
    # If no entities found, return a generic response
    if not canonical_entities:
        return "I couldn't identify specific characters or locations in your question. Please specify which characters, locations, or events you'd like to know about."
    
    # Gather information about the entities
    entity_info = []
    for orig_name, entity_data in canonical_entities.items():
        canonical_name = entity_data['canonical_name']
        entity_type = entity_data['type']
        
        # Get basic info about this entity
        if entity_type == 'characters':
            # For characters, get appearances and relationships
            appearances = sorted(list(resolved_registry[entity_type][canonical_name].get('block_occurrences', [])))
            
            # Find relationships with other characters
            relationships = []
            for node1, node2, data in graph.edges(data=True):
                if node1 == canonical_name or node2 == canonical_name:
                    other_node = node2 if node1 == canonical_name else node1
                    
                    # Only include character relationships
                    if graph.nodes[other_node].get('type') == 'characters':
                        relationships.append({
                            'character': other_node,
                            'interactions': data.get('count', 0),
                            'first_meeting': data.get('first_meeting')
                        })
            
            entity_info.append({
                'original_name': orig_name,
                'canonical_name': canonical_name,
                'type': entity_type,
                'appearances': appearances,
                'relationships': relationships,
                'alternate_names': list(resolved_registry[entity_type][canonical_name].get('all_alternate_names', [])),
                'descriptions': resolved_registry[entity_type][canonical_name].get('all_descriptions', [])
            })
        
        elif entity_type == 'locations':
            # For locations, get appearances and characters who visited
            appearances = sorted(list(resolved_registry[entity_type][canonical_name].get('block_occurrences', [])))
            
            # Find characters who visited
            visitors = []
            for node1, node2, data in graph.edges(data=True):
                if (node1 == canonical_name and graph.nodes[node2].get('type') == 'characters') or \
                   (node2 == canonical_name and graph.nodes[node1].get('type') == 'characters'):
                    visitor = node2 if node1 == canonical_name else node1
                    visitors.append({
                        'character': visitor,
                        'visits': data.get('count', 0),
                        'first_visit': data.get('first_meeting')
                    })
            
            entity_info.append({
                'original_name': orig_name,
                'canonical_name': canonical_name,
                'type': entity_type,
                'appearances': appearances,
                'visitors': visitors,
                'alternate_names': list(resolved_registry[entity_type][canonical_name].get('all_alternate_names', [])),
                'descriptions': resolved_registry[entity_type][canonical_name].get('all_descriptions', [])
            })
        
        # Handle other entity types as needed
    
    # Find relevant blocks where these entities appear
    relevant_blocks = set()
    for entity in entity_info:
        relevant_blocks.update(entity['appearances'])
    
    # Convert to list and sort
    relevant_blocks = sorted(list(relevant_blocks))
    
    # Get text snippets from relevant blocks (limit to avoid token overload)
    text_snippets = []
    max_snippets = 5  # Limit the number of snippets to include
    
    for i, block_num in enumerate(relevant_blocks):
        if i >= max_snippets:
            break
            
        if block_num in large_blocks:
            block_text = large_blocks[block_num].get('text', '')
            # Limit snippet length
            snippet = block_text[:1000] + "..." if len(block_text) > 1000 else block_text
            text_snippets.append({
                'block_number': block_num,
                'text': snippet
            })
    
    # Get timeline context for relevant blocks
    timeline_snippets = []
    for block_num in relevant_blocks[:max_snippets]:
        if block_num in timeline:
            timeline_snippets.append({
                'block_number': block_num,
                'title': timeline[block_num].get('title', ''),
                'summary': timeline[block_num].get('summary', '')
            })
    
    # Prepare the context for the LLM
    system_message = """
    You are an expert literature analyst. Your task is to provide accurate, detailed answers to questions
    about a novel based only on the information provided. Do not rely on any prior knowledge about the novel
    or its characters. Focus on directly answering the question using the evidence provided.
    """
    
    # Format entity information
    entity_sections = []
    for entity in entity_info:
        if entity['type'] == 'characters':
            section = f"Character: {entity['canonical_name']}\n"
            section += f"Also known as: {', '.join(entity['alternate_names'])}\n" if entity['alternate_names'] else ""
            section += f"Appears in chapters: {', '.join(map(str, entity['appearances']))}\n"
            section += "Descriptions:\n"
            for desc in entity['descriptions'][:3]:  # Limit number of descriptions
                section += f"- {desc.get('description', '')}\n"
            section += "Key relationships:\n"
            for rel in sorted(entity['relationships'], key=lambda x: x.get('interactions', 0), reverse=True)[:5]:
                section += f"- With {rel['character']}: {rel['interactions']} interactions, first in chapter {rel['first_meeting']}\n"
            
            entity_sections.append(section)
        
        elif entity['type'] == 'locations':
            section = f"Location: {entity['canonical_name']}\n"
            section += f"Also known as: {', '.join(entity['alternate_names'])}\n" if entity['alternate_names'] else ""
            section += f"Appears in chapters: {', '.join(map(str, entity['appearances']))}\n"
            section += "Descriptions:\n"
            for desc in entity['descriptions'][:3]:  # Limit number of descriptions
                section += f"- {desc.get('description', '')}\n"
            section += "Characters who visited:\n"
            for visitor in sorted(entity['visitors'], key=lambda x: x.get('visits', 0), reverse=True)[:5]:
                section += f"- {visitor['character']}: {visitor['visits']} visits, first in chapter {visitor['first_visit']}\n"
            
            entity_sections.append(section)
    
    # Format timeline information
    timeline_section = "Relevant Timeline:\n"
    for snippet in timeline_snippets:
        timeline_section += f"Chapter {snippet['block_number']}: {snippet['title']}\n"
        timeline_section += f"Summary: {snippet['summary']}\n\n"
    
    # Construct the prompt
    prompt = f"""
    Question: {question}
    
    Here is information about the entities mentioned in your question:
    
    {"\n\n".join(entity_sections)}
    
    {timeline_section}
    
    Text excerpts from relevant chapters:
    {"".join([f"\nChapter {s['block_number']}:\n{s['text']}\n" for s in text_snippets])}
    
    Please answer the question based only on the information provided above. If the information is insufficient
    to fully answer the question, explain what aspects you can address and what additional information would be needed.
    """
    
    # Call the LLM
    response = llm_call(system_message=system_message, prompt=prompt, temperature=0.3)
    return response