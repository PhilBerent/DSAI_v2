import json
import logging
from typing import Any, Dict, Optional
import sys
import os
import openai # Keep for potential specific error handling

# Adjust path to import from the root DSAI_v2_Scripts directory
# Assuming this file is in DSAI_v2_Scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = script_dir # Since it's in the root, parent is itself for module resolution?
sys.path.insert(0, parent_dir)

# Import necessary components from other modules
try:
    from globals import *
    from UtilityFunctions import *
    from DSAIParams import *
    from enums_and_constants import *
    from storage_pipeline.db_connections import client # Get the OpenAI client
except ImportError as e:
    print(f"Error importing modules in llm_calls.py: {e}")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def llm_call(
    system_message: str,
    prompt: str,
    temperature: float = DefaultLLMTemperature, # Use default from params
    max_tokens: Optional[int] = None, # Default to None unless specified
    response_format: Optional[Dict[str, str]] = None # e.g., {"type": "json_object"}
) -> str:
    """
    Core function to make an LLM call. Currently supports OpenAI models.

    Args:
        system_message: The system message content.
        prompt: The user prompt content.
        temperature: The sampling temperature.
        max_tokens: The maximum number of tokens to generate.
        response_format: The desired response format (e.g., for JSON mode).

    Returns:
        The raw string content of the LLM response.

    Raises:
        ValueError: If the response content is None.
        Exception: Propagates API errors or other exceptions.
    """
    # Prepend the initial instruction text from enums_and_constants based on flags
    final_prompt = initialPromptText + prompt

    try:
        
        logging.debug(f"Making LLM call to model: {LLM_model}")
        # Prepare parameters, only include non-None ones
        params = {
            "model": LLM_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": final_prompt}
            ]
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if response_format is not None:
            params["response_format"] = response_format

        # Currently assumes client is OpenAI client
        response = client.chat.completions.create(**params)

        raw_content = response.choices[0].message.content

        if raw_content is None:
             raise ValueError(f"LLM response content is None for model {LLM_model}")

        logging.debug(f"LLM call successful. Response length: {len(raw_content)}")
        return raw_content

    except openai.RateLimitError as rle:
        logging.warning(f"Rate limit hit during LLM call: {rle}")
        raise # Re-raise to be handled by caller if needed
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        raise


def call_llm_json_mode(
    system_message: str,
    prompt: str,
    temperature: float = DefaultLLMTemperature,
    # schema: Dict[str, Any] # Schema is not directly used by API call anymore, just for context/validation
) -> Dict[str, Any]:
    """
    Wrapper function to call the LLM in JSON mode and parse the result.
    Includes text cleaning for common encoding artifacts.

    Args:
        system_message: The system message (will be augmented for JSON mode).
        prompt: The user prompt.
        temperature: The sampling temperature.
        # schema: The expected JSON schema (currently unused, for potential validation).

    Returns:
        The parsed JSON object as a dictionary.

    Raises:
        ValueError: If the response content is None or JSON parsing fails.
        Exception: Propagates API errors or other exceptions.
    """
    # Modify system message slightly for JSON mode if desired
    json_system_message = system_message + " Ensure the output is a single, valid JSON object and nothing else."

    try:
        raw_content = llm_call(
            system_message=json_system_message,
            prompt=prompt,
            temperature=temperature,
            response_format={"type": "json_object"} # Request JSON mode
        )

        # --- Text Cleaning Step (moved from analysis.py) --- #
        cleaned_content = raw_content.replace('â€™', "'") \
                                   .replace('â€œ', '"') \
                                   .replace('â€�', '"') \
                                   .replace('â€¦', '...')
        # Add more replacements here if needed

        # Parse the cleaned JSON string
        result = json.loads(cleaned_content)

        # TODO: Add validation against a schema if provided/needed
        return result

    except json.JSONDecodeError as json_e:
        logging.error(f"JSON decoding failed after cleaning: {json_e}")
        logging.error(f"Original content that failed: {raw_content[:500]}...")
        raise # Re-raise the JSON error
    except Exception as e:
        # Catch errors from llm_call or cleaning/parsing
        logging.error(f"call_llm_json_mode failed: {e}")
        raise

# Add other wrapper functions here if needed for specific use cases
# e.g., a function for summarization that calls llm_call with specific params 