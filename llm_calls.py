import json
import logging
from typing import Any, Dict, Optional
import sys
import os
import openai # Keep for OpenAI client and specific error handling
import google.generativeai as genai # Import Gemini library

# Adjust path to import from the root DSAI_v2_Scripts directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = script_dir
sys.path.insert(0, parent_dir)

# Import necessary components from other modules
try:
    from globals import *
    from UtilityFunctions import *
    # Import specific params needed
    from DSAIParams import *
    from environment_params import GeminiSafetySettings
    from enums_and_constants import *
    # Get OpenAI client - still need this assuming it's initialized elsewhere
    from storage_pipeline.db_connections import client as openai_client
    # config_loader is no longer needed here for client setup
except ImportError as e:
    print(f"Error importing modules in llm_calls.py: {e}")
    raise

# --- Client Configuration REMOVED --- #
# Configuration is now handled in config_loader.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Provider-Specific Call Implementations --- #

def _openai_llm_call(
    system_message: str,
    prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    response_format: Optional[Dict[str, str]]
) -> str:
    """Makes an LLM call using the OpenAI API."""
    final_prompt = initialPromptText + prompt
    try:
        logging.debug(f"Making OpenAI call to model: {LLM_model}")
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

        response = openai_client.chat.completions.create(**params)
        raw_content = response.choices[0].message.content
        if raw_content is None:
             raise ValueError(f"OpenAI response content is None for model {LLM_model}")
        logging.debug("OpenAI call successful.")
        return raw_content
    except openai.RateLimitError as rle:
        logging.warning(f"OpenAI Rate limit hit: {rle}")
        raise
    except Exception as e:
        logging.error(f"OpenAI call failed: {e}")
        raise

def _gemini_llm_call(
    system_message: str,
    prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    response_format: Optional[Dict[str, str]] # Note: Gemini uses prompt instructions for JSON
) -> str:
    """Makes an LLM call using the Google Gemini API."""
    # Combine system message and user prompt for Gemini
    # Prepend initialPromptText (handles debug/training knowledge flags)
    full_prompt = initialPromptText + "\n\n" + system_message + "\n\n" + prompt

    # If JSON output is requested via response_format, ensure prompt reflects this strongly.
    # The call_llm_json_mode wrapper already adds a basic instruction.
    # You might want to add more specific JSON instructions here if needed.
    if response_format and response_format.get("type") == "json_object":
        # Example: Append a stronger instruction if not already present
        # Note: This is basic; robust JSON enforcement with Gemini requires careful prompt engineering.
        if "output a single, valid JSON object" not in full_prompt.lower():
             full_prompt += "\n\nStrictly adhere to the requested JSON format. Output ONLY the JSON object."

    try:
        logging.debug(f"Making Gemini call to model: {LLM_model}")
        # Assumes genai is configured by config_loader
        model = genai.GenerativeModel(LLM_model)

        # Configure generation parameters
        generation_config_params = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            # Gemini uses 'max_output_tokens'
            generation_config_params["max_output_tokens"] = max_tokens

        generation_config = genai.types.GenerationConfig(**generation_config_params)

        # Safety settings - Use imported settings
        safety_settings = GeminiSafetySettings

        response = model.generate_content(
            full_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Accessing the response text
        # Need to check response.parts structure and potential blocks/errors
        if response.parts:
            raw_content = "".join(part.text for part in response.parts)
        elif hasattr(response, 'text'): # Fallback check
             raw_content = response.text
        else:
             # Handle cases where the response might be blocked or empty
             logging.warning(f"Gemini response has no text parts. Blocked? Response: {response}")
             # Check for prompt feedback (blocking reasons)
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                  raise ValueError(f"Gemini call blocked. Reason: {response.prompt_feedback.block_reason}")
             else:
                  raise ValueError(f"Gemini response content is empty or inaccessible. Response: {response}")

        if raw_content is None:
             raise ValueError(f"Gemini response content is None for model {LLM_model}")

        logging.debug("Gemini call successful.")
        # Optional: Clean Gemini's JSON output if it includes markdown backticks
        if response_format and response_format.get("type") == "json_object":
             raw_content = raw_content.strip().removeprefix("```json").removesuffix("```")

        return raw_content

    except Exception as e:
        logging.error(f"Gemini call failed: {e}")
        # Potentially catch specific Gemini exceptions if needed
        raise

# --- Unified LLM Call Wrapper --- #

def llm_call(
    system_message: str,
    prompt: str,
    temperature: float = DefaultLLMTemperature,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, str]] = None
) -> str:
    """
    Core wrapper function to make an LLM call, dispatching to the correct provider.

    Args:
        system_message: The system message content.
        prompt: The user prompt content.
        temperature: The sampling temperature.
        max_tokens: The maximum number of tokens to generate.
        response_format: The desired response format (e.g., for JSON mode).

    Returns:
        The raw string content of the LLM response.

    Raises:
        ValueError: If the AIPlatform is unsupported or response content is None.
        Exception: Propagates API errors or other exceptions.
    """
    logging.debug(f"Dispatching LLM call via platform: {AIPlatform}")

    # Use uppercase comparison for dispatching
    if AIPlatform.upper() == "OPENAI":
        return _openai_llm_call(
            system_message=system_message,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
    elif AIPlatform.upper() == "GEMINI":
        return _gemini_llm_call(
            system_message=system_message,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
    else:
        raise ValueError(f"Unsupported AIPlatform configured: {AIPlatform}")

# --- JSON Mode Wrapper (Unchanged conceptually) --- #

def call_llm_json_mode(
    system_message: str,
    prompt: str,
    temperature: float = DefaultLLMTemperature,
) -> Dict[str, Any]:
    """
    Wrapper function to call the LLM in JSON mode and parse the result.
    Relies on the underlying llm_call dispatcher.
    Includes text cleaning for common encoding artifacts.
    """
    # Add JSON instruction for both providers
    json_system_message = system_message + " Ensure the output is a single, valid JSON object and nothing else."
    # For Gemini, the _gemini_llm_call might add further JSON instructions.

    try:
        raw_content = llm_call(
            system_message=json_system_message,
            prompt=prompt,
            temperature=temperature,
            response_format={"type": "json_object"} # Pass hint to llm_call/provider funcs
        )

        # --- Text Cleaning Step --- #
        cleaned_content = raw_content.replace('â€™', "'") \
                                   .replace('â€œ', '"') \
                                   .replace('â€', '"') \
                                   .replace('â€¦', '...')

        # Parse the cleaned JSON string
        result = json.loads(cleaned_content)
        return result

    except json.JSONDecodeError as json_e:
        # Log the content that failed *after* cleaning attempt
        logging.error(f"JSON decoding failed: {json_e}")
        logging.error(f"Cleaned content that failed to parse: {cleaned_content[:500]}...")
        # Log the original raw content as well
        # Need to ensure raw_content is accessible here, maybe return it from llm_call on error?
        # For now, log only cleaned_content.
        raise
    except Exception as e:
        logging.error(f"call_llm_json_mode failed: {e}")
        raise

# Add other wrapper functions here if needed for specific use cases
# e.g., a function for summarization that calls llm_call with specific params 