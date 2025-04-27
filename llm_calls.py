import json
import logging
from typing import Any, Dict, Optional, List, Callable
import sys
import os
import openai # Keep for OpenAI client and specific error handling
import google.generativeai as genai # Import Gemini library
import concurrent.futures
import time
import traceback

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
    from enums_constants_and_classes import *
    # Get OpenAI client - still need this assuming it's initialized elsewhere
    from storage_pipeline.db_connections import openai_client as openai_client
    from DSAIUtilities import *
    # config_loader is no longer needed here for client setup
except ImportError as e:
    print(f"Error importing modules in llm_calls.py: {e}")
    raise

# --- Client Configuration REMOVED --- #
# Configuration is now handled in config_loader.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Filter specific thread warning --- #
class ThreadWarningFilter(logging.Filter):
    def filter(self, record):
        # Check if the message matches the specific warning
        return "thread._ident is None in _get_related_thread!" not in record.getMessage()

# Apply the filter to the handlers created by basicConfig
try:
    if logging.root.handlers:
        for handler in logging.root.handlers:
            # Check if the filter is already added to avoid duplicates if script runs multiple times
            if not any(isinstance(f, ThreadWarningFilter) for f in handler.filters):
                handler.addFilter(ThreadWarningFilter())
                logging.debug("Applied ThreadWarningFilter to logging handler.")
            else:
                logging.debug("ThreadWarningFilter already present on handler.")
    else:
        # This case should ideally not happen after basicConfig if setup was successful
        logging.warning("Could not apply ThreadWarningFilter: No handlers found on root logger after basicConfig.")
except Exception as filter_err:
    logging.error(f"Failed to add ThreadWarningFilter: {filter_err}", exc_info=True)
# --- End Filter --- #

# --- Provider-Specific Call Implementations --- #

def _openai_llm_call(
    system_message: str,
    prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    response_format: Optional[Dict[str, str]]
) -> str:
    """Makes an LLM call using the OpenAI API."""
    try:
        logging.debug(f"Making OpenAI call to model: {LLM_model}")
        params = {
            "model": LLM_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if response_format is not None:
            params["response_format"] = response_format

        response = openai_client.chat.completions.create(**params)
        raw_content = response.choices[0].message.content
        raw_content = cleanText(raw_content) # Clean the response if needed        
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
    response_format: Optional[Dict[str, str]]
) -> str:
    """Makes an LLM call using the Google Gemini API."""
    try:
        logging.debug(f"Making Gemini call to model: gemini-2.0-flash")
        
        # Initialize the Gemini model with system instruction
        model = genai.GenerativeModel(
            LLM_model,
            system_instruction=system_message
        )
        
        # Set up generation config
        generation_config = {
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
            
        # Generate content
        response = model.generate_content(
            prompt,
            generation_config=generation_config
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
        raw_content = cleanText(raw_content) # Clean the response if needed
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
    response_format: Optional[Dict[str, str]] = None,
    add_initial_prompt: bool = True # Flag to control initial prompt addition
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
    prompt = initialPromptText + prompt  # Prepend initialPromptText to the user prompt
    # Use uppercase comparison for dispatching
    errorCount = 1
    executionError = False
    success = False
    errorMessage = ""
    result = None
    while errorCount <= 3:    
        try:
            success = True
            if AIPlatform.upper() == "OPENAI":
                result = _openai_llm_call(
                    system_message=system_message,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format
                )
            elif AIPlatform.upper() == "GEMINI":
                result = _gemini_llm_call(
                    system_message=system_message,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format
                )
            else:
                raise ValueError(f"Unsupported AIPlatform configured: {AIPlatform}")
            break
        except Exception as e:
            executionError = True
            success = False
            errorCount += 1
            errorMessage = traceback.format_exc()
            time.sleep(1)
    
    if executionError: 
        if success:
             logging.info(f"LLM call succeded after {errorCount} retries. Last error message:\n{errorMessage}")
        else:
             raise Exception(f"LLM call failed after {errorCount} retries:\n{errorMessage}")
    
    return result
# --- JSON Mode Wrapper (Unchanged conceptually) --- #

def call_llm_json_mode(system_message: str, prompt: str, 
        temperature: float = DefaultLLMTemperature, add_initial_prompt=True) -> Dict[str, Any]:
    """
    Wrapper function to call the LLM in JSON mode and parse the result.
    Relies on the underlying llm_call dispatcher.
    Includes text cleaning for common encoding artifacts.
    """
    # Add JSON instruction for both providers
    json_system_message = system_message + " Ensure the output is a single, valid JSON object and nothing else."
    # For Gemini, the _gemini_llm_call might add further JSON instructions.
    result = None
    result_string = None
    try:
        result_string = llm_call(
            system_message=json_system_message,
            prompt=prompt,
            temperature=temperature,
            response_format={"type": "json_object"}, # Pass hint to llm_call/provider funcs
            add_initial_prompt=add_initial_prompt
        )
        result = json.loads(result_string)
        return result

    except json.JSONDecodeError as json_e:
        # Log the content that failed *after* cleaning attempt
        logging.error(f"JSON decoding failed: {json_e}")
        logging.error(f"Cleaned content that failed to parse: {prompt[:500]}...")
        raise
    except Exception as e:
        logging.error(f"call_llm_json_mode failed: {e}")
        raise


# Attempt to import Google API core exceptions for Gemini rate limits
try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    google_exceptions = None
    logging.warning("google-api-core library not found. Gemini rate limit handling might be incomplete.")

def calc_num_instances(estimated_total_tokens_per_call: Optional[float]) -> int:
    """Calculates the optimal number of parallel instances based on token estimates and rate limits."""
    # Import necessary params (consider passing them if this func moves elsewhere)
    
    if estimated_total_tokens_per_call is None or estimated_total_tokens_per_call <= 0:
        dynamic_max_workers = MAX_WORKERS_FALLBACK
        logging.warning(f"Invalid token estimate ({estimated_total_tokens_per_call}). Using fallback max_workers: {dynamic_max_workers}")
    else:
        # Calculate limits based on tokens per minute (TPM)
        calls_per_minute_tpm = MAX_TPM / estimated_total_tokens_per_call
        # Calculate limits based on requests per minute (RPM)
        # Divide RPM by a factor representing how many sequential requests a worker might make per minute
        # (e.g., if WORKER_RATE_LIMIT_DIVISOR is 6, assumes roughly 10-second tasks)
        concurrent_limit_rpm = MAX_RPM / WORKER_RATE_LIMIT_DIVISOR
        # Take the minimum of the two limits
        calculated_max_workers = min(calls_per_minute_tpm, concurrent_limit_rpm)
        # Apply safety factor and ensure at least 1 worker
        dynamic_max_workers = max(1, int(calculated_max_workers * WORKER_SAFETY_FACTOR))
        logging.info(f"Calculated dynamic max_workers: {dynamic_max_workers} "
                     f"(Based on TPM={MAX_TPM}, RPM={MAX_RPM}, "
                     f"EstTotalTokens={estimated_total_tokens_per_call:.0f}, Factor={WORKER_SAFETY_FACTOR}, "
                     f"RPM Divisor={WORKER_RATE_LIMIT_DIVISOR})")

    # Ensure workers don't exceed a practical maximum if needed (e.g., MAX_WORKERS_FALLBACK)
    # dynamic_max_workers = min(dynamic_max_workers, MAX_WORKERS_FALLBACK) # Optional upper cap

    return dynamic_max_workers

def parallel_llm_calls(
    function_to_run: Callable[[Dict[str, Any], int], Optional[Any]], # Expects func(item, index)
    num_instances: int,
    input_data_list: List[Dict[str, Any]],
    platform: str, # To know which rate limit errors to catch ("OPENAI" or "GEMINI")
    rate_limit_sleep: int,
    additional_data: Optional[Any] = None # Placeholder for any additional data needed by the function
) -> List[Optional[Any]]:
    """Runs a given function in parallel for a list of inputs, handling rate limits."""
    results = [None] * len(input_data_list) # Initialize results list with None
    num_workers = min(num_instances, len(input_data_list)) # Cannot have more workers than tasks

    if num_workers <= 0:
        logging.warning("No instances requested or no data to process in parallel_llm_calls.")
        return results

    logging.info(f"Processing {len(input_data_list)} items in parallel with {num_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map futures to the original index to place results correctly
        future_to_index = {
            executor.submit(function_to_run, item, index, additional_data): index
            for index, item in enumerate(input_data_list)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            original_index = future_to_index[future]
            item_ref = input_data_list[original_index].get('ref', f'Index {original_index}') # Get ref for logging if available
            try:
                result = future.result() # Get result from the future
                results[original_index] = result # Place result in the correct spot
            except openai.RateLimitError as rle:
                if platform.upper() == "OPENAI":
                    logging.warning(f"OpenAI Rate limit hit processing item {item_ref}. Sleeping for {rate_limit_sleep}s. Error: {rle}")
                    # TODO: Implement retry logic here instead of just sleeping once
                    # For now, just log and the result remains None
                    time.sleep(rate_limit_sleep)
                else:
                    logging.error(f"Caught OpenAI RateLimitError but platform is {platform}. Item {item_ref} processing failed. Error: {rle}")
            except google_exceptions.ResourceExhausted as rle:
                 if platform.upper() == "GEMINI":
                     logging.warning(f"Gemini Rate limit (ResourceExhausted) hit processing item {item_ref}. Sleeping for {rate_limit_sleep}s. Error: {rle}")
                     # TODO: Implement retry logic here
                     time.sleep(rate_limit_sleep)
                 else:
                     logging.error(f"Caught Gemini ResourceExhausted but platform is {platform}. Item {item_ref} processing failed. Error: {rle}")
            except Exception as exc:
                logging.error(f"Item processing task for {item_ref} generated an exception: {exc}")
                # Result for this index remains None

    successful_count = sum(1 for r in results if r is not None)
    total_count = len(input_data_list)
    failed_count = total_count - successful_count
    logging.info(f"Parallel processing complete. Successfully processed {successful_count}/{total_count} items. ({failed_count} failures/skips)")

    return results

# Add other wrapper functions here if needed for specific use cases
# e.g., a function for summarization that calls llm_call with specific params 