#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Environment-specific parameters unlikely to change between runs."""

# Note: Requires google.generativeai to be installed
try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    # Define dummy types if library not installed, allows import without error
    # but will cause runtime error if Gemini is actually used.
    print("Warning: google.generativeai not installed. Gemini functionality will fail.")
    class HarmCategory:
        HARM_CATEGORY_HARASSMENT = None
        HARM_CATEGORY_HATE_SPEECH = None
        HARM_CATEGORY_SEXUALLY_EXPLICIT = None
        HARM_CATEGORY_DANGEROUS_CONTENT = None

    class HarmBlockThreshold:
        BLOCK_MEDIUM_AND_ABOVE = None

# Default safety settings for Gemini
GeminiSafetySettings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Add other environment-specific constants here if needed
# e.g., Base URLs, fixed resource paths, etc. 