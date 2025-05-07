import spacy
import re
from typing import List, Dict, Tuple, Any
from globals import *
from UtilityFunctions import *
from DSAIParams import *
from DSAIUtilities import *

import spacy
import re
from typing import List, Dict, Tuple, Any
import os
import sys

# Get the directory where the script is
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to storage_pipeline inside DSAI_v2_Scripts
storage_pipeline_path = os.path.join(script_dir, 'storage_pipeline')

# Add to sys.path
sys.path.insert(0, storage_pipeline_path)

# Now import
from storage_pipeline.alias_resolution import *



import openai

# Replace with your actual OpenAI API key
openai.api_key = OPENAI_API_KEY

# Path to your .wav file
# audio_file_path = r"C:\Users\Phil\Documents\DataSphere AI\DataSphere AI Not Code\Misc\2024-09-07-12-19-25.WAV"
audio_file_path = r"C:\Users\Phil\Documents\DataSphere AI\DataSphere AI Not Code\Misc\Recording.wav"
output =  r"C:\Users\Phil\Documents\DataSphere AI\DataSphere AI Not Code\Misc\wavFile.txt"

# Open the audio file in binary mode
# import openai

# # Replace this with your actual API key


# # Open and send the file
# with open(audio_file_path, "rb") as audio_file:
#     print("Uploading and transcribing... please wait.")
#     response = openai.Audio.transcribe(
#         model="whisper-1",
#         file=audio_file
#     )

# # Output the transcription
# print("\n--- Transcription ---\n")
    
# WriteToFile(response["text"], output)
# b=3
import os
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud.aiplatform_v1beta1.types import content as gapic_content

def transcribe_audio_simple(wav_file_path: str, project_id: str, region: str, model_name: str) -> str | None:
    load_dotenv()
    if not project_id or not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("Error: GCLOUD_PROJECT or GOOGLE_APPLICATION_CREDENTIALS not set.")
        return None

    aiplatform.init(project=project_id, location=region)
    model = aiplatform.GenerativeModel(model_name)

    with open(wav_file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    audio_part = gapic_content.Part(
        inline_data=gapic_content.Blob(mime_type="audio/wav", data=audio_bytes)
    )
    prompt_part = gapic_content.Part(text="Transcribe this audio.")
    
    request_contents = [prompt_part, audio_part]
    
    try:
        response = model.generate_content(contents=request_contents)
        if response.candidates and response.candidates[0].content.parts:
            return "".join(part.text for part in response.candidates[0].content.parts if part.text)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
    return None

# --- User-defined variables ---
wav_input = r"C:\path\to\your\audio_file.wav" # <--- REPLACE WITH ACTUAL PATH TO YOUR WAV FILE
txt_output = None

# --- Configuration (can be moved outside if preferred) ---
GCP_PROJECT_ID = os.getenv("GCLOUD_PROJECT") # Loaded by load_dotenv if in .env
GCP_REGION = "us-central1"
MODEL_NAME_VERTEX = "gemini-1.5-flash-001" # Or "gemini-1.5-pro-001"


# --- Perform transcription ---
txt_output = transcribe_audio_simple(
    wav_input,
    project_id=GCP_PROJECT_ID,
    region=GCP_REGION,
    model_name=MODEL_NAME_VERTEX
)

# --- Print output ---
if txt_output:
    print("Transcription:")
    print(txt_output)
else:
    print("Transcription failed.")