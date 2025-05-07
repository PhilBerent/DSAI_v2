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



import openai

# Replace with your actual OpenAI API key
openai.api_key = OPENAI_API_KEY

# Path to your .wav file
audio_file_path = r"C:\Users\Phil\Documents\DataSphere AI\DataSphere AI Not Code\Misc\2024-09-07-12-19-25.WAV"
# audio_file_path = r"C:\Users\Phil\Documents\DataSphere AI\DataSphere AI Not Code\Misc\Recording.wav"
output =  r"C:\Users\Phil\Documents\DataSphere AI\DataSphere AI Not Code\Misc\wavFile.txt"

import openai

# Replace this with your actual API key


# Open and send the file
client = openai.OpenAI(api_key=OPENAI_API_KEY)
with open(audio_file_path, "rb") as audio_file:
    print("Uploading and transcribing... please wait.")
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

WriteToFile(transcript.text, output)
a=2