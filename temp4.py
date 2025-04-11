import openai
import json
import os
from globals import *
from UtilityFunctions import *
from openai import OpenAI, OpenAIError

# Load OpenAI API Key
OAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Set the API key for the openai library
openai.api_key = OAI_API_KEY

def generate_text(prompt, max_tokens=5000):
    """Function to generate text using OpenAI GPT-4."""
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "system", "content": "You are an expert assistant."}, 
                 {"role": "user", "content": prompt}], 
        temperature=.7
    )
    answer = response.choices[0].message.content
    # Extract and return the response text
    output = response.choices[0].message.content.strip()
    return output

# Example Usage
if __name__ == "__main__":
    prompt = "tell me about the book 'Al-Shamandoura' by Mohammed Khalil Qasim. Please tell me what you know. Then if you can please give me a summary of the plot in as much detail as possible. Specifically tell me the names of the main characters and what happens in the first 3 chapters. Also please let me know if you dont have this information" 
    # If you are not familiar with this please tell me about other works of this author."
    response = generate_text(prompt)
    print(response)
