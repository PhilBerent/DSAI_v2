import os
import openai
import ast  # Use ast.literal_eval for safe parsing
from UtilityFunctions import *

OAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_text():
    prompt = """Return a Python list containing two elements:
    1. A poem of about 150 words as a string.
    2. An integer between 1 and 3000.
    
    Format the response exactly as a Python list, e.g.:
    ["This is a poem...", 1234]"""

    client = openai.Client(api_key=OAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful AI that generates structured Python responses."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
    )
    
    # Get the raw text output
    output = response.choices[0].message.content.strip()
    aa=2

    try:
        return ast.literal_eval(output)  # Safely convert the response into a Python list
    except (SyntaxError, ValueError) as e:
        print("Error parsing response:", e)
        return None

if __name__ == "__main__":
    response = generate_text()
    print(response)

