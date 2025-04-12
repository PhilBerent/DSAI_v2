from openai import OpenAI
import os

prompt = "What is the capital of France?"
glt_model = "gpt-4-turbo-preview"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model=glt_model, 
    messages=[{"role": "system", "content": "You are an expert assistant."}, 
            {"role": "user", "content": prompt}], 
    temperature=.7
)

print(response.choices[0].message.content)
