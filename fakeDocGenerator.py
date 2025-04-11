import openai
import json
import os
from globals import *
from UtilityFunctions import *

# Load OpenAI API Key
OAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OAI_API_KEY:
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY as an environment variable.")

# Supported document types
document_types = ["novel", "academic_paper", "business_report", "legal_contract", "historical_chronicle"]

# Default structural templates
DOCUMENT_TEMPLATES = {
    "novel": ["Plot Outline", "Character List", "Chapter Summaries", "Full Chapters"],
    "academic_paper": ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"],
    "business_report": ["Executive Summary", "Market Analysis", "Financial Data", "Recommendations"],
    "legal_contract": ["Preamble", "Definitions", "Terms & Conditions", "Liabilities", "Signatures"],
    "historical_chronicle": ["Era Overview", "Key Figures", "Major Events", "Cultural Impact", "Legacy"]
}

def generate_text(prompt, max_tokens=1000):
    """Function to generate text using OpenAI GPT-4o."""
    client = openai.Client(api_key=OAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful AI that generates structured text."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def generate_fake_document(doc_type, user_description, word_count, output_format="txt"):
    """Generates a fake document based on user inputs."""
    if doc_type not in document_types:
        raise ValueError(f"Invalid document type. Choose from {document_types}")
    
    # Step 1: Generate outline
    prompt_outline = f"Create a structured outline for a {doc_type} based on this description: {user_description}."
    outline = generate_text(prompt_outline, max_tokens=1000)
    
    # Step 2: Generate section summaries
    sections = DOCUMENT_TEMPLATES[doc_type]
    section_summaries = {}
    for section in sections:
        prompt_summary = f"Generate a detailed summary for the {section} of a {doc_type} following this outline: {outline}"
        section_summaries[section] = generate_text(prompt_summary, max_tokens=800)
    
    # Step 3: Generate full content
    document_content = {}
    words_per_section = max(500, word_count // len(sections))  # Ensure reasonable length per section
    for section, summary in section_summaries.items():
        prompt_full_text = f"Expand the following summary into a full {words_per_section}-word section for a {doc_type}: {summary}"
        document_content[section] = generate_text(prompt_full_text, max_tokens=min(2000, words_per_section * 2))
    
    # Format output
    if output_format == "json":
        output_data = {
            "document_type": doc_type,
            "description": user_description,
            "outline": outline,
            "content": document_content
        }
        return json.dumps(output_data, indent=4)
    else:
        text_output = f"\n\n{doc_type.upper()} - {user_description}\n\n"
        text_output += f"Outline:\n{outline}\n\n"
        for section, content in document_content.items():
            text_output += f"{section}\n{'-' * len(section)}\n{content}\n\n"
        return text_output

# Example Usage
if __name__ == "__main__":
    doc_type = "novel"
    user_desc = "A 19th-century novel about a shipping magnate's rise and fall in the style of Charles Dickens."
    fakeDocName = "fakeDoc.txt"
    fakeDocPath = os.path.join(FakeDocumentBasePath, fakeDocName)
    word_count = 5000
    output = generate_fake_document(doc_type, user_desc, word_count, output_format="txt")
    
    # Save output
    with open("fake_document.txt", "w", encoding="utf-8") as f:
        f.write(output)
    
    print("Fake document generated successfully.")
