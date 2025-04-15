import os
import re
import asyncio
import aiohttp
from openai import AsyncOpenAI
from UtilityFunctions import *
from globals import *
from DSAIParams import *

# Define the file path
file_path = r"C:\Users\Phil\Documents\DataSphere AI\DataSphere AI Not Code\Inputs And Outputs\Texts For Testing\Pride and PrejudiceDingo2.txt"

# Read the document
with open(file_path, "r", encoding="utf-8") as file:
    document_content = file.read()

# Split document into chapters (assuming "Chapter X" format)
chapters = re.split(r'(?=Chapter)', document_content)

# Initialize OpenAI client (Asynchronous)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def summarize_chapter(chapter_text, chapter_number):
    """ Sends a request to OpenAI to summarize a single chapter asynchronously. """
    prompt = (f"Summarize the following chapter in about 100 words using only the information in the document. "
              f"Do not use any prior knowledge about Jane Austen or Pride and Prejudice:\n\n{chapter_text[:4000]}")  # Truncate if necessary
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in literature analysis and summarization."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        summary = response.choices[0].message.content.strip()
        print(f"Processed Chapter {chapter_number}")  # Progress indicator
        return summary
    except Exception as e:
        print(f"Error processing Chapter {chapter_number}: {e}")
        return f"Error processing Chapter {chapter_number}"


async def main():
    """ Handles parallel execution of chapter summarization requests. """
    tasks = [summarize_chapter(chapter, i + 1) for i, chapter in enumerate(chapters) if chapter.strip()]
    chapter_summaries = await asyncio.gather(*tasks)

    # Generate final plot summary using all chapter summaries
    final_prompt = ("Here are the summaries of each chapter:\n\n" +
                    "\n\n".join(chapter_summaries) +
                    "\n\nProvide a summary of the novel's overall plot in about 300 words based on the chapter summaries. "
                    "Use only the information in the document. Under no circumstances should you use any prior knowledge about Jane Austen or Pride and Prejudice.")

    try:
        final_response = await client.chat.completions.create(
            model=glt_model,
            messages=[
                {"role": "system", "content": "You are an expert in literature analysis and summarization."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.7
        )
        final_summary = final_response.choices[0].message.content.strip()
    except Exception as e:
        final_summary = f"Error generating final summary: {e}"
    
    # Save summary to file
    output = "Chapter Summaries:\n\n" + "\n\n".join(chapter_summaries) + "\n\nFinal Plot Summary:\n" + final_summary
    WriteToFile(output)
    # print("\nFinal Plot Summary:\n", final_summary)


# Run the asynchronous event loop
asyncio.run(main())
