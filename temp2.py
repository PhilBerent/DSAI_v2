import asyncio
import httpx
import os
import json
from UtilityFunctions import *
from DSAIParams import *

# OpenAI API Key
MODEL_NAME = "text-embedding-3-small"  # Change if needed
NUM_BATCHES = 5  # Number of parallel API calls
BATCH_SIZE = 3   # Number of strings per batch
async def embed_batch(batch_id, batch_texts):
    """Send a batch of text chunks to OpenAI's embedding API."""
    headers = {
        "Authorization": f"Bearer {OAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "input": batch_texts  # List of strings
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload
            )

        # Debugging: Print response status and any errors
        if response.status_code != 200:
            print(f"❌ Batch {batch_id} failed: {response.status_code} - {response.text}")
            return None

        # Return embeddings
        return response.json()["data"]

    except Exception as e:
        print(f"⚠️ Error in batch {batch_id}: {e}")
        return None

async def main():
    """Create and send batches in parallel."""
    test_batches = [[f"This is test batch {i}, item {j}" for j in range(BATCH_SIZE)] for i in range(NUM_BATCHES)]

    # Launch all batches in parallel
    tasks = [embed_batch(i, batch) for i, batch in enumerate(test_batches)]
    results = await asyncio.gather(*tasks)

    print("\n🚀 Final Results:")
    for i, result in enumerate(results):
        if result:
            print(f"✅ Batch {i} returned {len(result)} embeddings")
        else:
            print(f"❌ Batch {i} failed.")

# Run the test
asyncio.run(main())
