import asyncio
import time
import uuid
import numpy as np
from pinecone import Pinecone
import os

# Import parameters (assuming PINECONE_API_KEY & IndexName are defined in DSAIParams)
import DSAIParams  

async def upsert_to_pinecone(index, vectors):
    """Async function to upsert vectors into Pinecone."""
    start_time = time.time()
    
    if vectors:
        index.upsert(vectors=vectors)

    elapsed_time = time.time() - start_time
    return len(vectors), elapsed_time

async def test_upsert_performance(index, batch_sizes, num_vectors=1000, vector_dim=1536):
    """
    Tests different batch sizes for upserting vectors into Pinecone in parallel,
    and also measures performance of upserting all vectors in a single call.
    
    :param index: Pinecone index object
    :param batch_sizes: List of batch sizes to test
    :param num_vectors: Total number of vectors to insert
    :param vector_dim: Dimensionality of embeddings
    """
    # Generate fake embedding vectors (simulating 400-token texts)
    vectors_to_upsert = [{
        "id": str(uuid.uuid4()),
        "values": np.random.rand(vector_dim).tolist(),  # Simulated embedding
        "metadata": {
            "text": f"Simulated text {i} with ~400 tokens"
        }
    } for i in range(num_vectors)]

    print(f"🚀 Testing upsert performance with {num_vectors} vectors...")

    results = []

    # 🔹 **Test Upserting All Vectors in a Single Batch**
    print("\n🔹 Testing Upserting All Vectors at Once...")
    start_time = time.time()
    # upsert vectors 250 at a time
    for i in range(0, num_vectors, 250):
        index.upsert(vectors=vectors_to_upsert[i:i + 250])
    total_time_single_batch = time.time() - start_time
    avg_time_per_vector_single_batch = total_time_single_batch / num_vectors

    print(f"✅ Single Batch Upsert: {num_vectors} vectors in {total_time_single_batch:.2f}s")
    print(f"⏳ Avg time per vector (Single Batch): {avg_time_per_vector_single_batch:.4f}s")

    results.append(("Single Batch", num_vectors, total_time_single_batch, None, avg_time_per_vector_single_batch))

    # 🔹 **Test Parallel Upserts with Different Batch Sizes**
    for batch_size in batch_sizes:
        print(f"\n🔹 Testing batch size: {batch_size}")

        # Create batch tasks
        upsert_tasks = [
            upsert_to_pinecone(index, vectors_to_upsert[i:i + batch_size])
            for i in range(0, num_vectors, batch_size)
        ]

        # Measure execution time
        start_time = time.time()
        upsert_results = await asyncio.gather(*upsert_tasks)
        total_time = time.time() - start_time

        # Summarize performance
        total_vectors_upserted = sum(r[0] for r in upsert_results)
        avg_batch_time = sum(r[1] for r in upsert_results) / len(upsert_results)
        avg_time_per_vector = total_time / num_vectors

        print(f"✅ Batch Size {batch_size}: Upserted {total_vectors_upserted} vectors in {total_time:.2f}s")
        print(f"⏳ Avg time per batch: {avg_batch_time:.2f}s")
        print(f"⏳ Avg time per vector: {avg_time_per_vector:.4f}s")

        results.append((batch_size, total_vectors_upserted, total_time, avg_batch_time, avg_time_per_vector))

    return results

# Example usage
if __name__ == "__main__":
    # Initialize Pinecone using your method
    Pc = Pinecone(api_key=DSAIParams.PINECONE_API_KEY)
    index = Pc.Index(DSAIParams.Index_name)  # Use globally defined IndexName

    # Define batch sizes to test
    batch_sizes_to_test = [50, 100, 200]
    numVectors = 1023
    
    # Run the test
    results = asyncio.run(test_upsert_performance(index, batch_sizes_to_test, num_vectors=numVectors))

    print("\n🚀 Final Results:")
    for batch_size, vectors_upserted, total_time, avg_b_time, avg_v_time in results:
        if batch_size == "Single Batch":
            print(f"🔹 {batch_size}: {vectors_upserted} vectors in {total_time:.2f}s (Avg per vector: {avg_v_time:.4f}s)")
        else:
            print(f"🔹 Batch Size {batch_size}: {vectors_upserted} vectors in {total_time:.2f}s "
                  f"(Avg per batch: {avg_b_time:.2f}s, Avg per vector: {avg_v_time:.4f}s)")
