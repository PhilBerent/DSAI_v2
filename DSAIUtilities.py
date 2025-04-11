import sqlite3
import pinecone
import tiktoken
from DSAIParams import *

def deleteAll_dbs_and_Indexes(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    try:
        # Get all table names in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("No tables found in the database.")
        else:
            confirm = input("WARNING: This will permanently delete all tables and indexes. Continue? (yes/no): ").strip().lower()
            if confirm not in ("yes", "y"):
                print("Operation canceled.")
                conn.close()
                return False

            # Delete all tables
            for (table_name,) in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                print(f"Deleted table: {table_name}")

            # Delete sqlite_sequence if it exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
            if cursor.fetchone():
                cursor.execute("DELETE FROM sqlite_sequence")
                print("Deleted sqlite_sequence (auto-increment tracking).")

            conn.commit()

        # ---- Pinecone: Permanently delete all indexes ----
        try:
            pinecone_indexes = pinecone.list_indexes()  # List all indexes
            for index_name in pinecone_indexes:
                pinecone.delete_index(index_name)  # Permanently delete index
                print(f"Deleted Pinecone index: {index_name}")

        except Exception as e:
            print(f"Error deleting Pinecone indexes: {e}")

    except Exception as e:
        print(f"Error while deleting database tables: {e}")

    finally:
        conn.close()
        print("All databases and indexes permanently removed.")
        return True

def count_tokens_in_document(document_text, emb_model_name):
    """Counts the total number of tokens in a document using OpenAI's tokenizer."""
    encoding = tiktoken.encoding_for_model(emb_model_name)  # Load the tokenizer for the model
    tokens = encoding.encode(document_text)  # Convert document text into tokens
    
    return len(tokens)  # Return total token count

def get_optimal_batch_size(total_tokens, chunk_size, max_batch_size=200):
    """Calculate the optimal batch size to ensure exactly 20 API calls while staying within OpenAI limits."""
    num_chunks = max(1, total_tokens // chunk_size)  # Ensure at least 1 chunk
    
    # Compute batch size to ensure exactly 20 API calls
    batch_size = max(1, num_chunks // 20)

    # Cap batch size if needed to prevent API slowdowns
    batch_size = min(batch_size, max_batch_size)

    return batch_size


