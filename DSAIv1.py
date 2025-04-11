import sys  
import os
import json
import sqlite3
import fitz
import time
import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
from openai import OpenAI, OpenAIError
import uuid
import tiktoken
from datetime import datetime
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from dotenv import load_dotenv
from DSAIParams import *
from UtilityFunctions import *
from globals import *
import asyncio
import re
import httpx
from openai import AsyncOpenAI
from DSAIUtilities import *


# OverWriteDatabaseAndIndex = True  # Set to True to reset everything
OverWriteDatabaseAndIndex = True
# OperationType can be addToLibrary, askQuestions, rebuildDataBase, printDictionary, printChunksFromDocument
# OperationType = "askQuestions"  
QandASessionComment = ""
OperationType = "addToLibrary"

# --- CONFIGURATION ---
load_dotenv()

# Configure logging to write to a file
logging.basicConfig(filename=Logfile, level=logging.INFO, encoding="utf-8", format="%(asctime)s - %(message)s")

# --- SETUP DATABASE ---
def setup_database(doc_dbName, chunk_dbName, db_file, overwriteExisting=False):
    # Validate table names against AllowedTables
    if doc_dbName not in AllowedTables or chunk_dbName not in AllowedTables:
        print("Error: One or more table names are not in the allowed list. Operation aborted.")
        return False

    # Delete existing database if requested
    if overwriteExisting:
        if os.path.exists(db_file):
            confirm = input(
                "WARNING: This will delete all data in the existing database. "
                "Continue? (yes/no): "
            ).strip().lower()
            if confirm not in ("yes", "y"):
                print("Operation canceled.")
                return False
            
            os.remove(db_file)
            print(f"Deleted existing database file: {db_file}")

    # Connect to database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Fetch all existing tables in one query for efficiency
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}  # Convert to set for fast lookup

    text_table_exists = doc_dbName in existing_tables
    chunk_table_exists = chunk_dbName in existing_tables

    # Create `doc_dbName` table if it doesn't exist
    if not text_table_exists:
        cursor.execute(f'''CREATE TABLE "{doc_dbName}" (
            document_id TEXT PRIMARY KEY,
            name TEXT UNIQUE,
            subject TEXT,
            date_added DATETIME,
            start_chunk_id INTEGER,
            end_chunk_id INTEGER,
            chunk_overlap INTEGER,
            file_path TEXT
        );''')

        # Add indexes for fast search & sorting
        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_name ON "{doc_dbName}"(name)')
        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_subject ON "{doc_dbName}"(subject)')
        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_date ON "{doc_dbName}"(date_added)')

        print(f"Database '{doc_dbName}' initialized.")

    # Create `chunk_dbName` table if it doesn't exist
    if not chunk_table_exists:
        cursor.execute(f'''CREATE TABLE "{chunk_dbName}" (
            chunk_id TEXT PRIMARY KEY,
            chunk_num INTEGER,
            document_id TEXT,
            text TEXT,
            FOREIGN KEY(document_id) REFERENCES "{doc_dbName}"(document_id)
        );''')

        # Add an index for faster retrieval of chunks
        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_document_id ON "{chunk_dbName}"(document_id)')
        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_chunk_num ON "{chunk_dbName}"(chunk_num)')
        print(f"Database '{chunk_dbName}' initialized.")

    # Commit and close connection
    conn.commit()
    conn.close()

    return True  # Indicate success

def read_document(doc_path):
    text = ""
    success=True
    if doc_path.lower().endswith(".pdf"):
        with fitz.open(doc_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    elif doc_path.lower().endswith((".txt", ".md")):
        with open(doc_path, "r", encoding="utf-8") as file:
            text = file.read()
    else:
        print("Error: Unsupported file type. Only PDF and text files are allowed.")
        success=False
    # endregion
    return text, success

async def embed_batch(batch_id, emb_model_name, batch_chunks):
    """Asynchronously sends a batch of text chunks to OpenAI's embedding API with retry logic."""
    
    headers = {
        "Authorization": f"Bearer {OAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": emb_model_name,
        "input": batch_chunks  # List of strings
    }

    retries = 0
    while retries <= MAX_BATCH_EMBED_RETRIES:
        backoff = INITIAL_BATCH_EMBED_BACKOFF * (2 ** retries)  # Exponential backoff time
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    json=payload
                )

            # Check API response
            if response.status_code == 200:
                return [res["embedding"] for res in response.json()["data"]]

            print(f"⚠️ Batch {batch_id} failed: {response.status_code} - {response.text}")
            raise ValueError(f"API error: {response.text}")

        except Exception as e:
            print(f"⚠️ Embedding error (retry {retries + 1}/{MAX_BATCH_EMBED_RETRIES}) for batch {batch_id}: {e}")
            await asyncio.sleep(backoff)  # Exponential backoff
            retries += 1

    print(f"❌ Failed embedding batch {batch_id} after {MAX_BATCH_EMBED_RETRIES} retries. Skipping batch.")
    return [None] * len(batch_chunks)  # Return placeholders for failed batches

# --- PROCESS DOCUMENT ---
async def process_document(emb_mod_name, db_file, doc_dbName, chunk_dbName, index, doc_path, docToAddMeta, 
	chunkSize, chunkOverlap, doc_text="", use_current_datetime=True, use_filename_for_name=True, read_from_file=True):

    # region check if databases are in safe list    
    if doc_dbName not in AllowedTables or chunk_dbName not in AllowedTables:
        print("Error: One or more table names are not in the allowed list. Operation aborted.")
        return
    # endregion
    doc_name, doc_subject, dtString = docToAddMeta
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # region get document
    # region Get document name and date
    # Use filename as document name if specified
    if use_filename_for_name:
        doc_name = os.path.basename(doc_path)
    # Determine document date
    if use_current_datetime:
        doc_dt = datetime.now()
    else:
        doc_dt, success = parse_datetime(dtString)
        if not success:
            print("Error: Invalid date format. Please use 'YYYY/MM/DD hh:mm[am/pm]' format.")
            conn.close()
            return
    # endregion
    
    # region Check if document already exists
    cursor.execute(f'SELECT COUNT(*) FROM "{doc_dbName}" WHERE name = ?', (doc_name,))
    if cursor.fetchone()[0] > 0:
        print(f"Error: A document named '{doc_name}' already exists. Aborting.")
        conn.close()
        return
    # endregion
    # region Get document text
    if read_from_file:
        text, success = read_document(doc_path)
        if not success:
            print("Error: Invalid file type. Only PDF and text files are allowed.")
            conn.close()
            return
    else:
        if doc_text == "":
            print("Error: No document text provided. Aborting.")
            conn.close()
            return
        text = doc_text
    # endregion
    # endregion

    # Generate a unique document ID
    document_id = str(uuid.uuid4())

    # Get the highest existing chunk number to ensure correct numbering
    cursor.execute(f'SELECT end_chunk_id FROM "{doc_dbName}" ORDER BY ROWID DESC LIMIT 1')
    result = cursor.fetchone()
    highest_chunk_num = result[0] if result else 0

    # region Chunk the text with progress tracking
    print("Chunking document...")
    
    # Get the correct encoding name
    encoding_name = tiktoken.encoding_for_model(emb_mod_name).name

    # Initialize the text splitter with the correct encoding
    text_splitter = TokenTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap, encoding_name=encoding_name) 
    encoding = tiktoken.encoding_for_model(emb_mod_name)

    # Split the document into chunks
    chunks = list(tqdm(text_splitter.split_text(text), desc="Chunking Progress", unit="chunk"))
    total_chunks = len(chunks)
    print(f"Total chunks created: {total_chunks}")
    # endregion

    # region Count tokens in the document and determmine batch size
    num_tokens = count_tokens_in_document(text, emb_mod_name)
    batch_size = max(1, min(get_optimal_batch_size(num_tokens, chunkSize), total_chunks))
    # endregion

    # region Process embeddings in batches with progress tracking
    chunk_num = highest_chunk_num + 1 if highest_chunk_num > 0 else 0
    start_chunk = chunk_num

    print("Generating embeddings and indexing in Pinecone (Async)...")

    # Prepare batches of text chunks
    chunk_batches = [chunks[i:i + batch_size] for i in range(0, total_chunks, batch_size)]
    tasks = [embed_batch(i, emb_mod_name, batch) for i, batch in enumerate(chunk_batches)]
    batch_results = await asyncio.gather(*tasks)

    # Process results and insert into database
    vectors_to_upsert = []
    chunk_details = [] # Collect all chunk details for this batch

    for batch_chunks, batch_vectors in zip(chunk_batches, batch_results):
        if any(v is None for v in batch_vectors):
            print(f"Skipping some failed embeddings in this batch.")
            batch_vectors = [v if v is not None else [0.0] * ModelDimension for v in batch_vectors]  # Replace with zero vectors

        for chunk_text, vector in zip(batch_chunks, batch_vectors):
            if vector is None:
                print(f"Skipping chunk {chunk_num} due to repeated embedding failures.")
                continue  # Skip failed chunks

            vector_id = str(uuid.uuid4())
            vectors_to_upsert.append({
                "id": vector_id,
                "values": vector,
                "metadata": {
                    "chunk_num": chunk_num,
                    "document_id": document_id,
                    "text": chunk_text
                }
            })
            chunk_details.append((vector_id, chunk_num, document_id, chunk_text))
            chunk_num += 1

    # Upsert vectors to Pinecone
    if vectors_to_upsert:
        # upsert vectors in batches
        for i in range(0, len(vectors_to_upsert), UPSERT_BATCH_SIZE):
            batch = vectors_to_upsert[i:i + UPSERT_BATCH_SIZE]
            index.upsert(vectors=batch)

    # Insert chunks into SQLite database
    if chunk_details:
        cursor.executemany(f'INSERT INTO "{chunk_dbName}" (chunk_id, chunk_num, document_id, text) VALUES (?, ?, ?, ?)', chunk_details)

    # endregion
    end_chunk = chunk_num - 1  # Adjust since chunk_num increments after the last chunk

    # region Store document metadata with chunk range
    cursor.execute(f'''INSERT INTO "{doc_dbName}"
                    (document_id, name, subject, date_added, start_chunk_id, end_chunk_id, chunk_overlap, file_path) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (document_id, doc_name, doc_subject, doc_dt, start_chunk, end_chunk, chunkOverlap, doc_path))

    conn.commit()
    conn.close()
    # endregion

    print(f"Document '{doc_name}' added with {total_chunks} chunks, starting at {start_chunk}.")
    
# --- QUERY SYSTEM ---
def query_library(embeddings, doc_dbName, chunk_dbName, db_file, index, max_query_batch_size=50, num_chunks_to_retrieve=5, response_temperature=0.7):
    # region Validate table names against AllowedTables
    if doc_dbName not in AllowedTables or chunk_dbName not in AllowedTables:
        print("Error: One or more table names are not in the allowed list. Operation aborted.")
        return
    # endregion

    # open db
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return

    conversation_history = []
    history_text = ""
    client = OpenAI(api_key=OAI_API_KEY)

    # region Logging setup
    if OverWriteLog:
        open(Logfile, "w", encoding="utf-8").close()  # Clear log file
    sessionComment = QandASessionComment if QandASessionComment != "" else "No comment"
    logging.info(f"\n=== Session date time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\nComments: {sessionComment}\n")
    # endregion

    try:
        while True:
            user_input = input("Your question: ").strip()
            if user_input.lower() in ["exit", "stop"]:
                confirm = input("Are you sure you want to exit? (yes/no): ").strip().lower()
                if confirm in ["yes", "y"]:
                    break
                continue

            query_embedding = embeddings.embed_query(user_input)
            results = index.query(
                vector=query_embedding, 
                top_k=num_chunks_to_retrieve,
                include_values=True,
                include_metadata=True
            )

            # Extract chunk IDs
            chunk_ids = [match["id"] for match in results["matches"]] if "matches" in results else []
            if not chunk_ids:
                print("No relevant chunks found. Trying alternative search...")
                logging.info("No relevant chunks found. Trying alternative search...\n")

                # Alternative approach: Retrieve based on metadata search (if applicable)
                cursor.execute(f'SELECT text FROM "{chunk_dbName}" WHERE text LIKE ?', ('%' + user_input + '%',))
                alt_results = cursor.fetchall()
                
                if alt_results:
                    logging.info(f"Alternative search found {len(alt_results)} possible matches:\n")
                    for text in alt_results[:3]:  # Displaying top 3
                        print(f"- {text[:200]}...\n")
                else:
                    print("No matches or alternative matches found.")
                    logging.info("No matches or alternative matches found.\n")
                continue
        
            retrieved_chunks = {}
            chunk_log = []
            for i in range(0, len(chunk_ids), max_query_batch_size):
                batch = chunk_ids[i:i + max_query_batch_size]
                query = f'SELECT document_id, chunk_id, chunk_num, text FROM "{chunk_dbName}" WHERE chunk_id IN ({",".join(["?"] * len(batch))})'
                cursor.execute(query, batch)
                for doc_id, chunk_id, chunk_num, text in cursor.fetchall():
                    text = text.strip()  # Keep text unchanged for storage
                    retrieved_chunks.setdefault(doc_id, []).append((chunk_num, text))

                    # Convert text for logging (remove non-ASCII characters)
                    text_log_safe = re.sub(r'[^\x00-\x7F]+', '', text)
                    chunk_log.append(f"Chunk {chunk_num}:\n{text_log_safe}\n")

            # Format context properly (original text, keeping all characters)
            context_text = "\n\n".join(
                [f"Document {doc_id}:\n" + "\n".join([f"{chunk_num}: {text}" for chunk_num, text in texts]) for doc_id, texts in retrieved_chunks.items()]
            )

            prompt = f"{history_text}\n\nContext:\n{context_text}\n\nUser: {user_input}"

            try:
                response = client.chat.completions.create(
                    model=Glt_model, 
                    messages=[{"role": "system", "content": LLMInstructions}, 
                              {"role": "user", "content": prompt}], 
                    temperature=response_temperature
                )
                answer = response.choices[0].message.content
            except OpenAIError as e:
                answer = f"OpenAI API error: {str(e)}"
            except Exception as e:
                answer = f"An unexpected error occurred: {str(e)}"
 
            history_text = f"{history_text}\nQ: {user_input}\nA: {answer}"
            print(f"\nAnswer: {answer}\n")

            # Append Q&A to the log file
            conversation_history.append((user_input, answer))

            # region Logging Q&A
            chunk_log_clean = "\n".join(chunk_log)  # Ensure only ASCII values in the log
            log_entry = f"\nQuestion: {user_input}\nAnswer: {answer}\n\nChunks Used:\n{chunk_log_clean}\n{EndMarker}\n"            
            logging.info(log_entry.strip())
            # endregion

            for handler in logging.getLogger().handlers:
                handler.flush()            
    finally:
        conn.commit()
        conn.close()

# --- PRINT DICTIONARY FUNCTION ---
def print_dictionary(doc_dbName, print_to_screen=True, print_to_file=False, dictPrintOutputPath=None):
    
    # region check if databases are in safe list
    if doc_dbName not in AllowedTables:
        print("Error: Table name is not in the allowed list. Operation aborted.")
        return
    # endregion
    
    conn = sqlite3.connect(Db_file)
    cursor = conn.cursor()
    cursor.execute(f"SELECT name, subject, date_added FROM {doc_dbName}")
    records = cursor.fetchall()
    conn.close()
    
    output = "\n".join([f"{name} - {subject} - {date_added}" for name, subject, date_added in records])
    if print_to_screen:
        print(output)
    if print_to_file and dictPrintOutputPath:
        with open(dictPrintOutputPath, "w", encoding="utf-8") as file:
            file.write(output)

#  function to print chunks from a document in the database and print to a file "chunkPrint.txt". Function specified start chunk and end chunk to print if chunk_start <=0 then start at first chunk. If chunk_end <=0 then end at last chunk
def print_chunks_from_document(doc_dbName, chunk_dbName, doc_name, chunkPrintOutputPath, chunk_start=0, chunk_end=0):
    # region check if databases are in safe list
    if doc_dbName not in AllowedTables or chunk_dbName not in AllowedTables:
        print("Error: One or more table names are not in the allowed list. Operation aborted.")
        return
    # endregion

    conn = sqlite3.connect(Db_file)
    cursor = conn.cursor()

    # Get document ID
    cursor.execute(f"SELECT document_id FROM {doc_dbName} WHERE name = ?", (doc_name,))
    result = cursor.fetchone()
    if not result:
        print(f"Error: Document '{doc_name}' not found.")
        conn.close()
        return
    doc_id = result[0]

    # region Get chunks
    query = f"SELECT text FROM {chunk_dbName} WHERE document_id = ?"
    params = [doc_id]

    if chunk_start > 0:
        query += " AND chunk_id >= ?"
        params.append(chunk_start)

    if chunk_end > 0:
        query += " AND chunk_id <= ?"
        params.append(chunk_end)

    query += " ORDER BY chunk_id ASC"
    cursor.execute(query, params)

    chunks = cursor.fetchall()
    conn.close()
    # endregion
    
    if not chunks:
        print(f"No chunks found for document '{doc_name}'.")
        return

    # Write chunks to file
    with open(chunkPrintOutputPath, "w", encoding="utf-8") as file:
        for i, (chunk,) in enumerate(chunks):
            file.write(f"Chunk {i}:\n{chunk}\n\n")

    print(f"Chunks for document '{doc_name}' have been written to {chunkPrintOutputPath}.")
    
def reconstruct_document(doc_name, doc_dbName, chunk_dbName):
    """Reconstructs a document from stored chunks, handling overlaps."""
    
    # region check if databases are in safe list
    if doc_dbName not in AllowedTables or chunk_dbName not in AllowedTables:
        print("Error: One or more table names are not in the allowed list. Operation aborted.")
        return
    # endregion

    conn = sqlite3.connect(Db_file)
    cursor = conn.cursor()

    # Retrieve chunk range for the document
    cursor.execute(f"SELECT start_chunk_id, end_chunk_id, chunk_overlap FROM {doc_dbName} WHERE name = ?", (doc_name,))
    result = cursor.fetchone()

    if not result:
        print(f"Error: No document found with name '{doc_name}'.")
        return None

    start_chunk_id, end_chunk_id, stored_chunk_overlap = result
    stored_chunk_overlap = stored_chunk_overlap if stored_chunk_overlap is not None else 0

    # Retrieve ordered chunks
    cursor.execute(f"SELECT text FROM {chunk_dbName} WHERE chunk_id BETWEEN ? AND ? ORDER BY chunk_id ASC", 
                   (start_chunk_id, end_chunk_id))
    chunks = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not chunks:
        print(f"Error: No chunks found for document '{doc_name}'.")
        return None

    # Remove overlap and reconstruct document
    reconstructed_text = chunks[0]  # Start with first chunk
    for i in range(1, len(chunks)):
        reconstructed_text += chunks[i][stored_chunk_overlap:]  # Use stored overlap

    return reconstructed_text

# function that reconstructs all documents in doc_dbName database by calling reconstruct_document and returns the reconstructed documents as a dictionary in the form {doc_name: reconstructed_text}
def reconstruct_all_documents(doc_dbName, chunk_dbName):
    """Reconstructs all documents stored in the database and returns them as a list of tuples."""

    # Validate table names
    if doc_dbName not in AllowedTables or chunk_dbName not in AllowedTables:
        print("Error: One or more table names are not in the allowed list. Operation aborted.")
        return []

    conn = sqlite3.connect(Db_file)
    cursor = conn.cursor()

    # Retrieve all stored documents
    cursor.execute(f"SELECT name FROM {doc_dbName}")
    stored_documents = cursor.fetchall()

    reconstructed_documents = []

    for doc in stored_documents:  # Unpack single-column result
        doc_name, doc_subject, doc_dateTime = doc
        docDTString = doc_dateTime.strftime('%Y-%m-%d %H:%M:%S')
        doc_text = reconstruct_document(doc_name, doc_dbName, chunk_dbName)
        docDetails = (doc_name, doc_subject, docDTString)
        if doc_text is not None:  # Ensure reconstruction was successful
            reconstructed_documents.append((docDetails, doc_text))

    conn.close()
    return reconstructed_documents

# function that rebuilds the database by reprocessing all stored documents using the latest parameters and returns the index
def rebuild_database_and_index(pc, embeddings, db_file, doc_dbName, chunk_db_name, doc_path, indexName, 
            maxEmbedBatchSize, 	recreateFromExistingChunks=False):

    """Rebuilds the database by reprocessing all stored documents using the latest parameters."""
    # region check if databases are in safe list
    if Doc_db_name not in AllowedTables or chunk_db_name not in AllowedTables:
        print("Error: One or more table names are not in the allowed list. Operation aborted.")
        return
    # endregion
    
    # region Ask for confirmation before proceeding 
    while True:
        confirm = input(f"This operation will delete all existing stored data and reprocess all documents stored with \
            root name {dbAndIndexRootName}. Do you wish to continue? (yes/no): ").strip().lower()
        
        if confirm == "yes" or confirm == "y":
            break  # Proceed with reinitialization
        elif confirm == "no":
            print("Operation canceled. Exiting program.")
            sys.exit(0)
        else:
            print("Invalid response. Please enter 'yes' or 'no'.")

    print("Rebuilding database...")
    conn = sqlite3.connect(Db_file)
    cursor = conn.cursor()

    setup_database(doc_dbName, chunk_db_name)
    index = getOrCreateIndex(pc, indexName, overwriteExisting=True)
    
    print("Database structure reset. Now reprocessing all documents...")

    # Reconstruct all documents
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {doc_dbName}")
        numDocs = cursor.fetchone()[0]
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        numDocs = None

    if not recreateFromExistingChunks and numDocs > 1:
        print("Error: Multiple documents found. Please specify a single document to rebuild.")
        return

    if recreateFromExistingChunks:
        stored_documents = reconstruct_all_documents(doc_dbName, chunk_db_name)
        if not stored_documents:
            print("No stored documents found. Database rebuild complete.")
            return

        # Reprocess each document
        for storedDoc in stored_documents:
            docDetails, docText = storedDoc
            doc_name, doc_subject, doc_dtString = docDetails
            print(f"Reprocessing document: {doc_name}")

        process_document(pc, embeddings, db_file, doc_dbName, chunk_db_name, indexName, doc_path, docDetails,   
            doc_text=docText, use_current_datetime=False, use_filename_for_name=False, max_embed_batch_size=maxEmbedBatchSize, 
            read_from_file=False)
    else:
        # Process the single document
        process_document(pc, embeddings, db_file, doc_dbName, chunk_db_name, indexName, doc_path,  Doc_to_add_meta,
                        max_embed_batch_size=maxEmbedBatchSize, read_from_file=True)
    
    print("Database rebuild complete.")
    return index

def getMaxEmbedBatchSize():
    APIRequestSize = ModelDimension*8
    maxBatchSizeFromAPILimit = MaxAPIRequestSize//APIRequestSize 
    maxBatchSizeFromChunkLimit = MaxBytesInChunkBatch // Chunk_size
    maxBatchSize = min(maxBatchSizeFromAPILimit, maxBatchSizeFromChunkLimit)
    return maxBatchSize

def getMaxQueryBatchSize():
    chunkIDSize = 40
    sizePerChunk = chunkIDSize + Chunk_size
    maxQueryBatchSize = MaxQuerySize // sizePerChunk
    return maxQueryBatchSize

# --- EXECUTION BASED ON OPERATION TYPE ---
def getOrCreateIndex(pc, indexName, overwriteExisting=False):
    MAX_WAIT_TIME = 60  # Maximum time to wait in seconds
    CHECK_INTERVAL = 1   # Time interval between checks
    start_time = time.time()

    if pc.has_index(indexName):
        if overwriteExisting:
            print(f"Deleting existing index: {indexName}")
            pc.delete_index(indexName)
        else:
            return pc.Index(indexName)

    try:
        pc.create_index(
            name=indexName,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    except pinecone.exceptions.PineconeException as e:
        print(f"Error creating Pinecone index: {e}")
        return None

    while True:
        try:
            status = pc.describe_index(indexName).status['ready']
            if status:
                return pc.Index(indexName)
        except Exception as e:
            print(f"Error checking index status: {e}")
        if time.time() - start_time > MAX_WAIT_TIME:
            raise TimeoutError(f"Pinecone index '{indexName}' did not become ready within {MAX_WAIT_TIME} seconds.")
        time.sleep(CHECK_INTERVAL)

def initializeLog():
    if OverWriteLog:
        open(Logfile, "w", encoding="utf-8").close()
    else:
        logging.info("\n=== New Session ===\n\n")
    logging.basicConfig(filename=Logfile, level=logging.INFO, encoding="utf-8", format="%(asctime)s - %(message)s")

setup_database(Doc_db_name, Chunk_db_name, Db_file, OverWriteDatabaseAndIndex)
Pc = Pinecone(api_key=PINECONE_API_KEY)
Index = getOrCreateIndex(Pc, Index_name, overwriteExisting=OverWriteDatabaseAndIndex)
Max_Embed_batch_size = getMaxEmbedBatchSize()
Max_Query_batch_size = getMaxQueryBatchSize()

if OperationType == "addToLibrary":
    # Run the async function
    asyncio.run(process_document(Embeddings_model_name, Db_file, Doc_db_name, Chunk_db_name, 
        Index, DocToAddPath, Doc_to_add_meta, Chunk_size, Chunk_overlap, use_current_datetime=UseCurrentDateTime, use_filename_for_name=UseFilenameForName, read_from_file=True))
elif OperationType == "askQuestions":
    query_library(Embeddings_model, Doc_db_name, Chunk_db_name, Db_file, Index, max_query_batch_size=Max_Query_batch_size, 
        num_chunks_to_retrieve=Num_chunks_to_retrieve, response_temperature=Temperature)
elif OperationType == "printDictionary":
    print_dictionary(print_to_screen=True, print_to_file=True, dictPrintOutputPath=DictionaryPrintOutputPath)
elif OperationType == "rebuildDatabase":
    Index = rebuild_database_and_index(Pc, Embeddings_model, Db_file, Doc_db_name, Chunk_db_name, Index_name, Max_Embed_batch_size)
elif OperationType == "printChunks":
    print_chunks_from_document(Doc_db_name, Chunk_db_name,  DocToPrint, ChunkPrintOutputPath, chunkStart=0, chunkEnd=0)