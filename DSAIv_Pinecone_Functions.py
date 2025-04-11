import os
import json
import sqlite3
import fitz
import pinecone
import openai
import uuid
from datetime import datetime
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from UtilityFunctions import *

# --- CONFIGURATION ---
load_dotenv()
OAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Ensure this is set
PINECONE_ENVIRONMENT = "us-east-1"  # Adjust based on Pinecone setup
db_file = "C:/Users/Phil/Documents/DataSphere AI/DataSphere AI Not Code/Inputs And Outputs/Texts For Testing/docDictionary.sqlite"
index_name = "project-library"
docToAddPath = r"C:\Users\Phil\Documents\DataSphere AI\DataSphere AI Not Code\Inputs And Outputs\Texts For Testing\Pride and PrejudiceDingo.txt"
docToAddName = "Test Doc"
docToAddSubject = "AI Research"
doc_dateTime =  "2019/12/1 3:15pm"
useCurrentDateTime = False
useFilenameForName = False
doc_to_add = (docToAddPath, docToAddName, docToAddSubject, doc_dateTime, useCurrentDateTime, useFilenameForName)
dictionaryPrintOutputPath=r"C:/Users/Phil/Documents/DataSphere AI/DataSphere AI Not Code/Inputs And Outputs/Outputs/dictionaryPrint.txt"
logfile = r"C:/Users/Phil/Documents/DataSphere AI/DataSphere AI Not Code/Inputs And Outputs/Outputs/qandALog.txt"

# search paramaterss
chunk_size = 512  # Suggested range: 256-1024
chunk_overlap = 100  # Suggested range: 50-200
embeddings_model = "text-embedding-3-small"
modelDimension = 1536
indexMetric = "cosine"
num_chunks_to_retrieve = 5
glt_model = "gpt-4-turbo-preview"
temperature = 0.7
resetDbWithOnlyNewInput = False  # Set to True to reset everything
overWriteLog = False # Set to True to overwrite the log file

# --- SETUP PINECONE ---
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = setup_pinecone_index(index_name)
# --- SETUP DATABASE ---
def setup_database():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
    table_exists = cursor.fetchone()

    if not table_exists:  # If the table doesn't exist, create it
        cursor.execute('''CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            name TEXT UNIQUE,
            subject TEXT,
            date_added DATETIME,
            start_chunk_id INTEGER,
            end_chunk_id INTEGER,
            chunk_overlap INTEGER,
            vector_ids TEXT,
            file_path TEXT
            );''')


        # Add indexes for fast search & sorting
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON documents(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subject ON documents(subject)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON documents(date_added)")

        print("Database initialized.")

    # --- Create `chunks` table ---
    cursor.execute('''CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id INTEGER PRIMARY KEY,
                        document_id TEXT,
                        text TEXT,
                        FOREIGN KEY(document_id) REFERENCES documents(document_id))''')

    # Add an index for faster retrieval of chunks
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON chunks(document_id)")

    conn.commit()
    conn.close()

setup_database()

import sys  # Required for exiting the program

def reset_db_with_only_new_input():
    if resetDbWithOnlyNewInput:
        confirm = input("This operation will delete all existing stored data. Do you wish to continue? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Operation canceled.")
            return False

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents")
        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('documents', 'chunks')")
        conn.commit()
        conn.close()

        index.delete(delete_all=True)  # Clear Pinecone index
        print("Database successfully reset.")
        return True
    return False

def process_document(doc_path, doc_name, doc_subject="", dtString="zzz", use_current_datetime=True, 
    	use_filename_for_name=True):
    # Handle reinitialization if required
    if reset_db_with_only_new_input():
        print("Rebuilding database with new document...")

    # Ensure a valid date is provided
    if not use_current_datetime:
        doc_dt, success = parse_datetime(dtString)
        if not success:
            print("Error: Invalid date format. Please use 'YYYY/MM/DD hh:mm[am/pm]' format.")
            return
    else:
        doc_dt = datetime.now()

    # Use filename as document name if specified
    if use_filename_for_name:
        doc_name = os.path.basename(doc_path)

    # Open database connection
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Check if document name already exists
    cursor.execute("SELECT COUNT(*) FROM documents WHERE name = ?", (doc_name,))
    if cursor.fetchone()[0] > 0:
        print(f"Error: A document with the name '{doc_name}' already exists. Aborting operation.")
        conn.close()
        return
 
    # Generate unique document ID
    document_id = str(uuid.uuid4())

    # Retrieve highest existing chunk number
    existing_chunk_lists = cursor.fetchall()

    highest_chunk_num = 0
    for row in existing_chunk_lists:
        if row[0]:  # Ensure there are existing chunks
            chunk_list = json.loads(row[0])
            if chunk_list:
                highest_chunk_num = max(highest_chunk_num, max(chunk_list))

    # --- Read the document content ---
    text = ""
    if doc_path.lower().endswith(".pdf"):
        with fitz.open(doc_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    elif doc_path.lower().endswith((".txt", ".md")):
        with open(doc_path, "r", encoding="utf-8") as file:
            text = file.read()
    else:
        print("Error: Unsupported file type. Only PDF and text files are allowed.")
        return

    # --- Chunk the text with progress tracking ---
    print("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)

    chunk_ids = []
    vector_ids = []
    
    total_chunks = len(chunks)
    print(f"Total chunks created: {total_chunks}")

    # --- Generate embeddings and store in Pinecone ---
    print("Generating embeddings and indexing in Pinecone...")
    embeddings = OpenAIEmbeddings(model=embeddings_model)

    # Use batch processing
    batch_size = max(1, total_chunks // 10)  # Avoids division by zero
    batch_data = []
    
    for idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks", unit="chunk")):
        chunk_num = highest_chunk_num + idx + 1
        vector_id = str(uuid.uuid4())  # Generate unique vector ID
        vector = embeddings.embed_documents([chunk])[0]  # Get embedding
        
        vector_ids.append(vector_id)
        chunk_ids.append(chunk_num)

        index.upsert([(vector_id, vector)])  # Store in Pinecone

        # Accumulate chunk data for batch insert
        batch_data.append((chunk_num, document_id, chunk))

        # Execute batch insert every 5% of chunks
        if (idx + 1) % (total_chunks // 20) == 0 or idx == total_chunks - 1:
            cursor.executemany("INSERT INTO chunks (chunk_id, document_id, text) VALUES (?, ?, ?)", batch_data)
            conn.commit()
            batch_data = []  # Clear batch after commit
            
            progress = ((idx + 1) / total_chunks) * 100
            print(f"Indexing progress: {progress:.1f}%")
   
    # Store document metadata using start and end chunk IDs
    cursor.execute('''INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
               (document_id, doc_name, doc_subject, doc_dt, chunk_ids[0], chunk_ids[-1], chunk_overlap, json.dumps(vector_ids), doc_path))
    
    conn.commit()
    conn.close()

    print(f"Document '{doc_name}' successfully added with {total_chunks} chunks, starting from chunk {highest_chunk_num + 1}.")

# --- QUERY SYSTEM ---
def query_library():
    conversation_history = []
    history_text = ""  # Maintain history incrementally

    # If overWriteLog is True, clear the log file at the start of the session
    if overWriteLog:
        open(logfile, "w", encoding="utf-8").close()  # Clear file
    else:
        with open(logfile, "a", encoding="utf-8") as log:
            log.write("\n=== New Session ===\n\n")  # Mark new session

    while True:
        user_input = input("Your question: ")
        if user_input.lower() in ["exit", "stop"]:
            break

        query_embedding = OpenAIEmbeddings(model=embeddings_model).embed_query(user_input)
        results = index.query(queries=[query_embedding], top_k=num_chunks_to_retrieve, include_metadata=False)
        retrieved_chunks = [result["id"] for result in results["matches"]]

        context = []
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        for chunk_id in retrieved_chunks:
            cursor.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            if row:
                context.append(row[0])  # row[0] contains the chunk text
        conn.close()

        # Construct prompt using accumulated history
        context_text = "\n\n".join(context)
        prompt = f"{history_text}\n\nContext from documents:\n{context_text}\n\nUser's question: {user_input}"

        response = openai.ChatCompletion.create(
            model=glt_model,
            messages=[{"role": "system", "content": "You are an expert assistant."},
                      {"role": "user", "content": prompt}],
            temperature=temperature
        )
        answer = response["choices"][0]["message"]["content"]
        
        # Append new question and answer to conversation history
        conversation_history.append({"Q": user_input, "A": answer})
        history_text += f"\nQ: {user_input}\nA: {answer}"  # Append directly

        print(f"Answer: {answer}\n")

        with open(logfile, "a", encoding="utf-8") as log:
            log.write(f"Q{len(conversation_history)}: {user_input}\nAnswer: {answer}\n\n")

# --- PRINT DICTIONARY FUNCTION ---
def print_dictionary(print_to_screen=True, print_to_file=False, dictionary_print_output_path=None):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name, subject, date_added FROM documents")
    records = cursor.fetchall()
    conn.close()
    
    output = "\n".join([f"{name} - {subject} - {date_added}" for name, subject, date_added in records])
    if print_to_screen:
        print(output)
    if print_to_file and dictionary_print_output_path:
        with open(dictionary_print_output_path, "w", encoding="utf-8") as file:
            file.write(output)

def rebuild_database():
    """Rebuilds the database by reprocessing all stored documents using the latest parameters."""
    
    # Ask for confirmation before proceeding
    while True:
        confirm = input("This operation will delete all existing stored data and reprocess all documents. Do you wish to continue? (yes/no): ").strip().lower()
        
        if confirm == "yes":
            break  # Proceed with reinitialization
        elif confirm == "no":
            print("Operation canceled. Exiting program.")
            sys.exit(0)
        else:
            print("Invalid response. Please enter 'yes' or 'no'.")

    print("Rebuilding database...")

    # Reset database and vector store
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Drop existing tables
    cursor.execute("DROP TABLE IF EXISTS documents")
    cursor.execute("DROP TABLE IF EXISTS chunks")
    
    # Recreate tables
    cursor.execute('''CREATE TABLE documents (
        document_id TEXT PRIMARY KEY,
        name TEXT UNIQUE,
        subject TEXT,
        date_added DATETIME,
        start_chunk_id INTEGER,
        end_chunk_id INTEGER,
        chunk_overlap INTEGER,
        vector_ids TEXT,
        file_path TEXT
        );''')

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON documents(name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_subject ON documents(subject)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON documents(date_added)")

    cursor.execute('''CREATE TABLE chunks (
                        chunk_id INTEGER PRIMARY KEY,
                        document_id TEXT,
                        text TEXT,
                        FOREIGN KEY(document_id) REFERENCES documents(document_id))''')

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON chunks(document_id)")

    conn.commit()
    conn.close()

    # Reset Pinecone vector store
    index_info = pinecone.describe_index(index_name)
    if index_info.dimension == modelDimension and index_info.metric == indexMetric:
        index.delete(delete_all=True)  # Only clear records
    else:
        pinecone.delete_index(index_name)
        pinecone.create_index(name=index_name, dimension=modelDimension, metric=indexMetric)
        index = pinecone.Index(index_name)  # Reinitialize if structure changed

    print("Database structure reset. Now reprocessing all documents...")

    # Retrieve all stored documents and reprocess them
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, name, subject, date_added FROM documents")
    stored_documents = cursor.fetchall()
    conn.close()

    if not stored_documents:
        print("No stored documents found. Database rebuild complete.")
        return

    # Reprocess each document
    for doc in stored_documents:
        doc_path, doc_name, doc_subject, doc_dateTime = doc
        print(f"Reprocessing document: {doc_name}")

        # Convert date back to string format if needed
        doc_dateTime = doc_dateTime.strftime("%Y/%m/%d %I:%M%p") if isinstance(doc_dateTime, datetime) else doc_dateTime

        process_document(doc_path, doc_name, doc_subject, doc_dateTime, use_current_datetime=False, use_filename_for_name=False)

    print("Database rebuild complete.")

def reconstruct_document(doc_name):
    """Reconstructs a document from stored chunks, handling overlaps."""
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Retrieve chunk range for the document
    cursor.execute("SELECT start_chunk_id, end_chunk_id, chunk_overlap FROM documents WHERE name = ?", (doc_name,))
    result = cursor.fetchone()

    if not result:
        print(f"Error: No document found with name '{doc_name}'.")
        return None

    start_chunk_id, end_chunk_id, stored_chunk_overlap = result
    stored_chunk_overlap = stored_chunk_overlap if stored_chunk_overlap is not None else 0

    # Retrieve ordered chunks
    cursor.execute("SELECT text FROM chunks WHERE chunk_id BETWEEN ? AND ? ORDER BY chunk_id ASC", 
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

def setup_pinecone_index(index_name):
    try:
        index_info = pinecone.describe_index(index_name)
        if index_info.dimension != modelDimension or index_info.metric != indexMetric:
            print("Existing index configuration mismatch. Recreating index...")
            pinecone.delete_index(index_name)
            pinecone.create_index(name=index_name, dimension=modelDimension, metric=indexMetric)
    except pinecone.exceptions.NotFoundException:
        print("Index not found. Creating a new index...")
        pinecone.create_index(name=index_name, dimension=modelDimension, metric=indexMetric)

    return pinecone.Index(index_name)

index = setup_pinecone_index()


# --- EXECUTION BASED ON OPERATION TYPE ---
operationType = "addToLibrary"  # Set this dynamically as needed
if operationType == "addToLibrary":
    process_document(*doc_to_add)
elif operationType == "askQuestions":
    query_library()
elif operationType == "printDictionary":
    print_dictionary(print_to_screen=True, print_to_file=True, dictionary_print_output_path=dictionaryPrintOutputPath)
