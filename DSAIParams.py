import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
from openai import OpenAI
import tiktoken
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from enums_and_constants import *

# Load tokenizer for OpenAI embeddings
encoding = tiktoken.get_encoding("cl100k_base")  # Correct for text-embedding-3 models

# # Use TokenTextSplitter with OpenAI tokenizer

globalDebugCount = 0
OAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Ensure this is set
PINECONE_ENVIRONMENT = "us-east-1"  # Adjust based on Pinecone setup
UPSERT_BATCH_SIZE = 250
PARALLEL_BATCH_SIZE = 20
NEO4J_URI = os.getenv("NEO4J_URI") 
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
Base_OutputPath = r"C:/Users/Phil/Documents/DataSphere AI/DataSphere AI Not Code/Inputs And Outputs/Outputs/"
Db_file = r"C:/Users/Phil/Documents/DataSphere AI/DataSphere AI Not Code/Inputs And Outputs/Texts For Testing/docDictionary.sqlite"
dbAndIndexRootName = "PandPDingo"
inputDocsPath = r"C:\Users\Phil\Documents\DataSphere AI\DataSphere AI Not Code\Inputs And Outputs\Texts For Testing\\"
# docToAddPath = r"C:\Users\Phil\Documents\DataSphere AI\DataSphere AI Not Code\Inputs And Outputs\Texts For Testing\Pride and PrejudiceDingo.txt"
docFileName = "Pride and PrejudiceDingo.txt"
# docFileName = "Pride and PrejudiceDingoTemp.txt"
DocToAddPath = os.path.join(inputDocsPath, docFileName)
DocToAddName = "PanPDingo"
DocToAddSubject = "Literature Pride and Prejudice"
Doc_dateTime = "12/01/2019 3:15pm"
UseCurrentDateTime = False
UseFilenameForName = False
Doc_to_add_meta = (DocToAddName, DocToAddSubject, Doc_dateTime)

# Use `os.path.join()` to construct paths dynamically
DictionaryPrintOutputPath = os.path.join(Base_OutputPath, "dictionaryPrint.txt")
ChunkPrintOutputPath = os.path.join(Base_OutputPath, "chunkPrint.txt")
Logfile = os.path.join(Base_OutputPath, "qandALog.txt")
# region comment on difference between rebuildDatabase OverWriteDatabaseAndIndex
# Not the difference between rebuildDatabase and OverWriteDatabaseAndIndex is that rebuildDatabase will delete all existing data and reprocess all documents stored with root name dbAndIndexRootName and put them in a new database while OverWriteDatabaseAndIndex will delete the existing database and index and create new ones
# params for printChunksFromDocument
# endregion
DocToPrint = "PanPDingo"
PrintChunkStart = 0
PrintChunkEnd = 0

# search paramaterss
IndexMetric = "cosine"
Num_chunks_to_retrieve = 10
Temperature = 0.7
Glt_model = "gpt-4-turbo-preview"
OverWriteLog = True # Set to True to overwrite the log file
# OverWriteLog = False

# data storage paramaters
Chunk_size = 400  # in tokens Suggested range: 256-1024
Chunk_overlap = 60  # Suggested range: 50-200
ModelDimension = 1536
MaxBytesInChunkBatch = 1000000
MaxAPIRequestSize = 4500000 #4.5MB
MaxQuerySize = 100000

Doc_db_name =  dbAndIndexRootName + "_text_db"
Chunk_db_name = dbAndIndexRootName + "_chunk_db"
AllowedTables = [Doc_db_name, Chunk_db_name]
Index_name = dbAndIndexRootName.lower() + "-index"
Embeddings_model_name = "text-embedding-3-small"
Embeddings_model = OpenAIEmbeddings(model=Embeddings_model_name)
# LLMInstructions = "You are a helpful assistant. Answer the question using only the information from the text in the database. Do not use any knowledge or understanding that you have from your training. Specifically do not use any knowledge that you have of the somewhat similar novel 'Pride and Prejudice' in to answer the question."
LLMInstructions = "You are a helpful assistant."
MAX_BATCH_EMBED_RETRIES = 3
INITIAL_BATCH_EMBED_BACKOFF = 1

# OpenAI GPT-4o Rate Limits (ensure these reflect your actual account limits)
GPT4O_TPM = 450000  # Tokens Per Minute
GPT4O_RPM = 5000   # Requests Per Minute
WORKER_SAFETY_FACTOR = 0.85 # Safety margin for API calls (e.g., 85%)
ANALYSIS_SAMPLE_SIZE = 7 # Number of blocks to sample for token estimation (Increased)
MAX_WORKERS_FALLBACK = 10 # Fallback worker count if sampling fails (Increased)
WORKER_RATE_LIMIT_DIVISOR = 6 # Divisor to convert RPM to concurrent workers (e.g., 6 for 10-second intervals)
ESTIMATED_OUTPUT_TOKEN_FRACTION = 0.2 # Estimated output tokens as a fraction of input tokens
RATE_LIMIT_SLEEP_SECONDS = 5 # Seconds to sleep after hitting a rate limit error
UseDebugMode = True
AllowUseOfTrainingKnowledge = True

if UseDebugMode:
    # Debugging prompt for additional information
    AllowUseOfTrainingKnowledge = False
initialPromptText = ""

if not AllowUseOfTrainingKnowledge:
    # Initial prompt for the LLM
    initialPromptText = "In answering this prompt do not use any knowledge or understanding from your training but only what you learn from the inputs which are supplied.\n\n"
if UseDebugMode:
    # Debugging prompt for additional information
    initialPromptText = "In answering this prompt do not use any knowledge or understanding from your training but only what you learn from the inputs which are supplied. Specifically do not use any thing you might know about the novel 'Pride and Prejudice' in your answer.\n\n"

StateStorageList = [
    StateStoragePoints.LargeBlockAnalysisCompleted,
    StateStoragePoints.IterativeAnalysisCompleted
]
# StateStorageList = []
RunCodeFrom = RunFromType.Start