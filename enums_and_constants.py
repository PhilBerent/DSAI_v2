from enum import Enum
from globals import *
from UtilityFunctions import *



class DocumentType(Enum):
    NOVEL = "Novel"
    BIOGRAPHY = "Biography"
    JOURNAL_ARTICLE = "JournalArticle"
    
class DocumentType(Enum):
    NOVEL = "Novel"
    BIOGRAPHY = "Biography"
    JOURNAL_ARTICLE = "JournalArticle"

class CodeStages(Enum):
    Start = "Start"
    LargeBlockAnalysisCompleted = "LargeBlockAnalysisCompleted"
    ReduceAnalysisCompleted = "ReduceAnalysisCompleted"
    DetailedBlockAnalysisCompleted = "DetailedBlockAnalysisCompleted"
    EmbeddingsCompleted = "EmbeddingsCompleted"
    GraphAnalysisCompleted = "GraphAnalysisCompleted"

class StateStoragePoints(Enum):
    LargeBlockAnalysisCompleted = "LargeBlockAnalysisCompleted"
    ReduceAnalysisCompleted = "ReduceAnalysisCompleted"
    DetailedBlockAnalysisCompleted = "DetailedBlockAnalysisCompleted"
    EmbeddingsCompleted = "EmbeddingsCompleted"
    GraphAnalysisCompleted = "GraphAnalysisCompleted"

Code_Stages_List = [
    "Start",                          # Ingest document and perform coarse chunking
    "LargeBlockAnalysisCompleted",   # Map phase complete; ready for reduce phase
    "ReduceAnalysisCompleted",    # Reduce phase complete; ready for chunking
    "DetailedBlockAnalysisCompleted", # Detailed block analysis complete; ready for embeddings
    "EmbeddingsCompleted",            # Embeddings phase complete; ready for graph analysis
    "GraphAnalysisCompleted",         # Graph analysis complete; ready for storage
]

DocumentTypeList = ["Novel", "Biography", "Journal Article"]



    

