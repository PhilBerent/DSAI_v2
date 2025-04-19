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
    IterativeAnalysisCompleted = "IterativeAnalysisCompleted"
    DetailedBlockAnalysisCompleted = "DetailedBlockAnalysisCompleted"

class StateStoragePoints(Enum):
    LargeBlockAnalysisCompleted = "LargeBlockAnalysisCompleted"
    IterativeAnalysisCompleted = "IterativeAnalysisCompleted"
    DetailedBlockAnalysisCompleted = "DetailedBlockAnalysisCompleted"

Code_Stages_List = [
    "Start",                          # Ingest document and perform coarse chunking
    "LargeBlockAnalysisCompleted",   # Map phase complete; ready for reduce phase
    "IterativeAnalysisCompleted",    # Reduce phase complete; ready for chunking
]

DocumentTypeList = ["Novel", "Biography", "Journal Article"]



    

