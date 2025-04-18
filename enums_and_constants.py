from enum import Enum
from globals import *
from UtilityFunctions import *
from DSAIParams import *


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

class StateStoragePoints(Enum):
    LargeBlockAnalysisCompleted = "LargeBlockAnalysisCompleted"
    IterativeAnalysisCompleted = "IterativeAnalysisCompleted"

class CodeStagesx(Enum):
    Start = "Start"
    LargeBlockAnalysisCompleted = "LargeBlockAnalysisCompleted"
    IterativeAnalysisCompleted = "IterativeAnalysisCompleted"

Code_Stages_List = [
    "Start",                          # Ingest document and perform coarse chunking
    "LargeBlockAnalysisCompleted",   # Map phase complete; ready for reduce phase
    "IterativeAnalysisCompleted",    # Reduce phase complete; ready for chunking
]

DocumentTypeList = ["Novel", "Biography", "Journal Article"]



    

