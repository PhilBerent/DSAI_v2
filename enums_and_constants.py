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

class RunFromType(Enum):
    Start = "Start"
    LargeBlockAnalysisCompleted = "LargeBlockAnalysisCompleted"
    IterativeAnalysisCompleted = "IterativeAnalysisCompleted"

class StateStoragePoints(Enum):
    LargeBlockAnalysisCompleted = "LargeBlockAnalysisCompleted"
    IterativeAnalysisCompleted = "IterativeAnalysisCompleted"


DocumentTypeList = ["Novel", "Biography", "Journal Article"]



    

