from enum import Enum
from globals import *
from UtilityFunctions import *


class DocumentType(Enum):
    NOVEL = "Novel"
    BIOGRAPHY = "Biography"
    JOURNAL_ARTICLE = "JournalArticle"

class DocumentType2(Enum):
    NOVEL = "Novel"
    BIOGRAPHY = "Biography"
    JOURNAL_ARTICLE = "JournalArticle"
# class RunFromType(Enum):
#     Start = "Start"
#     LargeBlockAnalysisCompleted = "LargeBlockAnalysisCompleted"
#     ReduceAnalysisCompleted = "ReduceAnalysisCompleted"

# class StateStoragePoints(Enum):
#     LargeBlockAnalysisCompleted = "LargeBlockAnalysisCompleted"
#     ReduceAnalysisCompleted = "ReduceAnalysisCompleted"

# DocumentTypeList = ["Novel", "Biography", "Journal Article"]
    
# # create a dictionary of different prompts to add in the final analysis section of the code for each document type
# additions_to_reduce_prompt = {
#     DocumentType.NOVEL: "Provide a detailed analysis of the novel, including its themes, characters, and plot structure. For the structure, create one entry for each chapter. When listing characters, locations, and organizations, draw from both the BLOCK DATA and the ENTITY DATA provided below. You must include all characters, locations, and organizations mentioned in either source. Many entities may be referred to by multiple names — for example, a character like Philip Jones might also appear as Mr. Jones, Philip, or a nickname. Similarly, locations and organizations may be referenced in varied ways throughout the text. Your output should include each character, location, and organization only once, using the name by which it is most commonly referred to in the text. Avoid duplicating entries due to name variations.",
#     DocumentType.BIOGRAPHY: "Please provide a detailed analysis of the biography, including its main themes and the life of the subject.",
#     DocumentType.JOURNAL_ARTICLE: "Please provide a detailed analysis of the journal article, including its main arguments and contributions to the field."
# }

