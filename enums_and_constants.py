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

WordsForFinalAnalysisSummary = 200
# create a dictionary of different prompts to add in the final analysis section of the code for each document type
additions_to_reduce_prompt = {
    DocumentType.NOVEL: f"""Provide a detailed analysis of the novel of about {WordsForFinalAnalysisSummary} words for the "overall_summary" par of the output, including its themes, characters, and plot structure. For the structure, create one entry for each chapter and for each entry create a brief title. When listing characters, locations, and organizations, draw from both the BLOCK DATA and the ENTITY DATA provided below. You must include ALL characters, ALL locations, and ALL organizations mentioned in either the BLOCK DATA or the ENTITY_DATA. However it is very important to be aware that many entities may be referred to by multiple names — for example, a character like "Philip Jones" might also appear as "Mr. Jones", "Philip", or a nickname. Similarly, locations and organizations may be referenced in varied ways throughout the text. Your output MUST include all characters, locations, and organizations but each one should be listed only once, using the name by which it is most commonly referred to in the text. Avoid duplicating entries due to name variations. 
    Your process should be as follows:
    1. start by looking at all of the characters, locations, and organizations in the ENTITY DATA. 
    2. Next, look at the BLOCK DATA. As you go through this data you will come to know which of the characters, locations and organizations in the "ENTITY DATA" are duplicates refering to the same characters locations or organizations.
    For each character, location, and organization return the name by which that entity is most commonly referred to in the text. 
    Every character, location, and organization in the "ENTITY DATA" MUST be included in the output unless you believe it to be a duplicate or variartion of another name. If you believe it to be a duplicate or variation of another name then you should include only the name by which the entity is most commonly referred to in the text. If you believe that a name in the ENTITY DATA is not a duplicate or variation then it MUST be included in the output.
    """,
    DocumentType.BIOGRAPHY: "Please provide a detailed analysis of the biography, including its main themes and the life of the subject.",
    DocumentType.JOURNAL_ARTICLE: "Please provide a detailed analysis of the journal article, including its main arguments and contributions to the field."
}

