from enum import Enum
from globals import *
from UtilityFunctions import *
from DSAIParams import *

class DocumentType(Enum):
    NOVEL = "Novel"
    BIOGRAPHY = "Biography"
    JOURNAL_ARTICLE = "JournalArticle"

DocumentTypeList = ["Novel", "Biography", "Journal Article"]
    
# create a dictionary of different prompts to add in the final analysis section of the code for each document type
additions_to_reduce_prompt = {
    DocumentType.NOVEL: "Provide a detailed analysis of the novel, including its themes, characters, and plot structure. Insure that you create one structure element for each chapter. In listing characters, locations and organizations list all characters, locations and organizations in the novel but do not list twice if they are reffred to by different names, for example, 'Mr. Darcy' and 'Darcy' should be listed as one character. In each case list the charachter, location or organization by the most common way it is refrred to in the text. Additionally, use the provided 'Raw Consolidated Entities' list below as a guide. This list contains all mentions found across blocks, including duplicates and variations (e.g., 'Darcy', 'Mr. Darcy'). Your task is to intelligently consolidate these variations into single entries, listing the entity by its most common name or full name as appropriate based on the summaries",
    DocumentType.BIOGRAPHY: "Please provide a detailed analysis of the biography, including its main themes and the life of the subject.",
    DocumentType.JOURNAL_ARTICLE: "Please provide a detailed analysis of the journal article, including its main arguments and contributions to the field."
}

initialPromptText = ""
if AllowUseOfTrainingKnowledge:
    # Initial prompt for the LLM
    initialPromptText = "In answering this prompt do not use any knowledge or understanding from your training but only what you learn from the inputs which are supplied.\n\n"
if UseDebugMode:
    # Debugging prompt for additional information
    initialPromptText = "In answering this prompt do not use any knowledge or understanding from your training but only what you learn from the inputs which are supplied. Specifically do not use any thing you might know about the novel 'Pride and Prejudice' in your answer.\n\n"
