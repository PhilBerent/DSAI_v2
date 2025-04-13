from enum import Enum
from globals import *
from UtilityFunctions import *

class DocumentType(Enum):
    NOVEL = "Novel"
    BIOGRAPHY = "Biography"
    JOURNAL_ARTICLE = "JournalArticle"

DocumentTypeList = ["Novel", "Biography", "Journal Article"]
    
# create a dictionary of different prompts to add in the final analysis section of the code for each document type
additions_to_reduce_prompt = {
    DocumentType.NOVEL: "Provide a detailed analysis of the novel, including its themes, characters, and plot structure. Insure that you create one structure element for each chapter. In listing characters, locations and organizations list all characters, locations and organizations in the novel but do not list twice if they are reffred to by different names, for example, 'Mr. Darcy' and 'Darcy' should be listed as one character. In each case list the charachter, location or organization by the most common way it is refrred to in the text.",
    DocumentType.BIOGRAPHY: "Please provide a detailed analysis of the biography, including its main themes and the life of the subject.",
    DocumentType.JOURNAL_ARTICLE: "Please provide a detailed analysis of the journal article, including its main arguments and contributions to the field."
}