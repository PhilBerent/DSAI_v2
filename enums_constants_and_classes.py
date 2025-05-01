from enum import Enum
from globals import *
from UtilityFunctions import *
import collections
from typing import List, Dict, TypedDict # Make sure TypedDict is imported

TITLE_LIST = ["Mr.", "Mrs.", "Ms.", "Miss.", "M.", "Dr.", "Prof.", "Sir", "Lady", "Lord", "Madam", "Dame"]
# create a list of military titles and their abbreviations just as a list not a dictionary
MILITARY_TITLES = ["General", "Colonel", "Major", "Captain", "Commander", "Lieutenant", "Ensign", "Admiral", "Commodore", "Midshipman", "Sergeant", "Corporal", "Specialist", "Private", "Gen.", "Col.", "Maj.", "Capt.", "Cmdr.", "Lt.", "Ens.", "Adm.", "Cdre.", "Midn.", "Sgt.", "Cpl.", "Spc.", "Pvt."]
FULL_TITLE_LIST = TITLE_LIST + MILITARY_TITLES 
ABREVIATION_TITLE_LIST = ["Mr", "Mrs", "Ms", "Miss", "M", "Dr", "Prof", "Gen", "Col", "Maj", "Capt", "Cmdr", "Lt", "Ens", "Adm", "Cdre", "Midn", "Sgt", "Cpl", "Spc", "Pvt"] 

    
SUFFIX_LIST = ["Jr.", "Sr.", "II", "III", "IV", "V", "PhD", "MD"]

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

# --- Schema Definitions (as defined before) ---
class AlternateNameInfo(TypedDict):
    alternate_name: str
    block_list: List[int]

class DescriptionInfo(TypedDict):
    description: str
    block_list: List[int]

class EntityDetails(TypedDict):
    block_list: List[int]
    alternate_names: List[AlternateNameInfo]
    descriptions: List[DescriptionInfo]

ConsolidatedEntitiesInCategory = Dict[str, EntityDetails]

class ConsolidatedOutputSchema(TypedDict):
    characters: ConsolidatedEntitiesInCategory
    locations: ConsolidatedEntitiesInCategory
    organizations: ConsolidatedEntitiesInCategory

