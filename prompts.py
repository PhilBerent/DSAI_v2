
def getNovelReducePrompt(NumBlocks, BlocksPerStructUnit=1, WordsForFinalAnalysisSummary=200):
    if BlocksPerStructUnit > 1:
        numStrucElements = NumBlocks // BlocksPerStructUnit
        # calc remainder
        remainder = NumBlocks % BlocksPerStructUnit
        lastElementLength = BlocksPerStructUnit + remainder
        remainderText = f" apart from the last element which will cover the last {lastElementLength} Blocks. " if remainder > 0 else "."
        strucElementsText = f"For the structure output create one entry for each {numStrucElements} Block (which are chapters or other distinct elements of the novel). As there are {NumBlocks} Blocks in total, create{numStrucElements} entries in total each one covering {BlocksPerStructUnit} Blocks{remainderText} "
    else :
       strucElementsText = f"For the structure output create one entry for each Block (which are chapters or other distinct elements of the novel). As there are {NumBlocks} Blocks in total, create {NumBlocks} entries in total. "
    strucElementsText += """ For each entry, assign a brief, descriptive title reflecting the main event or topic of that block summary (e.g., "Introduction of Character X", "The Climax at Location Y", "A Turning Point"). Use the block number for the number field."""
        
    ret = f"""
Overall Summary: Provide a detailed analysis of the novel (approx. {WordsForFinalAnalysisSummary}) for the overall_summary output. This should cover its main themes, key characters, and plot structure based on the provided Block Data summaries.

Structure Output: {strucElementsText}

Preliminary Key Entities (Characters, Locations, Organizations) Output: 
Your primary goal here is to identify the unique entities represented in the ENTITY DATA list and output a clean, deduplicated list for preliminary_key_entities.
Source: Use the ENTITY DATA as your sole source list for potential entities. The BLOCK DATA summaries should only be used as context to help you understand if different names in the ENTITY DATA refer to the same entity.
Task: Examine the lists of characters, locations, and organizations within ENTITY DATA. Identify all entries that are variations referring to the same underlying entity. For example, "Philip Jones", "Mr. Jones", "Jones", and "Philip" might all refer to one character.
Output Rule: For each unique entity identified, select only one canonical name to include in the final output lists (characters, locations, organizations).
Choosing the Canonical Name: Prefer the most complete or formal version of the name found in the ENTITY DATA for that entity (e.g., select "Professor Albus Dumbledore" if "Dumbledore" and "Albus" also appear for the same person). If multiple equally complete forms exist, choose the one that appears first or seems most representative based on the context from BLOCK DATA summaries.
No Variations: The final lists in preliminary_key_entities must contain only these single, canonical names. Do not include multiple variations of the same entity's name. Every unique entity concept represented by one or more names in the ENTITY DATA should appear exactly once in the output under its chosen canonical name.
"""
    
    return ret
    
def getBiographyReducePrompt(NumBlocks, BlocksPerStructUnit=1, WordsForFinalAnalysisSummary=200):
    ret = "Please provide a detailed analysis of the biography, including its main themes and the life of the subject."
    return ret

def getJournalArticleReducePrompt(NumBlocks, BlocksPerStructUnit=1, WordsForFinalAnalysisSummary=200):
    ret = "Please provide a detailed analysis of the journal article, including its main arguments and contributions to the field."
    return ret
