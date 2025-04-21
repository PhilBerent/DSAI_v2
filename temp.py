import spacy
import re
from typing import List, Dict, Tuple, Any
from globals import *
from UtilityFunctions import *
from DSAIParams import *
from DSAIUtilities import *

import spacy
import re
from typing import List, Dict, Tuple, Any

input = [{ 
    "level1": {
        "level2": {
            "level3": {
                "level4_key1": "value1",
                "level4_key2": 12345
            },
            "level3_key": ["a", "b", "c"]
        },
        "another_level2_key": True
    },
    "top_level_key": "done"
},
{
    "level1": {
        "level2": {
            "level3": [
                {
                    "level4_key1": "2233", 
                    "level4_key2": 67890
                },
                {
                    "level4_key1": "233", 
                    "level4_key2": 67890
                },
                {
                    "level4_key1": "22zzz33", 
                    "level4_key2": 67890
                },
                {
                    "level4_key1": "22â€33", 
                    "level4_key2": 67890
                }
            ],
            "level3_key": ["x", "y", "z"]
        },
        "another_level2_key": False
    },
    "top_level_key": "done"
}]

aa, bb = has_string(input, "zzz")
cc, dd = has_string(input, "â€")
d=5

testin="""{"document_type": "Novel", "structure": [{"type": "Chapter/Section", "title": "Conversation and First Impressions", "number": 1}, {"type": "Chapter/Section", "title": "Departure and Local Gossip", "number": 2}, {"type": "Chapter/Section", "title": "The Arrival of Mr. Collins", "number": 3}, {"type": "Chapter/Section", "title": "Mr. Collins's Visit and Lady Catherine", "number": 4}, {"type": "Chapter/Section", "title": "Mr. Collins's Intentions and an Encounter in Meryton", "number": 5}, {"type": "Chapter/Section", "title": "A Visit to Meryton and Wickham's Revelation", "number": 6}, {"type": "Chapter/Section", "title": "A Ball at Netherfield is Announced", "number": 7}, {"type": "Chapter/Section", "title": "The Netherfield Ball", "number": 8}, {"type": "Chapter/Section", "title": "Mr. Collins's Proposal", "number": 9}, {"type": "Chapter/Section", "title": "Babe Rejects Mr. Collins", "number": 10}, {"type": "Chapter/Section", "title": "Aftermath of the Proposal and News from London", "number": 11}, {"type": "Chapter/Section", "title": "Charlotte's Engagement to Mr. Collins", "number": 12}, {"type": "Chapter/Section", "title": "Reactions to Charlotte's Engagement", "number": 13}, {"type": "Chapter/Section", "title": "Bingley's Departure and Wickham's Influence", "number": 14}, {"type": "Chapter/Section", "title": "The Gardiners' Visit and Wickham's Charm", "number": 15}, {"type": "Chapter/Section", "title": "Mrs. Gardiner's Advice and Charlotte's Marriage", "number": 16}, {"type": "Chapter/Section", "title": "Arrival at the Parsonage and a Visit from Rosings", "number": 17}, {"type": "Chapter/Section", "title": "Dining at Rosings with Lady Catherine", "number": 18}], "overall_summary": "The novel centers around the social interactions and romantic pursuits of several characters in rural England. Babe Dingo and her sisters navigate societal expectations, familial pressures, and the pursuit of suitable marriages. The arrival of wealthy bachelors like Mr. Bingley and Mr. Darcy stirs the community, leading to various romantic entanglements and social clashes. Mr. Collins, a pompous clergyman, adds comedic relief and further complicates the marriage prospects of the Dingo sisters. The novel explores themes of social class, reputation, and the complexities of love and marriage, as characters grapple with societal expectations and personal desires. The plot unfolds through a series of balls, visits, and conversations, revealing the characters' personalities and motivations. Babe's evolving opinions of Darcy and Wickham, Kirsty's hopes for a connection with Mr. Bingley, and Charlotte Lucas's pragmatic decision to marry Mr. Collins drive the narrative forward, highlighting the diverse approaches to marriage and happiness within the constraints of their society.", "preliminary_key_entities": {"characters": ["Babe Dingo", "Bingley", "Caroline Bingley", "Catherine", "Charlotte Lucas", "Colonel Forster", "Darcy", "Denny", "Georgiana Darcy", "Hill", "Kirsty", "Lady Anne Darcy", "Lady Catherine de Bourgh", "Lizzy", "Louisa", "Maria Lucas", "Mimi", "Miss De Bourgh", "Miss King", "Mr. Bingley", "Mr. Collins", "Mr. Darcy", "Mr. Denny", "Mr. Fitzwilliam Darcy", "Mr. Gardiner", "Mr. Hurst", "Mr. Wickham", "Mrs. Dingo", "Mrs. Gardiner", "Mrs. Hurst", "Mrs. Jenkinson", "Mrs. Philips", "Nicholls", "Pamela", "Richard", "Sir Lewis de Bourgh", "Sir William Lucas", "Wickham", "uncle Philips"], "locations": ["Derbyshire", "Gracechurch Street", "Grosvenor Street", "Hertfordshire", "Hunsford", "Kent", "Lakes", "London", "Longbourn", "Lucas Lodge", "Meryton", "Netherfield", "Pemberley", "Rosings", "St. James\u2019s", "Westerham", "York"], "organizations": ["----shire", "Archbishop", "Church of England", "Pemberley House", "regiment"]}}"""

cleaned_text = cleanLLMOutput(testin)
a=5