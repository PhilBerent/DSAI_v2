import logging
import time
import os
import uuid
import sys
from typing import List, Dict, Any, Optional
import re
import copy
import re


# Adjust path to import from parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from globals import *
from UtilityFunctions import *
from DSAIParams import * 
from DSAIUtilities import *
from enums_constants_and_classes import CodeStages, StateStoragePoints, Code_Stages_List
from itertools import combinations  
from nicknames import NickNamer

nmr = NickNamer()

TITLE_LIST = ["Mr.", "Mrs.", "Ms.", "Miss.", "M.", "Mr. And Mrs.", "Mr. & Mrs.", "Dr.", "Prof.", "Sir", "Lady", "Lord", "Madam", "Dame"]
# create a list of military titles and their abbreviations just as a list not a dictionary
MILITARY_TITLES = ["General", "Colonel", "Major", "Captain", "Commander", "Lieutenant", "Ensign", "Admiral", "Commodore", "Midshipman", "Sergeant", "Corporal", "Specialist", "Private", "Gen.", "Col.", "Maj.", "Capt.", "Cmdr.", "Lt.", "Ens.", "Adm.", "Cdre.", "Midn.", "Sgt.", "Cpl.", "Spc.", "Pvt."]
FULL_TITLE_LIST = TITLE_LIST + MILITARY_TITLES 
ABREVIATION_TITLE_LIST = ["Mr", "Mrs", "Ms", "Miss", "M", "Dr", "Prof", "Gen", "Col", "Maj", "Capt", "Cmdr", "Lt", "Ens", "Adm", "Cdre", "Midn", "Sgt", "Cpl", "Spc", "Pvt", "Mr And Mrs", "Mr & Mrs"] 
MALE_TITLE_LIST = ["Mr.", "Sir", "Lord", "Dame", "Gen.", "Col.", "Maj.", "Capt.", "Cmdr.", "Lt.", "Ens.", "Adm.", "Cdre.", "Midn."]
FEMALE_TITLE_LIST = ["Mrs.", "Ms.", "Miss.", "Ms.", "Lady", "Madam", "Dame", "Gen.", "Col.", "Maj.", "Capt.", "Cmdr.", "Lt.", "Ens.", "Adm.", "Cdre.", "Midn."]
FOLLOWED_BY_FIRST_NAME_TITLES = ["Sir", "Lady", "Dame", "Brother", "Sister", "Father", "Pastor", "Rabbi", "Imam", "Reverend", "Rev.", "Saint"]
NAME_QUALIFIERS = {"Von", "Van", "De", "Del", "Di", "Da", "Le", "La", "El", "Al", "Mac", "Mc"}

SUFFIX_LIST = ["Jr.", "Sr.", "II", "III", "IV", "V", "PhD", "MD"]

addedCanToNickDict = {
    "jane": set(["jane", "janie", "janey"]),
    "philip": set(["phil", "philly", "pho-pho"]),
    #db
    "kirsty": set(["kirsty", "kirst", "kirsten", "jane"]),
    "babe": set(["elizabeth"]),
    "mimi": set(["mary"]),
    "catherine": set(["helen"]),
    "doggy": set(["kitty"]),
    "pamela": set(["lydia"]),
    "elizabeth": set(["babe"]),
    #ed
}
addedNickToCanDict = {
    "pho-pho": set(["philip"]),
    #ed
}

class Gender(Enum):
    MALE = "Male"
    FEMALE = "Female"
    UNKNOWN = "Unknown"
    
class MatchTest(Enum):
    MATCH = "Match"
    NO_MATCH = "No Match"
    UNKNOWN = "Unknown"


class NameDetails:
    def __init__(self, name):
        self.name = name
        self.title = ""
        self.name_no_title = ""
        self.first_name = ""
        self.middle_names = ""
        self.last_name = ""
        self.first_and_last_name = ""
        self.suffix = ""

        if not name.strip():
            return

        words = name.strip().split()

        # Check for suffix
        if words and words[-1] in SUFFIX_LIST:
            self.suffix = words[-1]
            words = words[:-1]

        # Check for title
        if words and words[0] in FULL_TITLE_LIST:
            self.title = words[0]
            words = words[1:]

        # Build name_no_title
        self.name_no_title = " ".join(words)
        numWords = len(words)

        # Parse name parts
        if numWords == 1:
            if self.title:
                if self.title in FOLLOWED_BY_FIRST_NAME_TITLES:
                    self.first_name = words[0]
                else:
                    self.last_name = words[0]
        elif numWords >= 2:
            poss_qualifer = words[numWords-2]
            if poss_qualifer in NAME_QUALIFIERS:
                self.last_name = words[-2] + " " + words[-1]
                end_middle = -2
                if numWords > 2:
                    self.first_name = words[0]
            else:
                self.last_name = words[-1]
                end_middle = -1
                self.first_name = words[0]
            if self.first_name != "":
                self.middle_names = " ".join(words[1:end_middle])

        if self.first_name and self.last_name:
            self.first_and_last_name = f"{self.first_name} {self.last_name}"
        self.has_first_name = self.first_name != ""
        self.has_last_name = self.last_name != ""
        self.has_title = self.title != ""
        self.has_middle_name = self.middle_names != ""
        self.has_first_and_last_name = self.first_and_last_name != ""
        self.has_suffix = self.suffix != ""
        self.has_title_first_and_last_name = self.has_title and self.has_first_and_last_name

    def as_dict(self):
        return {
            "input_name": self.name,
            "title": self.title,
            "name_no_title": self.name_no_title,
            "first_name": self.first_name,
            "middle_names": self.middle_names,
            "last_name": self.last_name,
            "first_and_last_name": self.first_and_last_name,
            "suffix": self.suffix
        }
    
def selectBestName(nameDetails1: NameDetails, nameDetails2: NameDetails, blockCount1, blockCount2):
    if nameDetails1.has_title_first_and_last_name and not nameDetails2.has_title_first_and_last_name:
        return nameDetails1, 1
    elif not nameDetails1.has_title_first_and_last_name and nameDetails2.has_title_first_and_last_name:
        return nameDetails2, 2
    elif nameDetails1.has_first_and_last_name and not nameDetails2.has_first_and_last_name:
        return nameDetails1, 1
    elif not nameDetails1.has_first_and_last_name and nameDetails2.has_first_and_last_name:
        return nameDetails2, 2
    elif blockCount1 > blockCount2:
        return nameDetails1, 1
    else:
        return nameDetails2, 2
        
def getEntityNames(entityData, entityType) -> List[str]:
    entityNames = []
    for entity in entityData[entityType]:
        entityNames.append(entity['name'])
    return entityNames  

def addCharNameToCombos(nameDetails: NameDetails):
    hasTitle = nameDetails["title"] != ""
    hasFirstName = nameDetails["first_name"] != ""
    hasLastName = nameDetails["last_name"] != ""
    hasMiddleName = nameDetails["middle_names"] != ""
    if hasTitle and hasLastName and not hasFirstName and not hasMiddleName:
        return False
    else:
        return True

def checkConsistency(name1: NameDetails, name2: NameDetails):
    name1FirstName = name1.first_name
    name1LastName = name1.last_name
    name1Title = name1.title
    name2FirstName = name2.first_name
    name2LastName = name2.last_name
    name2Title = name2.title
    name1FirstAndLastName = name1.first_and_last_name
    name2FirstAndLastName = name2.first_and_last_name

    name1hasFirstName = name1FirstName != ""
    name1hasLastName = name1LastName != ""
    name2hasFirstName = name2FirstName != ""
    name2hasLastName = name2LastName != ""
    name1hasTitle = name1Title != ""
    name2hasTitle = name2Title != ""

    name1Gender = getGender(name1)
    name2Gender = getGender(name2)

    if name1Gender != Gender.UNKNOWN and name2Gender != Gender.UNKNOWN:
        if name1Gender != name2Gender:
            return False

    if name1FirstAndLastName and name2FirstAndLastName:
        if not matchFirstName(name1FirstName, name2FirstName):
            return False

def getGender(name: NameDetails):
    title = name.title
    if title == "":
        return Gender.UNKNOWN
    elif title in MALE_TITLE_LIST:
        return Gender.MALE
    elif title in FEMALE_TITLE_LIST:
        return Gender.FEMALE
    else:
        return Gender.UNKNOWN

def isMale(name: NameDetails):
    title = name.title
    if title == "":
        return Answer.DONT_KNOW
    elif title in MALE_TITLE_LIST:
        return Answer.YES
    elif title in FEMALE_TITLE_LIST:
        return Answer.NO
    else:
        return Answer.DONT_KNOW

def isFemale(name: NameDetails):
    title = name.title
    if title == "":
        return Answer.DONT_KNOW
    elif title in FEMALE_TITLE_LIST:
        return Answer.YES
    elif title in MALE_TITLE_LIST:
        return Answer.NO
    else:
        return Answer.DONT_KNOW

FalseMatches = set({("caroline", "charles")})

def isFalseMatch(name1: str, name2: str) -> bool:
    name1 = name1.strip().lower()
    name2 = name2.strip().lower()
    if name1 == name2:
        return False
    if (name1, name2) in FalseMatches or (name2, name1) in FalseMatches:
        return True
    return False

def matchFirstName(name1: str, name2: str) -> bool:
    name1 = name1.strip().lower()
    name2 = name2.strip().lower()

    if isFalseMatch(name1, name2):
        return False

    if name1 == name2:
        return True

    # Get known nicknames and canonicals from built-in library
    n1Canonicals = nmr.canonicals_of(name1)
    n1Nicknames = nmr.nicknames_of(name1)
    n1All = n1Canonicals | n1Nicknames
    n1All.add(name1)    
    n1List = list(n1All)
    for n1 in n1List:
        toAdd = addedCanToNickDict.get(n1, set())
        n1All |= toAdd
        toAdd = addedNickToCanDict.get(n1, set())
        n1All |= toAdd        
 
    n2Canonicals = nmr.canonicals_of(name2)
    n2Nicknames = nmr.nicknames_of(name2)
    n2All = n2Canonicals | n2Nicknames
    n2List = list(n2All)
    n2All.add(name2)    
    for n2 in n2List:
        toAdd = addedCanToNickDict.get(n2, set())
        n2All |= toAdd
        toAdd = addedNickToCanDict.get(n2, set())
        n2All |= toAdd
         
    ret = not n1All.isdisjoint(n2All)
    
    return ret

def can_reject_match(nameDetails: NameDetails) -> bool:
    ret = True
    if getGender(nameDetails) == Gender.UNKNOWN and not nameDetails.has_first_and_last_name:
        ret = False
    return ret

def names_match(name1: NameDetails, name2: NameDetails) -> MatchTest:
    # Normalize fields to lowercase
    title1 = name1.title.lower()
    title2 = name2.title.lower()
    firstName1 = name1.first_name.lower()
    firstName2 = name2.first_name.lower()
    lastName1 = name1.last_name.lower()
    lastName2 = name2.last_name.lower()
    middleName1 = name1.middle_names.lower()
    middleName2 = name2.middle_names.lower()

    # Presence flags
    hasTitle1 = title1 != ""
    hasTitle2 = title2 != ""
    hasFirstName1 = firstName1 != ""
    hasFirstName2 = firstName2 != ""
    hasLastName1 = lastName1 != ""
    hasLastName2 = lastName2 != ""
    hasMiddleName1 = middleName1 != ""
    hasMiddleName2 = middleName2 != ""
    hasFirstAndLastName1 = hasFirstName1 and hasLastName1
    hasFirstAndLastName2 = hasFirstName2 and hasLastName2

    # Gender check
    gender1 = getGender(name1)
    gender2 = getGender(name2)
    if gender1 != Gender.UNKNOWN and gender2 != Gender.UNKNOWN:
        if gender1 != gender2:
            return MatchTest.NO_MATCH
    
    if hasLastName1 and hasLastName2 and gender1 == Gender.MALE and gender2 == Gender.MALE:
        if lastName1 != lastName2:
            return MatchTest.NO_MATCH

    # Match on first and last name
    if hasFirstAndLastName1 and hasFirstAndLastName2:
        if lastName1 == lastName2:
            if matchFirstName(firstName1, firstName2):
                return MatchTest.MATCH
            else:
                return MatchTest.NO_MATCH

    return MatchTest.UNKNOWN

def fix_titles_in_names(block_analysis_result):
    fixed_result = copy.deepcopy(block_analysis_result)
    
    characters = fixed_result.get('key_entities_in_block', {}).get('characters', [])
    
    for character in characters:
        if 'name' in character:
            character['name'] = fix_title_abbreviations(character['name'])
        
        if 'alternate_names' in character and isinstance(character['alternate_names'], list):
            character['alternate_names'] = [fix_title_abbreviations(alt_name) for alt_name in character['alternate_names']]

    return fixed_result


def fix_title_abbreviations(text):
    for title in ABREVIATION_TITLE_LIST:
        # Only match the title if not already followed by a period
        pattern = rf'\b{re.escape(title)}(?!\.)\s'
        replacement = f'{title}. '
        text = re.sub(pattern, replacement, text)
    return text


