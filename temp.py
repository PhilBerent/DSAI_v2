import spacy
import re
from typing import List, Dict, Tuple, Any
from globals import *
from UtilityFunctions import *
from DSAIParams import *

import spacy
import re
from typing import List, Dict, Tuple, Any

class DocumentProcessor:
    def __init__(self, model_name="en_core_web_sm"):
        # Load NLP model for entity recognition and text analysis
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Using blank model with limited functionality.")
            self.nlp = spacy.blank("en")
        
        # Set up for dialogue detection
        self.dialogue_patterns = [
            re.compile('"[^"]+"'),                      # Double quotes
            re.compile('\'[^\']+\''),                   # Single quotes
            re.compile('\u201c[^\u201d]+\u201d'),       # Smart double quotes (Unicode)
            re.compile('\u2018[^\u2019]+\u2019')        # Smart single quotes (Unicode)
        ]
    
    def process_novel(self, file_path: str) -> Dict[str, Any]:
        """Process a novel text file into structured components."""
        # Read the full text
        with open(DocToAddPath, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Extract structure
        books = self.identify_books(text)
        chapters = self.identify_chapters(text)
        scenes = self.identify_scenes(chapters)
        
        # Extract entities
        characters = self.extract_characters(text)
        locations = self.extract_locations(text)
        
        # Create the document structure
        document = {
            'full_text': text,
            'books': books,
            'chapters': chapters,
            'scenes': scenes,
            'characters': characters,
            'locations': locations
        }
        
        return document
    
    def identify_books(self, text: str) -> List[Dict[str, Any]]:
        """Identify book boundaries in the text."""
        # Detect book markers using regex - adapt for your specific novel format
        book_pattern = re.compile(r'BOOK [IVXLCDM0-9]+', re.IGNORECASE)
        books = []
        
        # Find all book headings
        matches = list(book_pattern.finditer(text))
        
        # If no books found, treat entire text as one book
        if not matches:
            books.append({
                'name': 'Book 1',
                'start_pos': 0,
                'end_pos': len(text),
                'text': text
            })
            return books
        
        # Process each book
        for i, match in enumerate(matches):
            start = match.start()
            # If not the last book, end is the start of the next book
            end = matches[i+1].start() if i < len(matches)-1 else len(text)
            
            book_text = text[start:end]
            book_name = book_text.split('\n')[0].strip()
            
            books.append({
                'name': book_name,
                'start_pos': start,
                'end_pos': end,
                'text': book_text
            })
        
        return books
    
    def identify_chapters(self, text: str) -> List[Dict[str, Any]]:
        """Identify chapter boundaries in the text."""
        # Detect chapter markers using regex
        chapter_pattern = re.compile(r'CHAPTER [IVXLCDM0-9]+', re.IGNORECASE)
        chapters = []
        
        # Find all chapter headings
        matches = list(chapter_pattern.finditer(text))
        
        # If no chapters found, try alternative patterns
        if not matches:
            # Try roman numerals alone
            chapter_pattern = re.compile(r'\n\s*[IVXLCDM]+\s*\n', re.IGNORECASE)
            matches = list(chapter_pattern.finditer(text))
            
            # If still no chapters, treat as one chapter
            if not matches:
                chapters.append({
                    'name': 'Chapter 1',
                    'start_pos': 0,
                    'end_pos': len(text),
                    'text': text
                })
                return chapters
        
        # Process each chapter
        for i, match in enumerate(matches):
            start = match.start()
            # If not the last chapter, end is the start of the next chapter
            end = matches[i+1].start() if i < len(matches)-1 else len(text)
            
            chapter_text = text[start:end]
            chapter_name = chapter_text.split('\n')[0].strip()
            
            chapters.append({
                'name': chapter_name,
                'start_pos': start,
                'end_pos': end,
                'text': chapter_text,
                'scenes': []  # Will be populated by identify_scenes
            })
        
        return chapters
    
    def identify_scenes(self, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify scene boundaries within chapters."""
        all_scenes = []
        scene_id = 0
        
        for chapter in chapters:
            chapter_text = chapter['text']
            chapter_scenes = []
            
            # Identify potential scene breaks
            # Look for combinations of:
            # 1. Multiple newlines
            # 2. Time transitions (e.g., "Later that day", "The next morning")
            # 3. Location changes
            paragraphs = re.split(r'\n\s*\n', chapter_text)
            
            current_scene_start = 0
            current_scene_text = []
            scene_markers = ["later", "next day", "morning", "evening", "night", 
                           "afternoon", "hour", "meanwhile", "elsewhere", "at the same time"]
            
            # Process paragraphs to identify scenes
            for i, para in enumerate(paragraphs):
                is_scene_break = False
                
                # Check if paragraph starts with a potential scene transition
                para_lower = para.lower().strip()
                if i > 0 and any(para_lower.startswith(marker) for marker in scene_markers):
                    is_scene_break = True
                
                # Check for location shift (crude approximation)
                if i > 0 and re.search(r'\b(at|in|on)\s+the\s+[a-z]+', para_lower):
                    is_scene_break = True
                
                if is_scene_break and current_scene_text:
                    # End previous scene
                    scene_text = "\n\n".join(current_scene_text)
                    scene_offset = chapter['start_pos'] + chapter_text.find(scene_text)
                    
                    chapter_scenes.append({
                        'id': scene_id,
                        'chapter_id': chapters.index(chapter),
                        'start_pos': scene_offset,
                        'end_pos': scene_offset + len(scene_text),
                        'text': scene_text
                    })
                    scene_id += 1
                    
                    # Start new scene
                    current_scene_text = [para]
                else:
                    current_scene_text.append(para)
            
            # Add the last scene if there's any content
            if current_scene_text:
                scene_text = "\n\n".join(current_scene_text)
                scene_offset = chapter['start_pos'] + chapter_text.find(scene_text)
                
                chapter_scenes.append({
                    'id': scene_id,
                    'chapter_id': chapters.index(chapter),
                    'start_pos': scene_offset,
                    'end_pos': scene_offset + len(scene_text),
                    'text': scene_text
                })
                scene_id += 1
            
            # Add scenes to chapter
            chapter['scenes'] = chapter_scenes
            all_scenes.extend(chapter_scenes)
        
        return all_scenes
    
    def extract_characters(self, text: str) -> List[Dict[str, Any]]:
        """Extract character entities from the text."""
        # For large texts, process in chunks
        max_length = 1000000  # Process 1M characters at a time
        characters = {}
        
        for i in range(0, len(text), max_length):
            chunk = text[i:i+max_length]
            doc = self.nlp(chunk)
            
            # Extract person entities
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text
                    if name not in characters:
                        characters[name] = {
                            'name': name,
                            'mentions': [],
                            'count': 0
                        }
                    
                    characters[name]['mentions'].append(i + ent.start_char)
                    characters[name]['count'] += 1
        
        # Filter out likely false positives
        filtered_characters = {}
        for name, data in characters.items():
            # Skip single-word names with few mentions (likely false positives)
            if ' ' not in name and data['count'] < 3:
                continue
            filtered_characters[name] = data
        
        # Convert to list and sort by frequency
        char_list = list(filtered_characters.values())
        char_list.sort(key=lambda x: x['count'], reverse=True)
        
        # Take top 50 characters or fewer if not enough
        return char_list[:50]
    
    def extract_locations(self, text: str) -> List[Dict[str, Any]]:
        """Extract location entities from the text."""
        # For large texts, process in chunks
        max_length = 1000000  # Process 1M characters at a time
        locations = {}
        
        for i in range(0, len(text), max_length):
            chunk = text[i:i+max_length]
            doc = self.nlp(chunk)
            
            # Extract location entities
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC", "FAC"]:
                    name = ent.text
                    if name not in locations:
                        locations[name] = {
                            'name': name,
                            'mentions': [],
                            'count': 0
                        }
                    
                    locations[name]['mentions'].append(i + ent.start_char)
                    locations[name]['count'] += 1
        
        # Filter out locations with few mentions
        filtered_locations = {name: data for name, data in locations.items() 
                             if data['count'] >= 2}
        
        # Convert to list and sort by frequency
        loc_list = list(filtered_locations.values())
        loc_list.sort(key=lambda x: x['count'], reverse=True)
        
        return loc_list

# Example usage
if __name__ == "__main__":
    docProcessor = DocumentProcessor()
    document = docProcessor.process_novel(DocToAddPath)
    
    # Print some statistics
    print(f"Processed document with:")
    print(f"- {len(document['books'])} books")
    print(f"- {len(document['chapters'])} chapters")
    print(f"- {len(document['scenes'])} scenes")
    print(f"- {len(document['characters'])} characters")
    print(f"- {len(document['locations'])} locations")


    document = docProcessor.process_novel(DocToAddPath)
    a=1