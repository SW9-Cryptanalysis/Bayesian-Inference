import math
import os

class DictionaryLanguageModel:
    """
    Scores plaintext hypotheses based on a word dictionary.
    
    This model assigns a score P_word(p) based on the percentage of
    space-delimited words in the text `p` that exist in a known
    word list.
    """
    
    def __init__(self, word_list_path: str):
        """
        Initializes the model by loading the word list.
        
        Args:
            word_list_path: Path to the text file containing the
                            word list (one word per line).
        """
        self.word_set = self.load_word_list(word_list_path)

    def load_word_list(self, path: str) -> set:
        """Loads the word list into a set for O(1) lookups.
        
        Args:
            path: Path to the word list file.
            
        Returns:
            set: A set of words loaded from the file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Word list file not found: {path}")
            
        word_set = set()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split on whitespace and take only the first part (the word)
                # This handles formats like "word 12345" or just "word"
                parts = line.split()
                if parts:
                    word = parts[0].lower()  # Store words in lowercase
                    word_set.add(word)
        return word_set

    def log_score_text(self, text: str) -> float:
        """
        Calculates the log-score for a given plaintext.
        
        The score scales with text length to match n-gram model behavior.
        For each word: log(P_valid) if valid, log(P_invalid) if invalid
        where P_valid and P_invalid are empirically chosen probabilities.
        
        Args:
            text: The space-segmented plaintext hypothesis (e.g., "I LIKE KILLING")
            
        Returns:
            float: The log-score (negative number, less negative is better).
        """
        words = text.split()
        
        if not words:
            return -float('inf')

        valid_word_count = 0
        for word in words:
            if word in self.word_set:
                valid_word_count += 1
        
        invalid_word_count = len(words) - valid_word_count
        
        log_p_valid = -10.0      # log(P) for valid word
        log_p_invalid = -50.0    # log(P) for invalid word
        
        total_log_prob = (valid_word_count * log_p_valid) + \
                        (invalid_word_count * log_p_invalid)
        
        return total_log_prob