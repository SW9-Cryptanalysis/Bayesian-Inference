import logging
import math
import os

logger = logging.getLogger(__name__)

class DictionaryLanguageModel:
    """Scores plaintext hypotheses based on a unigram word model.
    
    This model assigns a score P_word(p) based on word frequencies,
    computing the total log probability as the sum of log probabilities
    of all words in the text.
    """

    def __init__(self, word_list_path: str):
        """Initializes the model by loading the word list with frequencies.
        
        Args:
            word_list_path: Path to the text file containing the
                            word list (one word per line with frequency counts).

        """
        self.word_counts, self.total_word_tokens = self.load_word_list(word_list_path)
        self.log_probs = self._compute_log_probs()
        self.log_prob_unknown = math.log(1 / (self.total_word_tokens * 10))

    def load_word_list(self, path: str) -> tuple:
        """Loads the word list with frequencies.
        
        Args:
            path: Path to the word list file.
            
        Returns:
            tuple: A tuple containing (word_counts dict, total_word_tokens int).

        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Word list file not found: {path}")

        word_counts = {}
        total_tokens = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split on whitespace: "word count"
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0].lower()  # Store words in lowercase
                    try:
                        count = int(parts[1])
                        word_counts[word] = count
                        total_tokens += count
                    except ValueError:
                        logger.warning(f"Invalid count for word '{parts[0]}': {parts[1]}")
                        continue
                elif len(parts) == 1:
                    # If no count is provided, assume count of 1
                    word = parts[0].lower()
                    word_counts[word] = 1
                    total_tokens += 1

        return word_counts, total_tokens

    def _compute_log_probs(self) -> dict:
        """Computes log probabilities for all words in the dictionary.
        
        Returns:
            dict: A dictionary mapping words to their log probabilities.

        """
        log_probs = {}
        for word, count in self.word_counts.items():
            log_probs[word] = math.log(count / self.total_word_tokens)

        logger.info(f"Computed log probabilities for {len(log_probs)} words")
        return log_probs

    def log_score_text(self, text: str) -> float:
        """Calculates the total log-score from the unigram word model,
        normalized by the total number of CHARACTERS in the text.
        
        This normalization puts it on the same "per-character" scale
        as the n-gram model.
        """
        words = text.split()

        if not words:
            return -float("inf")

        total_dict_log_score = 0.0
        for word in words:
            # Get the log-prob for the word, or the "unknown" prob if not found
            word_lower = word.lower()
            if word_lower in self.log_probs:
                total_dict_log_score += self.log_probs[word_lower]
                # logger.debug(f'Valid word found: {word} (log_prob: {self.log_probs[word_lower]:.4f})')
            else:
                total_dict_log_score += self.log_prob_unknown
                # logger.debug(f'Unknown word: {word} (log_prob: {self.log_prob_unknown:.4f})')

        # Normalize by the CHARACTER length of the text (including spaces).
        # This puts it on the same scale as the n-gram model
        # (avg log prob per character).
        # We use max(1, len(text)) to avoid ZeroDivisionError for empty strings.
        return total_dict_log_score / max(1, len(text))
