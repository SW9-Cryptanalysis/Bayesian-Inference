"""
Efficient incremental scoring using the exchangeability property.

When sampling at a position, we only need to rescore the affected context window
rather than the entire plaintext. This dramatically speeds up sampling for long texts.
"""

import math
import logging
from typing import List, Tuple, Set
from nltk.lm.preprocessing import pad_both_ends
from lm_models.crp_source_model import CRPSourceModel
from lm_models.crp_channel_model import CRPChannelModel

logger = logging.getLogger(__name__)


class IncrementalScorer:
    """Handles efficient incremental scoring for CRP models.
    
    Uses the exchangeability property: when updating positions, pretend
    the affected region occurs at the end of the corpus, allowing us to
    share the cache for unchanged regions.
    """
    
    def __init__(self, source_model: CRPSourceModel, channel_model: CRPChannelModel):
        """Initialize the incremental scorer.
        
        Args:
            source_model: CRP source model P(p)
            channel_model: CRP channel model P(c|p)
        """
        self.source_model = source_model
        self.channel_model = channel_model
        self.n = source_model.n
    
    def find_affected_positions(self, ciphertext: List[int], changed_symbol: int, 
                               space_positions: Set[int]) -> Set[int]:
        """Find all positions in plaintext affected by changing a cipher symbol.
        
        When we change the mapping for a cipher symbol, all occurrences of that
        symbol in the ciphertext are affected.
        
        Args:
            ciphertext: The ciphertext symbols
            changed_symbol: The cipher symbol whose mapping changed
            space_positions: Current space positions
            
        Returns:
            Set[int]: Indices in the plaintext (with spaces) that are affected
        """
        affected = set()
        
        # Find all positions of this symbol in ciphertext
        for cipher_idx, symbol in enumerate(ciphertext):
            if symbol == changed_symbol:
                # Convert cipher index to plaintext index (accounting for spaces)
                plain_idx = cipher_idx
                for space_pos in sorted(space_positions):
                    if space_pos <= cipher_idx:
                        plain_idx += 1
                    else:
                        break
                affected.add(plain_idx)
        
        return affected
    
    def find_context_window(self, plaintext: str, affected_positions: Set[int]) -> Tuple[int, int]:
        """Find the context window that needs to be rescored.
        
        The context window extends n-1 characters before and after the affected region
        to capture all n-grams that include the changed characters.
        
        Args:
            plaintext: The full plaintext
            affected_positions: Set of affected character positions
            
        Returns:
            Tuple[int, int]: (start_index, end_index) of the context window
        """
        if not affected_positions:
            return (0, 0)
        
        # Find min and max affected positions
        min_pos = min(affected_positions)
        max_pos = max(affected_positions)
        
        # Extend by n-1 in each direction for context
        start = max(0, min_pos - (self.n - 1))
        end = min(len(plaintext), max_pos + self.n)
        
        return (start, end)
    
    def score_window_source(self, plaintext: str, start: int, end: int,
                           use_existing_cache: bool = True) -> float:
        """Score a window of plaintext using the source model.
        
        Args:
            plaintext: The full plaintext
            start: Start index of window
            end: End index of window
            use_existing_cache: If True, use cache built from text before start
            
        Returns:
            float: Log probability of the window
        """
        if start >= end:
            return 0.0
        
        window_text = plaintext[start:end]
        
        # If not using existing cache, clear it
        if not use_existing_cache:
            self.source_model.clear_cache()
            # Build cache from text before window
            if start > 0:
                self.source_model.build_cache_from_text(plaintext[:start])
        
        # Score the window
        char_list = list(window_text.lower())
        padded = list(pad_both_ends(char_list, n=self.n))
        
        total_log_prob = 0.0
        
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - (self.n - 1):i])
            char = padded[i]
            
            prob = self.source_model.score_char(char, context)
            
            if prob <= 0:
                return -float("inf")
            
            total_log_prob += math.log(prob)
            
            # Update cache for next character
            self.source_model.cache.add_ngram(context, char)
        
        return total_log_prob
    
    def score_window_channel(self, ciphertext: List[int], plaintext: str,
                            affected_positions: Set[int], key: dict,
                            use_existing_cache: bool = True) -> float:
        """Score affected substitutions using the channel model.
        
        Args:
            ciphertext: The ciphertext symbols
            plaintext: The full plaintext (with spaces)
            affected_positions: Positions in plaintext that changed
            key: Current substitution key
            use_existing_cache: If True, use cache built from unaffected pairs
            
        Returns:
            float: Log probability of the affected substitutions
        """
        if not affected_positions:
            return 0.0
        
        # Remove spaces to align with ciphertext
        plaintext_no_spaces = plaintext.replace(" ", "").lower()
        
        if not use_existing_cache:
            self.channel_model.clear_cache()
            # Build cache from unaffected pairs
            for i, (cipher_symbol, plain_char) in enumerate(zip(ciphertext, plaintext_no_spaces)):
                # Convert plaintext index (no spaces) to plaintext index (with spaces)
                # to check if it's affected
                # This is a simplified check - in practice, we track by cipher position
                if i not in affected_positions:
                    self.channel_model.cache.add_substitution(plain_char, cipher_symbol)
        
        # Score only the affected substitutions
        total_log_prob = 0.0
        
        for cipher_idx, cipher_symbol in enumerate(ciphertext):
            plain_char = plaintext_no_spaces[cipher_idx]
            
            # Check if this position is affected
            # (We need to map cipher_idx to plaintext_with_spaces_idx)
            # For now, rescore all positions where the symbol appears
            if key.get(cipher_symbol) != plain_char:
                continue  # Skip if key doesn't match (shouldn't happen)
            
            # Check if this cipher symbol was the one that changed
            # (This is a simplification - proper implementation would track exact positions)
            
            prob = self.channel_model.score_substitution(plain_char, cipher_symbol)
            
            if prob <= 0:
                return -float("inf")
            
            total_log_prob += math.log(prob)
        
        return total_log_prob
    
    def score_proposal_incremental(self, ciphertext: List[int], 
                                   old_plaintext: str, new_plaintext: str,
                                   old_key: dict, new_key: dict,
                                   changed_symbol: int,
                                   space_positions: Set[int]) -> Tuple[float, float]:
        """Score a proposal incrementally by only rescoring affected regions.
        
        This is the key optimization: instead of rescoring the entire text,
        we only rescore the parts that changed.
        
        Args:
            ciphertext: The ciphertext symbols
            old_plaintext: Previous plaintext hypothesis
            new_plaintext: Proposed new plaintext
            old_key: Previous key
            new_key: Proposed new key
            changed_symbol: Which cipher symbol's mapping changed
            space_positions: Current space positions
            
        Returns:
            Tuple[float, float]: (source_log_prob, channel_log_prob)
        """
        # Find affected positions
        affected_positions = self.find_affected_positions(ciphertext, changed_symbol, space_positions)
        
        # Find context window
        start, end = self.find_context_window(new_plaintext, affected_positions)
        
        # Score the window with source model
        source_score = self.score_window_source(new_plaintext, start, end, use_existing_cache=False)
        
        # Score affected substitutions with channel model
        # For simplicity, we'll rescore all substitutions for the changed symbol
        channel_score = 0.0
        plaintext_no_spaces = new_plaintext.replace(" ", "").lower()
        
        for cipher_idx, cipher_symbol in enumerate(ciphertext):
            if cipher_symbol == changed_symbol:
                plain_char = plaintext_no_spaces[cipher_idx]
                prob = self.channel_model.score_substitution(plain_char, cipher_symbol)
                if prob <= 0:
                    return (-float("inf"), -float("inf"))
                channel_score += math.log(prob)
        
        return (source_score, channel_score)
