"""
Efficient incremental scoring using the exchangeability property.

When sampling at a position, we only need to rescore the affected context window
rather than the entire plaintext. This dramatically speeds up sampling for long texts.

EXCHANGEABILITY PROPERTY (from paper):
The key insight: when sampling at position i, we pretend the affected area occurs 
at the end of the corpus. This means:

1. Both old and new derivations share the same cache for text before affected region
2. We only need to score the difference between old and new in the affected window
3. Score_new = Score_old - Score_old_window + Score_new_window

Example:
  Text: "the quick brown fox"
  Change: 'q' -> 'x' at position 4
  Affected window: "the [quick] brown" (with n-gram context)
  
  Instead of rescoring entire text:
    - Save cache state from "the "
    - Remove "quick" from cache and compute its score
    - Add "xuick" to cache and compute its score
    - Delta = score("xuick") - score("quick")
    
This is valid because CRP is exchangeable: the order of observations doesn't 
affect the final probability, as long as we account for the sequential cache updates.
"""

import math
import logging
from typing import List, Tuple, Set, Optional
from nltk.lm.preprocessing import pad_both_ends
from lm_models.crp_source_model import CRPSourceModel
from lm_models.crp_channel_model import CRPChannelModel
from lm_models.crp_cache import CRPCache, ChannelCache

logger = logging.getLogger(__name__)


class IncrementalScorer:
    """Handles efficient incremental scoring for CRP models.
    
    Uses the exchangeability property: when updating positions, pretend
    the affected region occurs at the end of the corpus, allowing us to
    share the cache for unchanged regions.
    
    This implementation properly:
    1. Saves the cache state before scoring a proposal
    2. Scores only the affected window with proper context
    3. Computes the score delta (not absolute scores)
    4. Restores cache if proposal is rejected
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
    
    def find_affected_positions_in_plaintext(self, ciphertext: List[int], changed_symbol: int, 
                                             space_positions: Set[int]) -> List[int]:
        """Find all positions in plaintext affected by changing a cipher symbol.
        
        When we change the mapping for a cipher symbol, all occurrences of that
        symbol in the ciphertext are affected.
        
        Args:
            ciphertext: The ciphertext symbols
            changed_symbol: The cipher symbol whose mapping changed
            space_positions: Current space positions
            
        Returns:
            List[int]: Sorted indices in the plaintext (with spaces) that are affected
        """
        affected = []
        
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
                affected.append(plain_idx)
        
        return sorted(affected)
    
    def find_context_window(self, plaintext: str, affected_positions: List[int]) -> Tuple[int, int]:
        """Find the context window that needs to be rescored.
        
        The context window extends n-1 characters before and after the affected region
        to capture all n-grams that include the changed characters. We also extend to
        word boundaries (spaces) to properly handle word-level scoring.
        
        Args:
            plaintext: The full plaintext
            affected_positions: Sorted list of affected character positions
            
        Returns:
            Tuple[int, int]: (start_index, end_index) of the context window
        """
        if not affected_positions:
            return (0, 0)
        
        # Find min and max affected positions
        min_pos = affected_positions[0]
        max_pos = affected_positions[-1]
        
        # Extend by n-1 in each direction for n-gram context
        start = max(0, min_pos - (self.n - 1))
        end = min(len(plaintext), max_pos + self.n)
        
        # Extend to word boundaries (spaces) for proper word scoring
        while start > 0 and plaintext[start - 1] != ' ':
            start -= 1
        while end < len(plaintext) and plaintext[end] != ' ':
            end += 1
        
        return (start, end)
    
    def score_window_source(self, plaintext: str, start: int, end: int, 
                           saved_cache: Optional[CRPCache] = None) -> float:
        """Score a window of plaintext using the source model with proper cache handling.
        
        Uses exchangeability: the cache is built from text before the window, then
        we score the window and update the cache as we go.
        
        Args:
            plaintext: The full plaintext
            start: Start index of window
            end: End index of window
            saved_cache: Pre-saved cache state from before the window (if None, builds from scratch)
            
        Returns:
            float: Log probability of the window
        """
        if start >= end:
            return 0.0
        
        # Set or build cache for text before window
        if saved_cache is not None:
            self.source_model.set_cache(saved_cache)
        else:
            self.source_model.clear_cache()
            if start > 0:
                self.source_model.build_cache_from_text(plaintext[:start])
        
        # Score the window
        window_text = plaintext[start:end]
        char_list = list(window_text.lower())
        padded = list(pad_both_ends(char_list, n=self.n))
        
        total_log_prob = 0.0
        
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - (self.n - 1):i])
            char = padded[i]
            
            prob = self.source_model.score_char(char, context)
            
            if prob <= 0:
                logger.warning(f"Zero prob in window scoring: char='{char}', context={context}")
                return -float("inf")
            
            total_log_prob += math.log(prob)
            
            # Update cache for next character
            self.source_model.cache.add_ngram(context, char)
        
        return total_log_prob
    
    def score_substitutions_for_symbol(self, ciphertext: List[int], plaintext: str,
                                       changed_symbol: int, key: dict,
                                       saved_cache: Optional[ChannelCache] = None) -> float:
        """Score all substitutions for a specific cipher symbol using channel model.
        
        Args:
            ciphertext: The ciphertext symbols
            plaintext: The plaintext (with spaces)
            changed_symbol: The cipher symbol to score
            key: Current substitution key
            saved_cache: Pre-saved cache state (if None, builds from scratch)
            
        Returns:
            float: Log probability of all substitutions for this symbol
        """
        plaintext_no_spaces = plaintext.replace(" ", "").lower()
        
        if len(ciphertext) != len(plaintext_no_spaces):
            logger.error(f"Length mismatch: cipher={len(ciphertext)}, plain={len(plaintext_no_spaces)}")
            return -float("inf")
        
        # Set or build cache
        if saved_cache is not None:
            self.channel_model.set_cache(saved_cache)
        else:
            self.channel_model.clear_cache()
            # Build cache from all substitutions except those involving changed_symbol
            for cipher_symbol, plain_char in zip(ciphertext, plaintext_no_spaces):
                if cipher_symbol != changed_symbol:
                    self.channel_model.cache.add_substitution(plain_char, cipher_symbol)
        
        # Score only substitutions for the changed symbol
        total_log_prob = 0.0
        count = 0
        
        for cipher_symbol, plain_char in zip(ciphertext, plaintext_no_spaces):
            if cipher_symbol == changed_symbol:
                # Verify key consistency
                if key.get(cipher_symbol) != plain_char:
                    logger.error(f"Key mismatch: {cipher_symbol} -> {key.get(cipher_symbol)} != {plain_char}")
                    return -float("inf")
                
                prob = self.channel_model.score_substitution(plain_char, cipher_symbol)
                
                if prob <= 0:
                    logger.warning(f"Zero prob in channel: {plain_char} -> {cipher_symbol}")
                    return -float("inf")
                
                total_log_prob += math.log(prob)
                count += 1
                
                # Update cache for next occurrence
                self.channel_model.cache.add_substitution(plain_char, cipher_symbol)
        
        return total_log_prob
    
    def _remove_window_from_cache(self, plaintext: str, start: int, end: int, cache: CRPCache) -> None:
        """Remove n-grams in the window from the cache.
        
        Used to create the 'shared cache' by removing the affected region.
        """
        if start >= end:
            return
            
        window_text = plaintext[start:end]
        char_list = list(window_text.lower())
        padded = list(pad_both_ends(char_list, n=self.n))
        
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - (self.n - 1):i])
            char = padded[i]
            cache.remove_ngram(context, char)

    def _remove_substitutions_from_cache(self, ciphertext: List[int], plaintext: str, 
                                         changed_symbol: int, key: dict, cache: ChannelCache) -> None:
        """Remove substitutions for a specific symbol from the cache.
        
        Used to create the 'shared cache' by removing the affected substitutions.
        """
        plaintext_no_spaces = plaintext.replace(" ", "").lower()
        
        for cipher_symbol, plain_char in zip(ciphertext, plaintext_no_spaces):
            if cipher_symbol == changed_symbol:
                cache.remove_substitution(plain_char, cipher_symbol)

    def _add_window_to_cache(self, plaintext: str, start: int, end: int, cache: CRPCache) -> None:
        """Add n-grams in the window to the cache."""
        if start >= end:
            return
            
        window_text = plaintext[start:end]
        char_list = list(window_text.lower())
        padded = list(pad_both_ends(char_list, n=self.n))
        
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - (self.n - 1):i])
            char = padded[i]
            cache.add_ngram(context, char)

    def _add_substitutions_to_cache(self, ciphertext: List[int], plaintext: str, 
                                    changed_symbol: int, key: dict, cache: ChannelCache) -> None:
        """Add substitutions for a specific symbol to the cache."""
        plaintext_no_spaces = plaintext.replace(" ", "").lower()
        
        for cipher_symbol, plain_char in zip(ciphertext, plaintext_no_spaces):
            if cipher_symbol == changed_symbol:
                cache.add_substitution(plain_char, cipher_symbol)

    def apply_key_changes_to_cache(self, ciphertext: List[int], 
                                  old_plaintext: str, new_plaintext: str,
                                  old_key: dict, new_key: dict,
                                  changed_symbol: int,
                                  space_positions: Set[int]) -> None:
        """Apply the changes to the actual model caches (after acceptance)."""
        
        # Find affected positions
        affected_positions = self.find_affected_positions_in_plaintext(
            ciphertext, changed_symbol, space_positions
        )
        if not affected_positions:
            return

        # Find context window
        start, end = self.find_context_window(new_plaintext, affected_positions)
        
        # Update Source Cache
        # Remove old window
        self._remove_window_from_cache(old_plaintext, start, end, self.source_model.cache)
        # Add new window
        self._add_window_to_cache(new_plaintext, start, end, self.source_model.cache)
        
        # Update Channel Cache
        # Remove old substitutions
        self._remove_substitutions_from_cache(ciphertext, old_plaintext, changed_symbol, old_key, self.channel_model.cache)
        # Add new substitutions
        self._add_substitutions_to_cache(ciphertext, new_plaintext, changed_symbol, new_key, self.channel_model.cache)

    def apply_space_changes_to_cache(self, old_plaintext: str, new_plaintext: str,
                                    old_start: int, old_end: int,
                                    new_start: int, new_end: int) -> None:
        """Apply space changes to the source model cache (after acceptance)."""
        # Remove old window
        self._remove_window_from_cache(old_plaintext, old_start, old_end, self.source_model.cache)
        # Add new window
        self._add_window_to_cache(new_plaintext, new_start, new_end, self.source_model.cache)
        
        # Channel cache is invariant to spaces, so no update needed

    def score_key_proposal(self, ciphertext: List[int], 
                          old_plaintext: str, new_plaintext: str,
                          old_key: dict, new_key: dict,
                          changed_symbol: int,
                          space_positions: Set[int],
                          old_source_score: float, old_channel_score: float) -> Tuple[float, float, float, float]:
        """Score a key change proposal using proper incremental scoring with exchangeability.
        
        This method:
        1. Saves current cache states
        2. Finds affected positions and context window
        3. Removes old window/substitutions from caches (creating shared caches)
        4. Scores old window/substitutions using shared caches
        5. Scores new window/substitutions using shared caches
        6. Returns score deltas
        
        The cache is restored if the proposal is rejected (handled by caller).
        
        Args:
            ciphertext: The ciphertext symbols
            old_plaintext: Current plaintext hypothesis
            new_plaintext: Proposed new plaintext
            old_key: Current substitution key
            new_key: Proposed substitution key
            changed_symbol: Which cipher symbol's mapping changed
            space_positions: Current space positions
            old_source_score: Current source model score (for computing delta)
            old_channel_score: Current channel model score (for computing delta)
            
        Returns:
            Tuple[float, float, float, float]: (new_source_score, new_channel_score, 
                                                  source_delta, channel_delta)
        """
        # Save current cache states (these are full caches)
        saved_source_cache = self.source_model.get_cache_copy()
        saved_channel_cache = self.channel_model.get_cache_copy()
        
        # Find affected positions in plaintext
        affected_positions = self.find_affected_positions_in_plaintext(
            ciphertext, changed_symbol, space_positions
        )
        
        if not affected_positions:
            # No changes needed
            return (old_source_score, old_channel_score, 0.0, 0.0)
        
        # Find context window for source scoring
        start, end = self.find_context_window(new_plaintext, affected_positions)
        
        # Create SHARED caches by removing the affected regions from the full caches
        # This implements the "pretend affected area is at the end" logic
        
        # 1. Prepare Shared Source Cache
        shared_source_cache = saved_source_cache.copy()
        self._remove_window_from_cache(old_plaintext, start, end, shared_source_cache)
        
        # 2. Prepare Shared Channel Cache
        shared_channel_cache = saved_channel_cache.copy()
        self._remove_substitutions_from_cache(ciphertext, old_plaintext, changed_symbol, old_key, shared_channel_cache)
        
        # Score OLD window (using shared cache)
        # We use a copy because scoring modifies the cache
        old_window_score = self.score_window_source(old_plaintext, start, end, shared_source_cache.copy())
        
        # Score NEW window (using shared cache)
        new_window_score = self.score_window_source(new_plaintext, start, end, shared_source_cache.copy())
        
        # Source delta
        source_delta = new_window_score - old_window_score
        new_source_score = old_source_score + source_delta
        
        # Score OLD channel substitutions (using shared cache)
        old_channel_score_for_symbol = self.score_substitutions_for_symbol(
            ciphertext, old_plaintext, changed_symbol, old_key, shared_channel_cache.copy()
        )
        
        # Score NEW channel substitutions (using shared cache)
        new_channel_score_for_symbol = self.score_substitutions_for_symbol(
            ciphertext, new_plaintext, changed_symbol, new_key, shared_channel_cache.copy()
        )
        
        # Channel delta
        channel_delta = new_channel_score_for_symbol - old_channel_score_for_symbol
        new_channel_score = old_channel_score + channel_delta
        
        return (new_source_score, new_channel_score, source_delta, channel_delta)
    
    def restore_caches(self, saved_source_cache: CRPCache, saved_channel_cache: ChannelCache) -> None:
        """Restore cache states (used when proposal is rejected).
        
        Args:
            saved_source_cache: Saved source model cache
            saved_channel_cache: Saved channel model cache
        """
        self.source_model.set_cache(saved_source_cache)
        self.channel_model.set_cache(saved_channel_cache)
