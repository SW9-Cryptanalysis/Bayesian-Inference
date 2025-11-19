"""
CRP-based channel model P(c|p) for Bayesian decipherment.

Implements the channel substitution model using Chinese Restaurant Process:
P(ci | pi) = (β * P0(ci|pi) + C^{i-1}_1(pi, ci)) / (β + C^{i-1}_1(pi))

where:
- β is the Dirichlet prior (low value favors sparse/deterministic substitution)
- P0 is the base distribution (uniform for channel)
- C^{i-1}_1 tracks substitutions seen before position i
"""

import math
import logging
from typing import List
from lm_models.crp_cache import ChannelCache
from utils.constants import BETA

logger = logging.getLogger(__name__)


class CRPChannelModel:
    """CRP-based channel model for scoring cipher-to-plaintext substitutions.
    
    Models P(c|p) - the probability of observing cipher symbol c given plaintext p.
    Uses a uniform base distribution and cache of observed substitutions.
    """
    
    def __init__(self, num_cipher_symbols: int, beta: float = BETA):
        """Initialize the CRP channel model.
        
        Args:
            num_cipher_symbols: Total number of unique cipher symbols
            beta: Dirichlet prior hyperparameter (default from paper: 0.01)
        """
        self.num_cipher_symbols = num_cipher_symbols
        self.beta = beta
        self.cache = ChannelCache()
        
        # Uniform base distribution: each plaintext can map to any cipher symbol equally
        self.base_prob = 1.0 / num_cipher_symbols
    
    def score_substitution(self, plaintext_char: str, cipher_symbol: int) -> float:
        """Score a single substitution using CRP formula.
        
        P(cipher | plaintext) = (β * P0 + C(plain, cipher)) / (β + C(plain))
        
        Args:
            plaintext_char: The plaintext character
            cipher_symbol: The cipher symbol it maps to
            
        Returns:
            float: Probability P(cipher_symbol | plaintext_char)
        """
        # Get cache counts
        cache_count = self.cache.get_count(plaintext_char, cipher_symbol)
        plaintext_total = self.cache.get_plaintext_total(plaintext_char)
        
        # CRP formula with uniform base distribution
        numerator = (self.beta * self.base_prob) + cache_count
        denominator = self.beta + plaintext_total
        
        return numerator / denominator
    
    def log_score_key(self, ciphertext: List[int], plaintext: str, key: dict) -> float:
        """Calculate log probability of ciphertext given plaintext and key.
        
        This builds the cache incrementally as it processes the cipher-plaintext pairs.
        
        Args:
            ciphertext: List of cipher symbols
            plaintext: The plaintext string (without spaces for alignment)
            key: Mapping from cipher symbols to plaintext chars
            
        Returns:
            float: Average log probability per character
        """
        if not ciphertext or not plaintext:
            return -float("inf")
        
        # Remove spaces from plaintext for alignment with ciphertext
        plaintext_no_spaces = plaintext.replace(" ", "").lower()
        
        if len(ciphertext) != len(plaintext_no_spaces):
            logger.warning(f"Length mismatch: cipher={len(ciphertext)}, plain={len(plaintext_no_spaces)}")
            return -float("inf")
        
        # Clear cache for this scoring
        self.cache.clear()
        
        total_log_prob = 0.0
        
        # Score each substitution sequentially, updating cache as we go
        for cipher_symbol, plaintext_char in zip(ciphertext, plaintext_no_spaces):
            # Verify this substitution matches the key
            expected_char = key.get(cipher_symbol)
            if expected_char != plaintext_char:
                logger.warning(f"Key mismatch: symbol {cipher_symbol} -> {expected_char} but plaintext has {plaintext_char}")
                return -float("inf")
            
            # Score using current cache state
            prob = self.score_substitution(plaintext_char, cipher_symbol)
            
            if prob <= 0:
                logger.warning(f"Zero probability for substitution {plaintext_char} -> {cipher_symbol}")
                return -float("inf")
            
            total_log_prob += math.log(prob)
            
            # Update cache with this observation
            self.cache.add_substitution(plaintext_char, cipher_symbol)
        
        # Return average log probability per character
        return total_log_prob / len(ciphertext)
    
    def build_cache_from_pairs(self, ciphertext: List[int], plaintext: str) -> None:
        """Build the cache from cipher-plaintext pairs without scoring.
        
        Used to initialize the cache with the current hypothesis.
        
        Args:
            ciphertext: List of cipher symbols
            plaintext: The plaintext string (spaces will be removed)
        """
        if not ciphertext or not plaintext:
            return
        
        plaintext_no_spaces = plaintext.replace(" ", "").lower()
        
        if len(ciphertext) != len(plaintext_no_spaces):
            logger.warning(f"Length mismatch in build_cache: cipher={len(ciphertext)}, plain={len(plaintext_no_spaces)}")
            return
        
        self.cache.clear()
        
        for cipher_symbol, plaintext_char in zip(ciphertext, plaintext_no_spaces):
            self.cache.add_substitution(plaintext_char, cipher_symbol)
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()
    
    def get_cache_copy(self) -> ChannelCache:
        """Get a copy of the current cache.
        
        Returns:
            ChannelCache: Deep copy of the cache
        """
        return self.cache.copy()
    
    def set_cache(self, cache: ChannelCache) -> None:
        """Set the cache to a specific state.
        
        Args:
            cache: The cache to use
        """
        self.cache = cache
