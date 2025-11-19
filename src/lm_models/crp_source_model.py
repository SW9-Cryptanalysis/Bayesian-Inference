"""
CRP-based source model P(p) for Bayesian decipherment.

Implements the Chinese Restaurant Process formulation from the paper:
P(pi | pi-1) = (α * P0(pi|pi-1) + C^{i-1}_1(pi-1, pi)) / (α + C^{i-1}_1(pi-1))

where:
- α is the Dirichlet prior (high value favors base distribution)
- P0 is the base distribution (from pre-trained language model)
- C^{i-1}_1 is the cache tracking n-grams seen before position i
"""

import math
import logging
from typing import Tuple
from nltk.lm.preprocessing import pad_both_ends
from lm_models.n_gram_model import NgramLanguageModel
from lm_models.crp_cache import CRPCache
from utils.constants import ALPHA

logger = logging.getLogger(__name__)


class CRPSourceModel:
    """CRP-based source model for scoring plaintext hypotheses.
    
    Combines a base language model P0 with a cache of observed n-grams,
    using the Chinese Restaurant Process formulation.
    """
    
    def __init__(self, base_ngram_model: NgramLanguageModel, alpha: float = ALPHA):
        """Initialize the CRP source model.
        
        Args:
            base_ngram_model: The base n-gram language model (provides P0)
            alpha: Dirichlet prior hyperparameter (default from paper: 10^4)
        """
        self.base_model = base_ngram_model
        self.alpha = alpha
        self.n = base_ngram_model.n
        self.cache = CRPCache(self.n)
    
    def score_char(self, char: str, context: Tuple[str, ...]) -> float:
        """Score a single character given its context using CRP formula.
        
        P(char | context) = (α * P0(char|context) + C(context, char)) / (α + C(context))
        
        Args:
            char: The character to score
            context: Tuple of previous characters (length n-1)
            
        Returns:
            float: Probability P(char | context)
        """
        # Get base probability from pre-trained model
        base_prob = self.base_model.score_char(char, context)
        
        # Get cache counts
        cache_count = self.cache.get_count(context, char)
        context_total = self.cache.get_context_total(context)
        
        # CRP formula
        numerator = (self.alpha * base_prob) + cache_count
        denominator = self.alpha + context_total
        
        return numerator / denominator
    
    def log_score_text(self, text: str) -> float:
        """Calculate the log probability of a text using CRP.
        
        This builds the cache incrementally as it scores the text,
        implementing the sequential generative process.
        
        Args:
            text: The plaintext to score
            
        Returns:
            float: Average log probability per character
        """
        if not text:
            return -float("inf")
        
        # Preprocess text (lowercase, spaces only)
        text = text.lower().strip()
        if not text:
            return -float("inf")
        
        # Clear cache for this scoring
        self.cache.clear()
        
        # Convert to character list and pad
        char_list = list(text)
        padded = list(pad_both_ends(char_list, n=self.n))
        
        total_log_prob = 0.0
        
        # Score each character sequentially, updating cache as we go
        for i in range(self.n - 1, len(padded)):
            # Get context (previous n-1 characters)
            context = tuple(padded[i - (self.n - 1):i])
            char = padded[i]
            
            # Score using current cache state
            prob = self.score_char(char, context)
            
            if prob <= 0:
                logger.warning(f"Zero probability for char '{char}' with context {context}")
                return -float("inf")
            
            total_log_prob += math.log(prob)
            
            # Update cache with this observation (for future characters)
            self.cache.add_ngram(context, char)
        
        # Return average log probability per character
        return total_log_prob / len(char_list)
    
    def build_cache_from_text(self, text: str) -> None:
        """Build the cache from a complete text without scoring.
        
        Used to initialize the cache with the current plaintext hypothesis.
        
        Args:
            text: The plaintext to build cache from
        """
        if not text:
            return
        
        text = text.lower().strip()
        if not text:
            return
        
        self.cache.clear()
        
        char_list = list(text)
        padded = list(pad_both_ends(char_list, n=self.n))
        
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - (self.n - 1):i])
            char = padded[i]
            self.cache.add_ngram(context, char)
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()
    
    def get_cache_copy(self) -> CRPCache:
        """Get a copy of the current cache.
        
        Returns:
            CRPCache: Deep copy of the cache
        """
        return self.cache.copy()
    
    def set_cache(self, cache: CRPCache) -> None:
        """Set the cache to a specific state.
        
        Args:
            cache: The cache to use
        """
        self.cache = cache
