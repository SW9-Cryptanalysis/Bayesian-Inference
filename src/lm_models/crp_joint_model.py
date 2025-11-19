"""
CRP-based joint model combining source P(p) and channel P(c|p).

Implements the full Bayesian decipherment approach from the paper:
P(p, c) = P(p) * P(c|p)

where both P(p) and P(c|p) use Chinese Restaurant Process formulations.
"""

import logging
from typing import List, Tuple
from lm_models.crp_source_model import CRPSourceModel
from lm_models.crp_channel_model import CRPChannelModel
from lm_models.dictionary_model import DictionaryLanguageModel

logger = logging.getLogger(__name__)


class CRPJointModel:
    """Joint model combining CRP source and channel models.
    
    This is the main scoring model for CRP-based Bayesian decipherment.
    It combines:
    1. CRP source model P(p) - character n-grams with cache
    2. CRP channel model P(c|p) - substitution probabilities with cache
    3. Optional dictionary model for word-level scoring
    """
    
    def __init__(self, 
                 crp_source_model: CRPSourceModel,
                 crp_channel_model: CRPChannelModel,
                 dict_model: DictionaryLanguageModel | None = None,
                 ngram_weight: float = 0.1,
                 word_weight: float = 0.9):
        """Initialize the joint model.
        
        Args:
            crp_source_model: CRP-based n-gram source model
            crp_channel_model: CRP-based channel model
            dict_model: Dictionary model for word scoring (interpolated with n-gram)
            ngram_weight: Weight for n-gram model in P(p) (default 0.1 from paper)
            word_weight: Weight for word model in P(p) (default 0.9 from paper)
        """
        self.crp_source = crp_source_model
        self.crp_channel = crp_channel_model
        self.dict_model = dict_model
        
        # Weights for interpolating n-gram and word models (from paper)
        self.ngram_weight = ngram_weight
        self.word_weight = word_weight
    
    def log_score(self, ciphertext: List[int], plaintext: str, key: dict) -> float:
        """Calculate the complete log P(p, c) = log P(p) + log P(c|p).
        
        This is the main scoring function for CRP-based decipherment.
        P(p) is computed as an interpolation of n-gram and word models.
        
        Args:
            ciphertext: List of cipher symbols
            plaintext: The plaintext hypothesis (with spaces)
            key: Substitution key mapping cipher symbols to plaintext chars
            
        Returns:
            float: Combined log probability score
        """
        # Score source: P(p) = interpolated n-gram + word model
        ngram_score = self.crp_source.log_score_text(plaintext)
        word_score = 0.0
        if self.dict_model is not None:
            word_score = self.dict_model.log_score_text(plaintext)
        
        # Interpolate n-gram and word models for P(p) (paper uses 0.1, 0.9)
        source_score = (self.ngram_weight * ngram_score + 
                       self.word_weight * word_score)
        
        # Score channel: P(c|p)
        channel_score = self.crp_channel.log_score_key(ciphertext, plaintext, key)
        
        # Final score: log P(p, c) = log P(p) + log P(c|p)
        combined_score = source_score + channel_score
        
        return combined_score
    
    def log_score_separate(self, ciphertext: List[int], plaintext: str, 
                          key: dict) -> Tuple[float, float, float, float]:
        """Get separate scores for debugging.
        
        Args:
            ciphertext: List of cipher symbols
            plaintext: The plaintext hypothesis
            key: Substitution key
            
        Returns:
            Tuple[float, float, float, float]: (ngram_score, word_score, interpolated_source_score, channel_score)
        """
        ngram_score = self.crp_source.log_score_text(plaintext)
        word_score = 0.0
        if self.dict_model is not None:
            word_score = self.dict_model.log_score_text(plaintext)
        
        # Interpolated source score
        source_score = (self.ngram_weight * ngram_score + 
                       self.word_weight * word_score)
        
        channel_score = self.crp_channel.log_score_key(ciphertext, plaintext, key)
        
        return (ngram_score, word_score, source_score, channel_score)
    
    def initialize_caches(self, ciphertext: List[int], plaintext: str) -> None:
        """Initialize both source and channel caches with current hypothesis.
        
        This should be called once at the start with the initial plaintext.
        
        Args:
            ciphertext: The ciphertext symbols
            plaintext: The initial plaintext hypothesis
        """
        self.crp_source.build_cache_from_text(plaintext)
        self.crp_channel.build_cache_from_pairs(ciphertext, plaintext)
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.crp_source.clear_cache()
        self.crp_channel.clear_cache()
