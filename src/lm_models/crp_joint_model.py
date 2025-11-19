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
                 dict_model: DictionaryLanguageModel = None,
                 source_weight: float = 0.5,
                 dict_weight: float = 0.4,
                 channel_weight: float = 0.1):
        """Initialize the joint model.
        
        Args:
            crp_source_model: CRP-based n-gram source model
            crp_channel_model: CRP-based channel model
            dict_model: Optional dictionary model for word scoring
            source_weight: Weight for CRP source model (default 0.5)
            dict_weight: Weight for dictionary model (default 0.4)
            channel_weight: Weight for channel model (default 0.1)
        """
        self.crp_source = crp_source_model
        self.crp_channel = crp_channel_model
        self.dict_model = dict_model
        
        # Weights for combining models (should sum to 1.0)
        self.source_weight = source_weight
        self.dict_weight = dict_weight
        self.channel_weight = channel_weight
        
        # Normalize weights
        total = source_weight + dict_weight + channel_weight
        self.source_weight /= total
        self.dict_weight /= total
        self.channel_weight /= total
    
    def log_score(self, ciphertext: List[int], plaintext: str, key: dict) -> float:
        """Calculate the complete log P(p, c) = log P(p) + log P(c|p).
        
        This is the main scoring function for CRP-based decipherment.
        
        Args:
            ciphertext: List of cipher symbols
            plaintext: The plaintext hypothesis (with spaces)
            key: Substitution key mapping cipher symbols to plaintext chars
            
        Returns:
            float: Combined log probability score
        """
        # Score source: P(p)
        source_score = self.crp_source.log_score_text(plaintext)
        
        # Score channel: P(c|p)
        channel_score = self.crp_channel.log_score_key(ciphertext, plaintext, key)
        
        # Optional dictionary score
        dict_score = 0.0
        if self.dict_model is not None:
            dict_score = self.dict_model.log_score_text(plaintext)
        
        # Combine with weights
        combined_score = (self.source_weight * source_score + 
                         self.dict_weight * dict_score +
                         self.channel_weight * channel_score)
        
        return combined_score
    
    def log_score_separate(self, ciphertext: List[int], plaintext: str, 
                          key: dict) -> Tuple[float, float, float]:
        """Get separate scores for debugging.
        
        Args:
            ciphertext: List of cipher symbols
            plaintext: The plaintext hypothesis
            key: Substitution key
            
        Returns:
            Tuple[float, float, float]: (source_score, dict_score, channel_score)
        """
        source_score = self.crp_source.log_score_text(plaintext)
        channel_score = self.crp_channel.log_score_key(ciphertext, plaintext, key)
        dict_score = 0.0
        if self.dict_model is not None:
            dict_score = self.dict_model.log_score_text(plaintext)
        
        return (source_score, dict_score, channel_score)
    
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
