"""
CRP-based Bayesian sampler for substitution cipher decipherment.

This is the main implementation following the paper's approach using
Chinese Restaurant Process formulations for both source and channel models.
"""

import random
import math
import logging
import json
from collections import Counter
from typing import Tuple, Dict, List, Set, Union

from lm_models.crp_joint_model import CRPJointModel
from lm_models.crp_source_model import CRPSourceModel
from lm_models.crp_channel_model import CRPChannelModel
from lm_models.n_gram_model import NgramLanguageModel
from lm_models.dictionary_model import DictionaryLanguageModel
from utils.constants import TOTAL_ITERATIONS, INITIAL_TEMPERATURE, PROJECT_ROOT, ALPHA, BETA

logger = logging.getLogger(__name__)


class CRPBayesianSampler:
    """Bayesian sampler using CRP for source P(p) and channel P(c|p) models.
    
    This implements the full approach from the paper including:
    - Chinese Restaurant Process with cache tracking
    - Dirichlet priors (α for source, β for channel)
    - Type sampling for keys
    - Space sampling for word boundaries
    - Joint scoring P(p, c) = P(p) * P(c|p)
    """
    
    def __init__(self, 
                 ciphertext: List[int],
                 ngram_model: NgramLanguageModel,
                 dict_model: DictionaryLanguageModel,
                 ground_truth_key: Dict[str, List[int]],
                 seed: int | None = None,
                 use_crp: bool = True):
        """Initialize the CRP Bayesian sampler.
        
        Args:
            ciphertext: List of cipher symbols
            ngram_model: Base n-gram language model (provides P0 for source)
            dict_model: Dictionary language model for word-level scoring
            ground_truth_key: Ground truth key for evaluation
            seed: Random seed for reproducibility
            use_crp: If True, use CRP models; if False, use standard models (for comparison)
        """
        if seed is not None:
            random.seed(seed)
        
        self._ciphertext = ciphertext
        self.use_crp = use_crp
        self.ground_truth_key = self._convert_ground_truth_key(ground_truth_key)
        
        # Initialize CRP models
        if use_crp:
            self.crp_source = CRPSourceModel(ngram_model, alpha=ALPHA)
            self.crp_channel = CRPChannelModel(len(set(ciphertext)), beta=BETA)
            self.model = CRPJointModel(
                self.crp_source,
                self.crp_channel,
                dict_model,
                ngram_weight=0.1,  # Paper uses 0.1 for n-gram
                word_weight=0.9    # Paper uses 0.9 for word dictionary
            )
        
        # Initialize state
        self.temperature = INITIAL_TEMPERATURE
        self.space_positions: Set[int] = self.initialize_space_positions()
        self.current_key = self._initialize_key()
        self.current_plaintext = self.decrypt()
        
        # Initialize caches with starting hypothesis
        if use_crp:
            assert isinstance(self.model, CRPJointModel)
            self.model.initialize_caches(self._ciphertext, self.current_plaintext)
            self.current_score = self.model.log_score(
                self._ciphertext, self.current_plaintext, self.current_key
            )
        
        # Track best solution
        self.best_key = self.current_key.copy()
        self.best_space_positions = self.space_positions.copy()
        self.best_plaintext = self.current_plaintext
        self.best_score = self.current_score
    
    def _convert_ground_truth_key(self, key_from_json: Dict[str, List[int]]) -> Dict[int, str]:
        """Convert ground truth key from JSON format to internal format."""
        inverted_key = {}
        for letter, symbols in key_from_json.items():
            for symbol in symbols:
                inverted_key[symbol] = letter.lower()
        return inverted_key
    
    def initialize_space_positions(self) -> Set[int]:
        """Initialize space positions with 2-6 character gaps."""
        positions: Set[int] = set()
        if not self._ciphertext:
            return positions
        
        min_gap, max_gap = 2, 6
        idx = random.randint(0, max_gap)
        text_len = len(self._ciphertext)
        
        while idx < text_len:
            positions.add(idx)
            idx += random.randint(min_gap, max_gap)
        
        return positions
    
    def _initialize_key(self) -> Dict[int, str]:
        """Initialize key by matching frequent cipher symbols to frequent letters."""
        freq_file = PROJECT_ROOT / "data" / "frequencies" / "english_letter_frequencies.json"
        with open(freq_file) as f:
            freq_data = json.load(f)
        
        sorted_english_letters = sorted(freq_data.keys(), key=lambda x: freq_data[x], reverse=True)
        sorted_english_letters = [char.lower() for char in sorted_english_letters]
        
        symbol_counts = Counter(self._ciphertext)
        sorted_cipher_symbols = [symbol for symbol, _ in symbol_counts.most_common()]
        
        substitution_key = {}
        num_symbols = len(sorted_cipher_symbols)
        symbol_idx = 0
        
        for letter in sorted_english_letters:
            expected_proportion = freq_data[letter.upper()]
            num_symbols_for_letter = max(1, round(expected_proportion * num_symbols))
            
            for _ in range(num_symbols_for_letter):
                if symbol_idx >= num_symbols:
                    break
                substitution_key[sorted_cipher_symbols[symbol_idx]] = letter
                symbol_idx += 1
            
            if symbol_idx >= num_symbols:
                break
        
        while symbol_idx < num_symbols:
            letter = sorted_english_letters[symbol_idx % len(sorted_english_letters)]
            substitution_key[sorted_cipher_symbols[symbol_idx]] = letter
            symbol_idx += 1
        
        return substitution_key
    
    def decrypt(self) -> str:
        """Decrypt ciphertext using current key and space positions."""
        return self._build_plaintext_from_key(self.current_key, self.space_positions)
    
    def _build_plaintext_from_key(self, key: Dict[int, str], space_positions: Set[int]) -> str:
        """Build plaintext from key and space positions."""
        plaintext_chars: List[str] = []
        for idx, char_code in enumerate(self._ciphertext):
            if idx in space_positions:
                plaintext_chars.append(" ")
            plaintext_chars.append(key[char_code])
        if len(self._ciphertext) in space_positions:
            plaintext_chars.append(" ")
        return "".join(plaintext_chars)
    
    def update_temperature(self, iteration: int, num_iterations: int) -> None:
        """Update temperature for simulated annealing (linear schedule)."""
        self.temperature = INITIAL_TEMPERATURE - (9.0 * iteration / num_iterations)
    
    def sample_key_pass(self) -> None:
        """Perform one pass of type sampling over all cipher symbols."""
        unique_symbols = list(set(self._ciphertext))
        random.shuffle(unique_symbols)
        
        plaintext_chars = "abcdefghijklmnopqrstuvwxyz"
        
        for symbol in unique_symbols:
            # Propose new mapping
            proposed_char = random.choice(plaintext_chars)
            
            # Create proposed key
            proposed_key = self.current_key.copy()
            proposed_key[symbol] = proposed_char
            
            # Generate proposed plaintext
            proposed_plaintext = self._build_plaintext_from_key(proposed_key, self.space_positions)
            
            # Calculate proposed score
            if self.use_crp:
                # For CRP, we need to rescore with the joint model
                # Clear and rebuild caches for the proposal
                assert isinstance(self.model, CRPJointModel)
                self.model.clear_caches()
                self.model.initialize_caches(self._ciphertext, proposed_plaintext)
                proposed_score = self.model.log_score(
                    self._ciphertext, proposed_plaintext, proposed_key
                )
            
            # Metropolis-Hastings acceptance
            log_acceptance_ratio = (proposed_score - self.current_score) / self.temperature
            
            if log_acceptance_ratio > 0.0 or math.log(random.uniform(0, 1)) < log_acceptance_ratio:
                # Accept proposal
                self.current_key = proposed_key
                self.current_plaintext = proposed_plaintext
                self.current_score = proposed_score
                
                # Update best if improved
                if self.current_score > self.best_score:
                    self.best_key = self.current_key.copy()
                    self.best_space_positions = self.space_positions.copy()
                    self.best_plaintext = self.current_plaintext
                    self.best_score = self.current_score
            else:
                # Proposal rejected - restore caches
                if self.use_crp:
                    assert isinstance(self.model, CRPJointModel)
                    self.model.clear_caches()
                    self.model.initialize_caches(self._ciphertext, self.current_plaintext)
    
    def sample_space_pass(self) -> None:
        """Perform one pass of space sampling."""
        num_proposals = max(1, len(self._ciphertext))
        
        for _ in range(num_proposals):
            # Pick random position to toggle space
            position = random.randint(0, len(self._ciphertext))
            
            proposed_space_positions = self.space_positions.copy()
            if position in proposed_space_positions:
                proposed_space_positions.remove(position)
            else:
                proposed_space_positions.add(position)
            
            # Generate proposed plaintext
            proposed_plaintext = self._build_plaintext_from_key(
                self.current_key, proposed_space_positions
            )
            
            # Calculate proposed score
            if self.use_crp:
                assert isinstance(self.model, CRPJointModel)
                self.model.clear_caches()
                self.model.initialize_caches(self._ciphertext, proposed_plaintext)
                proposed_score = self.model.log_score(
                    self._ciphertext, proposed_plaintext, self.current_key
                )
            
            # Metropolis-Hastings acceptance
            log_acceptance_ratio = (proposed_score - self.current_score) / self.temperature
            
            if log_acceptance_ratio > 0.0 or math.log(random.uniform(0, 1)) < log_acceptance_ratio:
                self.space_positions = proposed_space_positions
                self.current_plaintext = proposed_plaintext
                self.current_score = proposed_score
                
                if self.current_score > self.best_score:
                    self.best_key = self.current_key.copy()
                    self.best_space_positions = self.space_positions.copy()
                    self.best_plaintext = self.current_plaintext
                    self.best_score = self.current_score
            else:
                # Proposal rejected - restore caches
                if self.use_crp:
                    assert isinstance(self.model, CRPJointModel)
                    self.model.clear_caches()
                    self.model.initialize_caches(self._ciphertext, self.current_plaintext)
    
    def calculate_ser(self, key: Dict[int, str] | None = None) -> float:
        """Calculate Symbol Error Rate."""
        key_to_check = key if key is not None else self.best_key
        mismatches = sum(
            1 for symbol in key_to_check 
            if key_to_check[symbol] != self.ground_truth_key.get(symbol, None)
        )
        return mismatches / len(key_to_check)
    
    def run(self, num_iterations: int = TOTAL_ITERATIONS, log_interval: int = 100) -> Tuple[Dict[int, str], str]:
        """Run the main sampling loop."""
        logger.info(f"Starting CRP Bayesian sampling for {num_iterations} iterations...")
        logger.info(f"Using CRP: {self.use_crp}")
        logger.info(f"Initial score: {self.current_score:.2f}")
        
        initial_ser = self.calculate_ser()
        logger.info(f"Initial SER: {initial_ser:.4f} ({initial_ser*100:.2f}% errors)")
        
        # Log component scores for debugging
        if self.use_crp:
            assert isinstance(self.model, CRPJointModel)
            ngram_score, word_score, source_score, channel_score = self.model.log_score_separate(
                self._ciphertext, self.current_plaintext, self.current_key
            )
            logger.info(f"Initial CRP n-gram score: {ngram_score:.2f}")
            logger.info(f"Initial word dict score: {word_score:.2f}")
            logger.info(f"Initial interpolated P(p): {source_score:.2f}")
            logger.info(f"Initial channel P(c|p): {channel_score:.2f}")
        
        logger.info(f"Initial plaintext: {self.current_plaintext[:100]}...")
        
        for i in range(num_iterations):
            # Type sampling (key)
            self.sample_key_pass()
            
            # Space sampling
            self.sample_space_pass()
            
            # Update temperature
            self.update_temperature(i, num_iterations)
            
            # Log progress
            if i % log_interval == 0:
                current_ser = self.calculate_ser()
                logger.info(f"\nIteration {i}:")
                logger.info(f"  Temperature: {self.temperature:.2f}")
                logger.info(f"  Current score: {self.current_score:.2f}")
                logger.info(f"  Best score: {self.best_score:.2f}")
                logger.info(f"  Best SER: {current_ser:.4f} ({current_ser*100:.2f}% errors)")
                
                if self.use_crp:
                    assert isinstance(self.model, CRPJointModel)
                    ngram, word, source, channel = self.model.log_score_separate(
                        self._ciphertext, self.best_plaintext, self.best_key
                    )
                    logger.info(f"  Best n-gram: {ngram:.2f}, word: {word:.2f}, P(p): {source:.2f}, P(c|p): {channel:.2f}")
                
                logger.info(f"  Current plaintext: {self.current_plaintext[:100]}...")
                logger.info(f"  Best plaintext: {self.best_plaintext[:100]}...")
        
        logger.info("\nSampling complete!")
        logger.info(f"Final best score: {self.best_score:.2f}")
        
        final_ser = self.calculate_ser()
        logger.info(f"Final SER: {final_ser:.4f} ({final_ser*100:.2f}% symbol errors)")
        logger.info(f"Final best plaintext: {self.best_plaintext}")
        
        return self.best_key, self.best_plaintext
    
    def get_best_solution(self) -> Tuple[Dict[int, str], str]:
        """Return the best solution found."""
        return self.best_key, self.best_plaintext
