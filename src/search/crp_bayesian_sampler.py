"""
CRP-based Bayesian sampler for substitution cipher decipherment.

This is the main implementation following the paper's approach using
Chinese Restaurant Process formulations for both source and channel models.

Key Implementation Details (matching paper):
1. CRP Formulas:
   - Source: P(pi|pi-1) = (α·P0 + C(pi-1,pi)) / (α + C(pi-1))
   - Channel: P(ci|pi) = (β·P0 + C(pi,ci)) / (β + C(pi))
   - α = 10,000 (high, favors base LM), β = 0.01 (low, favors deterministic substitution)

2. Incremental Scoring with Exchangeability:
   - When sampling at position i, affected region is scored separately
   - Cache is saved before proposal, restored on rejection (NOT rebuilt!)
   - Only affected context window is rescored, not entire plaintext
   - This is the key efficiency optimization from the paper

3. Type Sampling:
   - Sample plaintext letter for each cipher symbol type (not position-wise)
   - All occurrences of a symbol are updated simultaneously
   - Uses Metropolis-Hastings acceptance with simulated annealing

4. Space Sampling:
   - Second pass that adds/removes word boundaries
   - Also uses incremental scoring for efficiency
   - Enables decipherment of ciphers without spaces

5. Interpolated Language Model:
   - P(p) = 0.1 × P_ngram(p) + 0.9 × P_word(p)
   - Combines character n-grams with word dictionary
   - Makes model robust to variations and misspellings

6. Cache Management (Critical Fix):
   - Caches are NEVER cleared/rebuilt during sampling except initialization
   - Proposals modify cache temporarily; restoration on rejection
   - This maintains efficiency and correctness of CRP inference
"""

import random
import math
import logging
import json
from collections import Counter
from typing import Tuple, Dict, List, Set

from lm_models.crp_joint_model import CRPJointModel
from lm_models.crp_source_model import CRPSourceModel
from lm_models.crp_channel_model import CRPChannelModel
from lm_models.n_gram_model import NgramLanguageModel
from lm_models.dictionary_model import DictionaryLanguageModel
from lm_models.incremental_scorer import IncrementalScorer
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
            use_crp: If True, use CRP models with incremental scoring; if False, use standard models
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
            
            # Always use incremental scorer with CRP for efficiency
            self.incremental_scorer = IncrementalScorer(self.crp_source, self.crp_channel)
        
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
            
            # Track component scores separately for incremental updates
            ngram, word, source, channel = self.model.log_score_separate(
                self._ciphertext, self.current_plaintext, self.current_key
            )
            self.current_ngram_score = ngram
            self.current_word_score = word
            self.current_source_score = source
            self.current_channel_score = channel
        
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
        """Perform one pass of type sampling over all cipher symbols.
        
        Uses proper incremental scoring with exchangeability property to efficiently
        evaluate proposals without rescoring the entire plaintext.
        """
        unique_symbols = list(set(self._ciphertext))
        random.shuffle(unique_symbols)
        
        plaintext_chars = "abcdefghijklmnopqrstuvwxyz"
        
        for symbol in unique_symbols:
            # Propose new mapping
            proposed_char = random.choice(plaintext_chars)
            
            # Skip if this is already the current mapping
            if self.current_key.get(symbol) == proposed_char:
                continue
            
            # Create proposed key
            proposed_key = self.current_key.copy()
            proposed_key[symbol] = proposed_char
            
            # Generate proposed plaintext
            proposed_plaintext = self._build_plaintext_from_key(proposed_key, self.space_positions)
            
            # Calculate proposed score
            if self.use_crp:
                assert isinstance(self.model, CRPJointModel)
                
                # Get current component scores (n-gram only, no word dict yet)
                current_ngram_score = self.current_score  # This is just n-gram for now
                current_channel_score = self.current_channel_score  # Track separately
                
                # Use incremental scoring with proper exchangeability
                # Note: score_key_proposal now uses copies of caches, so main caches are safe
                new_source_score, new_channel_score, source_delta, channel_delta = \
                    self.incremental_scorer.score_key_proposal(
                        self._ciphertext, self.current_plaintext, proposed_plaintext,
                        self.current_key, proposed_key, symbol, self.space_positions,
                        current_ngram_score, current_channel_score
                    )
                
                # For now, word dictionary component is computed separately (TODO: make incremental)
                # This is acceptable since word scoring is lightweight compared to CRP n-gram scoring
                if self.model.dict_model is not None:
                    old_word_score = self.model.dict_model.log_score_text(self.current_plaintext)
                    new_word_score = self.model.dict_model.log_score_text(proposed_plaintext)
                    word_delta = new_word_score - old_word_score
                else:
                    word_delta = 0.0
                
                # Combine using paper's interpolation weights (0.1 n-gram, 0.9 word)
                if self.model.dict_model is not None:
                    new_word_score = old_word_score + word_delta
                    combined_source_score = 0.1 * new_source_score + 0.9 * new_word_score
                else:
                    combined_source_score = new_source_score
                proposed_score = combined_source_score + new_channel_score
            
            # Metropolis-Hastings acceptance
            log_acceptance_ratio = (proposed_score - self.current_score) / self.temperature
            
            if log_acceptance_ratio > 0.0 or math.log(random.uniform(0, 1)) < log_acceptance_ratio:
                # Accept proposal
                
                # Update caches if using CRP
                if self.use_crp:
                    self.incremental_scorer.apply_key_changes_to_cache(
                        self._ciphertext, self.current_plaintext, proposed_plaintext,
                        self.current_key, proposed_key, symbol, self.space_positions
                    )
                    # Update tracked component scores
                    self.current_ngram_score = new_source_score
                    self.current_channel_score = new_channel_score

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
                # Proposal rejected - no cache restoration needed as we used copies
                pass
    
    def _get_plaintext_index(self, cipher_idx: int, space_positions: Set[int]) -> int:
        """Get the index in plaintext corresponding to the cipher symbol at cipher_idx."""
        # Count spaces that appear before or at this index (since space is inserted before char)
        num_spaces = sum(1 for p in space_positions if p <= cipher_idx)
        return cipher_idx + num_spaces

    def sample_space_pass(self) -> None:
        """Perform one pass of space sampling.
        
        Iterates over all adjacent character pairs and samples whether to insert/remove a space.
        Uses incremental scoring with exchangeability.
        """
        # Iterate over all possible space positions (between characters)
        # We skip index 0 (start of text) as spaces are usually between words
        possible_positions = list(range(1, len(self._ciphertext)))
        random.shuffle(possible_positions)
        
        for position in possible_positions:
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
                
                # Save cache states
                saved_source_cache = self.crp_source.get_cache_copy()
                # Channel score is invariant to spaces (as it ignores them), so we don't need to update it
                
                # Find affected region around the space change
                # We need indices in both old and new plaintext
                old_plain_idx = self._get_plaintext_index(position, self.space_positions)
                new_plain_idx = self._get_plaintext_index(position, proposed_space_positions)
                
                context_size = self.crp_source.n + 5  # Sufficient context
                
                # Define window in OLD plaintext
                old_start = max(0, old_plain_idx - context_size)
                old_end = min(len(self.current_plaintext), old_plain_idx + context_size)
                
                # Define window in NEW plaintext
                new_start = max(0, new_plain_idx - context_size)
                new_end = min(len(proposed_plaintext), new_plain_idx + context_size)
                
                # Prepare SHARED cache by removing old window
                # This implements "pretend affected area is at the end"
                shared_source_cache = saved_source_cache.copy()
                self.incremental_scorer._remove_window_from_cache(
                    self.current_plaintext, old_start, old_end, shared_source_cache
                )
                
                # Score old window using shared cache
                old_window_score = self.incremental_scorer.score_window_source(
                    self.current_plaintext, old_start, old_end, shared_source_cache.copy()
                )
                
                # Score new window using shared cache
                new_window_score = self.incremental_scorer.score_window_source(
                    proposed_plaintext, new_start, new_end, shared_source_cache.copy()
                )
                
                source_delta = new_window_score - old_window_score
                new_ngram_score = self.current_ngram_score + source_delta
                
                # Channel score unchanged
                new_channel_score = self.current_channel_score
                
                # Word dictionary score (recompute - it's fast)
                if self.model.dict_model is not None:
                    new_word_score = self.model.dict_model.log_score_text(proposed_plaintext)
                else:
                    new_word_score = 0.0
                
                # Combine using interpolation weights
                if self.model.dict_model is not None:
                    combined_source_score = 0.1 * new_ngram_score + 0.9 * new_word_score
                else:
                    combined_source_score = new_ngram_score
                    
                proposed_score = combined_source_score + new_channel_score
            
            # Metropolis-Hastings acceptance
            log_acceptance_ratio = (proposed_score - self.current_score) / self.temperature
            
            if log_acceptance_ratio > 0.0 or math.log(random.uniform(0, 1)) < log_acceptance_ratio:
                # Accept proposal
                
                if self.use_crp:
                    self.incremental_scorer.apply_space_changes_to_cache(
                        self.current_plaintext, proposed_plaintext,
                        old_start, old_end, new_start, new_end
                    )
                    self.current_ngram_score = new_ngram_score
                    self.current_channel_score = new_channel_score
                
                self.space_positions = proposed_space_positions
                self.current_plaintext = proposed_plaintext
                self.current_score = proposed_score
                
                if self.current_score > self.best_score:
                    self.best_key = self.current_key.copy()
                    self.best_space_positions = self.space_positions.copy()
                    self.best_plaintext = self.current_plaintext
                    self.best_score = self.current_score
            else:
                # Proposal rejected - no cache restoration needed as we used copies
                pass
    
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
