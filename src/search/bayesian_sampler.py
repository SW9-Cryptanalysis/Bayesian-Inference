import random
import math
import logging
import json
from typing import Counter, List, Dict, Tuple, Optional, Set
from lm_models.interpolated_model import InterpolatedLanguageModel
from utils.constants import TOTAL_ITERATIONS, INITIAL_TEMPERATURE, PROJECT_ROOT

logger = logging.getLogger(__name__)

class BayesianSampler:
    def __init__(self, ciphertext: List[int], lm_model: InterpolatedLanguageModel, ground_truth_key: Dict[str, List[int]]):
        """
        Initializes the Bayesian sampler with the ciphertext, current key, and language model.
        
        Args:
            ciphertext: List of integers representing the ciphertext.
            lm_model: An instance of InterpolatedLanguageModel for scoring plaintexts.
            ground_truth_key: Ground truth key mapping plaintext letters to cipher symbol lists (from cipher JSON).
        """
        self._ciphertext = ciphertext
        self.lm_model = lm_model
        self.ground_truth_key = self._convert_ground_truth_key(ground_truth_key)
        self.temperature = INITIAL_TEMPERATURE  # Initial temperature for simulated annealing
        self.space_positions: Set[int] = self.initialize_space_positions()
        self.current_key = self._initialize_key()
        self.current_plaintext = self.decrypt()
        self.current_score = lm_model.log_score_text(self.current_plaintext)
        
        # Track the best solution found
        self.best_key = self.current_key.copy()
        self.best_space_positions = self.space_positions.copy()
        self.best_plaintext = self.current_plaintext
        self.best_score = self.current_score
    
    def _convert_ground_truth_key(self, key_from_json: Dict[str, List[int]]) -> Dict[int, str]:
        """
        Converts the ground truth key from JSON format {letter: [symbols]} to 
        the format used internally {symbol: letter}.
        
        Args:
            key_from_json: Key from cipher JSON file mapping letters to lists of cipher symbols.
            
        Returns:
            Dict[int, str]: Inverted key mapping cipher symbols to plaintext letters.
        """
        inverted_key = {}
        for letter, symbols in key_from_json.items():
            for symbol in symbols:
                inverted_key[symbol] = letter.lower()
        return inverted_key

    def initialize_space_positions(self) -> Set[int]:
        """
        Initializes space positions so there are a space for every 2-6 characters.
        
        Returns:
            Set[int]: A set representing initial space positions.
        """
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
        """
        Initializes a substitution key by matching frequent cipher symbols to frequent plaintext letters.
        Uses a proportional allocation strategy: if 'e' represents 12.7% of English text and you have
        30 cipher symbols, approximately 3-4 symbols should map to 'e'. This helps homophonic cipher
        decryption converge faster than simple round-robin allocation.
        
        Returns:
            Dict[int, str]: A substitution key mapping cipher symbol codes to plaintext characters.
        """
        freq_file = PROJECT_ROOT / 'data' / 'frequencies' / 'english_letter_frequencies.json'
        with open(freq_file, 'r') as f:
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
        """
        Decrypts the ciphertext using the current substitution key and stored space positions.
        
        Returns:
            str: The decrypted plaintext string with spaces inserted.
        """
        return self._build_plaintext_from_key(self.current_key, self.space_positions)

    def _build_plaintext_from_key(self, key: Dict[int, str], space_positions: Set[int]) -> str:
        """Constructs plaintext by applying a key and inserting spaces at recorded positions."""
        plaintext_chars: List[str] = []
        for idx, char_code in enumerate(self._ciphertext):
            if idx in space_positions:
                plaintext_chars.append(' ')
            plaintext_chars.append(key[char_code])
        if len(self._ciphertext) in space_positions:
            plaintext_chars.append(' ')
        return ''.join(plaintext_chars)
    
    def update_temperature(self, iteration: int, num_iterations: int) -> None:
        """
        Updates the temperature for simulated annealing using a linear schedule.
        Temperature decreases from INITIAL_TEMPERATURE to 1.0 over the course of sampling.
        
        Args:
            iteration: Current iteration number.
            num_iterations: Total number of iterations to run.
        """
        self.temperature = INITIAL_TEMPERATURE - (9.0 * iteration / num_iterations)
    
    def sample_key_pass(self) -> None:
        """
        Performs one pass of type sampling.
        Resamples the key by iterating through all unique cipher symbols
        in random order and proposing new character mappings.
        """
        unique_symbols = list(set(self._ciphertext))
        random.shuffle(unique_symbols)
        
        plaintext_chars = 'abcdefghijklmnopqrstuvwxyz'
        
        for symbol in unique_symbols:
            # Propose a new mapping for this symbol
            proposed_char = random.choice(plaintext_chars)
            
            # Create proposed key
            proposed_key = self.current_key.copy()
            proposed_key[symbol] = proposed_char
            
            # Generate proposed plaintext
            proposed_plaintext = self._build_plaintext_from_key(proposed_key, self.space_positions)
            
            # Calculate proposed score
            proposed_score = self.lm_model.log_score_text(proposed_plaintext)
            
            # Metropolis-Hastings acceptance
            log_acceptance_ratio = (proposed_score - self.current_score) / self.temperature
            
            # Accept if better, or with probability based on score difference
            if log_acceptance_ratio > 0.0 or math.log(random.uniform(0, 1)) < log_acceptance_ratio:
                self.current_key = proposed_key
                self.current_plaintext = proposed_plaintext
                self.current_score = proposed_score
                
                # Update best solution if this is the best we've seen
                if self.current_score > self.best_score:
                    self.best_key = self.current_key.copy()
                    self.best_space_positions = self.space_positions.copy()
                    self.best_plaintext = self.current_plaintext
                    self.best_score = self.current_score
    
    def calculate_ser(self, key: Optional[Dict[int, str]] = None) -> float:
        """
        Calculates the Symbol Error Rate (SER) by comparing a key against the ground truth.
        SER is the proportion of cipher symbols that are incorrectly mapped.
        
        Args:
            key: The key to evaluate. If None, uses the current best key.
        
        Returns:
            The SER value between 0.0 (perfect) and 1.0 (all wrong).
        """
        key_to_check = key if key is not None else self.best_key
        mismatches = sum(1 for symbol in key_to_check if key_to_check[symbol] != self.ground_truth_key.get(symbol, None))
        return mismatches / len(key_to_check)
    
    def sample_space_pass(self) -> None:
        """
        Performs one pass of space sampling.
        Proposes inserting or removing spaces at random positions
        to find word boundaries.
        """
        num_proposals = max(1, len(self._ciphertext))
        available_positions = len(self._ciphertext) + 1
        
        for _ in range(num_proposals):
            if available_positions == 0:
                continue
            
            # Pick a random insertion/removal point between symbols (inclusive of ends)
            position = random.randint(0, len(self._ciphertext))
            proposed_space_positions = set(self.space_positions)
            if position in proposed_space_positions:
                proposed_space_positions.remove(position)
            else:
                proposed_space_positions.add(position)
            
            # Generate proposed plaintext with updated spaces
            proposed_plaintext = self._build_plaintext_from_key(self.current_key, proposed_space_positions)
            
            # Calculate proposed score
            proposed_score = self.lm_model.log_score_text(proposed_plaintext)
            
            # Metropolis-Hastings acceptance
            log_acceptance_ratio = (proposed_score - self.current_score) / self.temperature
            
            # Accept if better, or with probability based on score difference
            if log_acceptance_ratio > 0.0 or math.log(random.uniform(0, 1)) < log_acceptance_ratio:
                self.space_positions = proposed_space_positions
                self.current_plaintext = proposed_plaintext
                self.current_score = proposed_score
                
                # Update best solution if this is the best we've seen
                if self.current_score > self.best_score:
                    self.best_key = self.current_key.copy()
                    self.best_space_positions = self.space_positions.copy()
                    self.best_plaintext = self.current_plaintext
                    self.best_score = self.current_score
    
    def run(self, num_iterations: int = TOTAL_ITERATIONS, log_interval: int = 100) -> Tuple[Dict[int, str], str]:
        """
        Runs the main sampling loop, alternating between key and space sampling.
        
        Args:
            num_iterations: Number of iterations to run (default: TOTAL_ITERATIONS).
            log_interval: How often to log progress (default: every 100 iterations).
            
        Returns:
            Tuple of (best_key, best_plaintext).
        """
        logger.info(f"Starting Bayesian sampling for {num_iterations} iterations...")
        logger.info(f"Initial score: {self.current_score:.2f}")
        
        # Log initial SER
        initial_ser = self.calculate_ser()
        logger.info(f"Initial SER: {initial_ser:.4f} ({initial_ser*100:.2f}% errors)")
        
        # Log component scores for debugging
        ngram_score = self.lm_model.ngram_lm.log_score_text(self.current_plaintext)
        dict_score = self.lm_model.dict_lm.log_score_text(self.current_plaintext)
        logger.info(f"Initial n-gram score: {ngram_score:.2f}")
        logger.info(f"Initial dict score: {dict_score:.2f}")
        
        logger.info(f"Initial plaintext: {self.current_plaintext[:100]}...")
        
        for i in range(num_iterations):
            # Move 1: Resample the Key (Type Sampling)
            self.sample_key_pass()
            
            # Move 2: Resample the Spaces (Space Operator)
            self.sample_space_pass()
            
            # Update temperature (simulated annealing)
            self.update_temperature(i, num_iterations)
            
            # Log progress
            if i % log_interval == 0:
                current_ser = self.calculate_ser()
                best_ngram = self.lm_model.ngram_lm.log_score_text(self.best_plaintext)
                best_dict = self.lm_model.dict_lm.log_score_text(self.best_plaintext)
                
                logger.info(f"\nIteration {i}:")
                logger.info(f"  Temperature: {self.temperature:.2f}")
                logger.info(f"  Current score: {self.current_score:.2f}")
                logger.info(f"  Best score: {self.best_score:.2f}")
                logger.info(f"  Best SER: {current_ser:.4f} ({current_ser*100:.2f}% errors)")
                logger.info(f"  Best n-gram: {best_ngram:.2f}, Best dict: {best_dict:.2f}")
                logger.info(f"  Current plaintext: {self.current_plaintext[:100]}...")
                logger.info(f"  Best plaintext: {self.best_plaintext[:100]}...")
        
        logger.info(f"\nSampling complete!")
        logger.info(f"Final best score: {self.best_score:.2f}")
        
        # Log final SER
        final_ser = self.calculate_ser()
        logger.info(f"Final SER: {final_ser:.4f} ({final_ser*100:.2f}% symbol errors)")
        
        logger.info(f"Final best plaintext: {self.best_plaintext}")
        
        return self.best_key, self.best_plaintext
    
    def get_best_solution(self) -> Tuple[Dict[int, str], str]:
        """
        Returns the best solution found during sampling.
        
        Returns:
            Tuple of (best_key, best_plaintext).
        """
        return self.best_key, self.best_plaintext