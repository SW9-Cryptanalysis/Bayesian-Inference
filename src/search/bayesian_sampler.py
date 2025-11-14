import random
import math
import logging
import json
from typing import List, Dict, Tuple
from collections import Counter
from lm_models.interpolated_model import InterpolatedLanguageModel
from utils.constants import (
    TOTAL_ITERATIONS,
    INITIAL_TEMPERATURE,
    PLAINTEXT_LENGTH_TO_SHOW,
    LOG_INTERVAL,
    MIN_SPACE_PROPOSALS,
    MAX_SPACE_PROPOSALS,
    SPACE_PROPOSAL_RATIO,
    PLAINTEXT_ALPHABET,
    PROJECT_ROOT
)

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
        self._lm_model = lm_model
        self._ground_truth_key = self._convert_ground_truth_key(ground_truth_key)
        self._temperature: float = INITIAL_TEMPERATURE  # Initial temperature for simulated annealing
        self._unique_symbols = list(set(ciphertext))
        self._current_key = self._initialize_key()
        self._space_positions: List[int] = []
        
        # Calculate dynamic number of space proposals based on cipher length
        self._num_space_proposals = min(MAX_SPACE_PROPOSALS, max(MIN_SPACE_PROPOSALS, int(len(ciphertext) * SPACE_PROPOSAL_RATIO)))
        
        self._current_plaintext = self._decrypt()
        self._current_score = lm_model.log_score_text(self._current_plaintext)
        self._best_key = self._current_key.copy()
        self._best_space_positions = self._space_positions.copy()
        self._best_plaintext = self._current_plaintext
        self._best_score = self._current_score
    
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
    
    def _build_plaintext(self, key: Dict[int, str], space_positions: List[int]) -> str:
        """
        Helper method to build plaintext from a given key and space positions.
        
        Args:
            key: The substitution key to use for decryption.
            space_positions: Sorted list of positions (in unspaced text) where spaces should be inserted.
            
        Returns:
            str: The plaintext with spaces inserted at specified positions.
        """
        unspaced = ''.join(key[char_code] for char_code in self._ciphertext)
        
        if not space_positions:
            return unspaced
        
        # Build plaintext with spaces using offset tracking to handle position shifts
        result = []
        space_idx = 0
        for i, char in enumerate(unspaced):
            # Insert spaces at positions from the sorted list
            while space_idx < len(space_positions) and space_positions[space_idx] == i:
                result.append(' ')
                space_idx += 1
            result.append(char)
        
        return ''.join(result)
        
    def _decrypt(self) -> str:
        """
        Decrypts the ciphertext using the current substitution key and space positions.
        
        Returns:
            str: The decrypted plaintext string with spaces inserted at tracked positions.
        """
        return self._build_plaintext(self._current_key, self._space_positions)
    
    def _accept_proposal(self, proposed_score: float, proposed_key: Dict[int, str], 
                        proposed_space_positions: List[int], proposed_plaintext: str) -> bool:
        """
        Metropolis-Hastings acceptance criterion for proposed state.
        Updates current and best state if proposal is accepted.
        
        Args:
            proposed_score: The language model score of the proposed plaintext.
            proposed_key: The proposed substitution key.
            proposed_space_positions: The proposed space positions.
            proposed_plaintext: The proposed plaintext string.
            
        Returns:
            bool: True if proposal was accepted, False otherwise.
        """
        log_acceptance_ratio = (proposed_score - self._current_score) / self._temperature
        
        # Accept if better, or with probability based on score difference
        if log_acceptance_ratio >= 0.0 or math.log(max(random.uniform(0, 1), 1e-10)) < log_acceptance_ratio:
            self._current_key = proposed_key
            self._space_positions = proposed_space_positions
            self._current_plaintext = proposed_plaintext
            self._current_score = proposed_score
            
            if self._current_score > self._best_score:
                self._best_key = self._current_key.copy()
                self._best_space_positions = self._space_positions.copy()
                self._best_plaintext = proposed_plaintext
                self._best_score = self._current_score
            
            return True
        
        return False
        
    def _update_temperature(self, iteration: int, num_iterations: int) -> None:
        """
        Updates the temperature for simulated annealing using a linear schedule.
        Temperature decreases from INITIAL_TEMPERATURE to 1.0 over the course of sampling.
        
        Args:
            iteration: Current iteration number (0-indexed).
            num_iterations: Total number of iterations to run.
        """
        self._temperature = INITIAL_TEMPERATURE - ((INITIAL_TEMPERATURE - 1.0) * (iteration + 1) / num_iterations)
    
    def _sample_key_pass(self) -> None:
        """
        Performs one pass of key sampling.
        Resamples the key by iterating through all unique cipher symbols
        in random order and proposing new character mappings.
        For homophonic ciphers, multiple symbols can map to the same plaintext character.
        """
        symbols_to_sample = self._unique_symbols.copy()
        random.shuffle(symbols_to_sample)
        
        for symbol in symbols_to_sample:
            proposed_char = random.choice(PLAINTEXT_ALPHABET)
             
            proposed_key = self._current_key.copy()
            proposed_key[symbol] = proposed_char
            
            proposed_plaintext = self._build_plaintext(proposed_key, self._space_positions)
            proposed_score = self._lm_model.log_score_text(proposed_plaintext)
            self._accept_proposal(proposed_score, proposed_key, self._space_positions, proposed_plaintext)
    
    def _sample_space_pass(self) -> None:
        """
        Performs one pass of space sampling.
        Proposes inserting or removing spaces at random positions
        to find word boundaries. Operates on space positions in the unspaced plaintext.
        Limited to a fixed number of proposals per pass based on cipher length and constants
        to prevent too much computation.
        """
        for _ in range(self._num_space_proposals):
            # Pick a random position in the unspaced plaintext
            i = random.randint(0, len(self._ciphertext) - 1)
            
            # Propose space insertion or removal
            proposed_space_positions = self._space_positions.copy()
            if i in proposed_space_positions:
                # Remove the space
                proposed_space_positions.remove(i)
            else:
                # Insert a space before position i
                proposed_space_positions.append(i)
                proposed_space_positions.sort()  # Keep sorted for correct insertion order
            
            # Generate proposed plaintext with current key
            proposed_plaintext = self._build_plaintext(self._current_key, proposed_space_positions)
            
            # Calculate proposed score
            proposed_score = self._lm_model.log_score_text(proposed_plaintext)
            
            # Apply Metropolis-Hastings acceptance
            self._accept_proposal(proposed_score, self._current_key, proposed_space_positions, proposed_plaintext)
            
    def calculate_SER(self) -> float:
        """
        Calculates the Symbol Error Rate (SER) by comparing a key against the ground truth.
        SER is the proportion of cipher symbols that are incorrectly mapped.
        
        Args:
            key: The key to evaluate. If None, uses the current best key.
        
        Returns:
            The SER value between 0.0 (perfect) and 1.0 (all wrong).
        """
        key_to_check = self._best_key
        mismatches = sum(1 for symbol in key_to_check if key_to_check[symbol] != self._ground_truth_key.get(symbol, None))
        return mismatches / len(key_to_check)
    
    def run(self, num_iterations: int = TOTAL_ITERATIONS, log_interval: int = LOG_INTERVAL) -> Tuple[Dict[int, str], str, List[int]]:
        """
        Runs the main sampling loop, alternating between key and space sampling.
        
        Args:
            num_iterations: Number of iterations to run (default: TOTAL_ITERATIONS).
            log_interval: How often to log progress (default: every LOG_INTERVAL iterations).
            
        Returns:
            Tuple of (best_key, best_plaintext, best_space_positions).
        """
        logger.info(f"Starting Bayesian sampling for {num_iterations} iterations...")
        logger.info(f"Initial score: {self._current_score}")
        logger.info(f"Initial plaintext: {self._current_plaintext[:PLAINTEXT_LENGTH_TO_SHOW]}...")
        logger.info(f"Initial key: {self._current_key}")
        
        for i in range(num_iterations):
            # Move 1: Resample the Key (Type Sampling)
            self._sample_key_pass()
            
            # Move 2: Resample the Spaces (Space Operator)
            self._sample_space_pass()
            
            # Update temperature (simulated annealing)
            self._update_temperature(i, num_iterations)
            
            # Log progress
            if i % log_interval == 0:
                logger.info(f"\nIteration {i}:")
                logger.info(f"  Temperature: {self._temperature:.2f}")
                logger.info(f"  Current score: {self._current_score:.2f}")
                logger.info(f"  Best score: {self._best_score:.2f}")
                logger.info(f"  Best SER: {self.calculate_SER():.4f} ({self.calculate_SER()*100:.2f}% errors)")
                
                logger.info(f"  Current plaintext: {self._current_plaintext[:PLAINTEXT_LENGTH_TO_SHOW]}...")
                logger.info(f"  Best plaintext: {self._best_plaintext[:PLAINTEXT_LENGTH_TO_SHOW]}...")
        
        logger.info(f"\nSampling complete!")
        logger.info(f"Final best score: {self._best_score}")
        
        # Log final SER
        final_ser = self.calculate_SER()
        logger.info(f"Final SER: {final_ser:.4f} ({final_ser*100:.2f}% symbol errors)")
        
        logger.info(f"Final best plaintext: {self._best_plaintext}")
        
        return self._best_key, self._best_plaintext, self._best_space_positions
    
    def get_best_solution(self) -> Tuple[Dict[int, str], str, List[int]]:
        """
        Returns the best solution found during sampling.
        
        Returns:
            Tuple of (best_key, best_plaintext, best_space_positions).
        """
        return self._best_key, self._best_plaintext, self._best_space_positions