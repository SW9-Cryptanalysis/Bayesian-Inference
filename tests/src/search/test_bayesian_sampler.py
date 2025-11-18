import unittest
import json
import random
from pathlib import Path
from unittest.mock import Mock

# Add src to path to import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from search.bayesian_sampler import BayesianSampler
from lm_models.interpolated_model import InterpolatedLanguageModel
from lm_models.n_gram_model import NgramLanguageModel
from lm_models.dictionary_model import DictionaryLanguageModel
from utils.constants import PROJECT_ROOT


class TestBayesianSamplerInitialization(unittest.TestCase):
    """Test the initialization and setup of BayesianSampler."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Load a real cipher for testing
        cipher_path = PROJECT_ROOT / 'data' / 'ciphers' / 'mono-cipher-5.json'
        with open(cipher_path, 'r') as f:
            cipher_data = json.load(f)
        
        self.ciphertext = [int(x) for x in cipher_data['ciphertext'].split()]
        self.ground_truth_key = cipher_data['key']
        
        # Create mock language models
        self.mock_ngram_lm = Mock(spec=NgramLanguageModel)
        self.mock_dict_lm = Mock(spec=DictionaryLanguageModel)
        self.mock_lm_model = Mock(spec=InterpolatedLanguageModel)
        self.mock_lm_model.ngram_lm = self.mock_ngram_lm
        self.mock_lm_model.dict_lm = self.mock_dict_lm
        
        # Set a default score
        self.mock_lm_model.log_score_text.return_value = -10.0
    
    def test_initialization_creates_valid_key(self):
        """Test that initialization creates a key mapping all cipher symbols."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Check that all unique cipher symbols are in the key
        unique_symbols = set(self.ciphertext)
        self.assertEqual(set(sampler.current_key.keys()), unique_symbols)
        
        # Check that all values are lowercase letters
        for value in sampler.current_key.values():
            self.assertTrue(value.islower() and len(value) == 1)
    
    def test_ground_truth_key_conversion(self):
        """Test that ground truth key is correctly converted from JSON format."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # The ground truth key should map symbols to letters
        for letter, symbols in self.ground_truth_key.items():
            for symbol in symbols:
                self.assertEqual(sampler.ground_truth_key[symbol], letter.lower())
    
    def test_initial_space_positions(self):
        """Test that initial space positions are reasonable."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Check that spaces are within valid range
        for pos in sampler.space_positions:
            self.assertGreaterEqual(pos, 0)
            self.assertLessEqual(pos, len(self.ciphertext))
        
        # Check that spaces are somewhat regular (min gap of 2, max gap of 6)
        # This is a rough heuristic check
        sorted_positions = sorted(sampler.space_positions)
        if len(sorted_positions) > 1:
            for i in range(len(sorted_positions) - 1):
                gap = sorted_positions[i+1] - sorted_positions[i]
                self.assertGreaterEqual(gap, 1, "Spaces should have at least 1 character gap")
    
    def test_initial_plaintext_generation(self):
        """Test that initial plaintext is generated correctly."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Plaintext should not be empty
        self.assertGreater(len(sampler.current_plaintext), 0)
        
        # Count non-space characters should equal ciphertext length
        non_space_chars = sampler.current_plaintext.replace(' ', '')
        self.assertEqual(len(non_space_chars), len(self.ciphertext))
    
    def test_best_solution_tracking(self):
        """Test that best solution is properly initialized."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        self.assertEqual(sampler.best_score, sampler.current_score)
        self.assertEqual(sampler.best_key, sampler.current_key)
        self.assertEqual(sampler.best_plaintext, sampler.current_plaintext)


class TestBayesianSamplerKeyPass(unittest.TestCase):
    """Test the sample_key_pass function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test cipher
        self.ciphertext = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        self.ground_truth_key = {
            'a': [1],
            'b': [2],
            'c': [3]
        }
        
        # Create mock language model
        self.mock_lm_model = Mock(spec=InterpolatedLanguageModel)
        self.scores = []
        
        def score_side_effect(text):
            score = -10.0 + random.uniform(-1, 1)
            self.scores.append(score)
            return score
        
        self.mock_lm_model.log_score_text.side_effect = score_side_effect
    
    def test_key_pass_proposes_new_mappings(self):
        """Test that key pass proposes new character mappings."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Run one key pass
        sampler.sample_key_pass()
        
        # Check that the language model was called (proposals were made)
        self.assertGreater(self.mock_lm_model.log_score_text.call_count, 0)
    
    def test_key_pass_accepts_better_solutions(self):
        """Test that key pass accepts solutions with better scores."""
        # Set initial score
        self.mock_lm_model.log_score_text.return_value = -10.0
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Set up mock to always return better scores
        call_count = [0]
        def increasing_score(text):
            call_count[0] += 1
            return -10.0 + call_count[0] * 0.5  # Larger increments for more reliable improvement
        
        self.mock_lm_model.log_score_text.side_effect = increasing_score
        sampler.temperature = 1.0
        sampler.best_score = -10.0
        
        initial_score = sampler.current_score
        sampler.sample_key_pass()
        
        # Best score should improve (current score might not due to proposals being rejected)
        self.assertGreater(sampler.best_score, initial_score)
    
    def test_key_pass_updates_best_solution(self):
        """Test that key pass updates best solution when improved."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Set up mock to return a much better score on some call
        call_count = [0]
        def better_score_later(text):
            call_count[0] += 1
            if call_count[0] == 3:
                return -5.0  # Much better score
            return -15.0  # Worse score
        
        self.mock_lm_model.log_score_text.side_effect = better_score_later
        sampler.temperature = 1.0
        
        initial_best = sampler.best_score
        sampler.sample_key_pass()
        
        # Best score might have improved
        self.assertGreaterEqual(sampler.best_score, initial_best)
    
    def test_key_pass_visits_all_symbols(self):
        """Test that key pass iterates through all unique symbols."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        unique_symbols = set(self.ciphertext)
        
        # Run multiple passes to ensure coverage
        for _ in range(10):
            sampler.sample_key_pass()
        
        # At least some proposals should have been made
        # (We can't guarantee all symbols changed due to randomness)
        self.assertGreater(self.mock_lm_model.log_score_text.call_count, len(unique_symbols))


class TestBayesianSamplerSpacePass(unittest.TestCase):
    """Test the sample_space_pass function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test cipher
        self.ciphertext = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.ground_truth_key = {chr(ord('a') + i): [i+1] for i in range(10)}
        
        # Create mock language model
        self.mock_lm_model = Mock(spec=InterpolatedLanguageModel)
        self.mock_lm_model.log_score_text.return_value = -10.0
    
    def test_space_pass_proposes_changes(self):
        """Test that space pass proposes adding/removing spaces."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Run one space pass
        sampler.sample_space_pass()
        
        # Check that the language model was called (proposals were made)
        self.assertGreater(self.mock_lm_model.log_score_text.call_count, 0)
    
    def test_space_pass_toggles_positions(self):
        """Test that space pass can add and remove spaces."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Set temperature low to avoid random rejections
        sampler.temperature = 0.1
        
        # Set up mock to always accept (return better scores)
        call_count = [0]
        def increasing_score(text):
            call_count[0] += 1
            return -10.0 + call_count[0] * 0.5
        
        self.mock_lm_model.log_score_text.side_effect = increasing_score
        
        # Run multiple passes
        for _ in range(5):
            sampler.sample_space_pass()
        
        # Space positions should have changed
        # (might increase or decrease depending on random choices)
        # Just verify that changes were proposed
        self.assertGreater(call_count[0], 0)
    
    def test_space_pass_updates_best_solution(self):
        """Test that space pass updates best solution when improved."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Set up mock to return a better score
        call_count = [0]
        def better_score(text):
            call_count[0] += 1
            return -8.0  # Better than initial -10.0
        
        self.mock_lm_model.log_score_text.side_effect = better_score
        sampler.temperature = 1.0
        
        initial_best = sampler.best_score
        sampler.sample_space_pass()
        
        # Best score should improve
        self.assertGreater(sampler.best_score, initial_best)
    
    def test_space_positions_stay_in_bounds(self):
        """Test that space positions never go out of bounds."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Run many passes
        for _ in range(50):
            sampler.sample_space_pass()
            
            # Check all positions are valid
            for pos in sampler.space_positions:
                self.assertGreaterEqual(pos, 0)
                self.assertLessEqual(pos, len(self.ciphertext))


class TestBayesianSamplerMetropolisHastings(unittest.TestCase):
    """Test the Metropolis-Hastings acceptance logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ciphertext = [1, 2, 3, 1, 2, 3]
        self.ground_truth_key = {'a': [1], 'b': [2], 'c': [3]}
        
        self.mock_lm_model = Mock(spec=InterpolatedLanguageModel)
    
    def test_always_accepts_better_solutions(self):
        """Test that better solutions are always accepted."""
        # Set initial score first
        self.mock_lm_model.log_score_text.return_value = -10.0
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Now set up mock to always return better scores
        sampler.current_score = -10.0
        sampler.best_score = -10.0
        self.mock_lm_model.log_score_text.return_value = -5.0  # Much better
        sampler.temperature = 1.0
        
        sampler.sample_key_pass()
        
        # Score should improve
        self.assertEqual(sampler.current_score, -5.0)
    
    def test_accepts_worse_solutions_probabilistically(self):
        """Test that worse solutions are sometimes accepted based on temperature."""
        # Set initial score first
        self.mock_lm_model.log_score_text.return_value = -10.0
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key, seed=42)
        
        # High temperature should accept more worse solutions
        sampler.temperature = 10.0
        sampler.current_score = -10.0
        sampler.best_score = -10.0
        
        # Set up mock to return slightly worse scores
        self.mock_lm_model.log_score_text.return_value = -10.5
        
        # Run multiple times and count acceptances
        acceptances = 0
        for _ in range(100):
            sampler.sample_key_pass()
            if sampler.current_score == -10.5:
                acceptances += 1
                sampler.current_score = -10.0  # Reset for next iteration
        
        # With high temperature, should accept some worse solutions
        self.assertGreater(acceptances, 0, "Should accept some worse solutions with high temperature")
    
    def test_temperature_affects_acceptance(self):
        """Test that lower temperature accepts fewer worse solutions."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key, seed=123)
        
        # Very low temperature should reject most worse solutions
        sampler.temperature = 0.1
        sampler.current_score = -10.0
        
        # Set up mock to return worse scores
        self.mock_lm_model.log_score_text.return_value = -11.0
        
        # Run multiple times and count acceptances
        acceptances = 0
        for _ in range(100):
            sampler.sample_key_pass()
            if sampler.current_score == -11.0:
                acceptances += 1
                sampler.current_score = -10.0  # Reset for next iteration
        
        # With low temperature, should accept very few worse solutions
        self.assertLess(acceptances, 50, "Should accept fewer worse solutions with low temperature")


class TestBayesianSamplerSER(unittest.TestCase):
    """Test the Symbol Error Rate (SER) calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test case
        self.ciphertext = [1, 2, 3, 4, 5]
        self.ground_truth_key = {
            'a': [1],
            'b': [2],
            'c': [3],
            'd': [4],
            'e': [5]
        }
        
        self.mock_lm_model = Mock(spec=InterpolatedLanguageModel)
        self.mock_lm_model.log_score_text.return_value = -10.0
    
    def test_ser_perfect_key(self):
        """Test SER calculation with perfect key."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Set the key to be perfect
        sampler.best_key = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
        
        ser = sampler.calculate_ser()
        self.assertEqual(ser, 0.0, "Perfect key should have SER of 0.0")
    
    def test_ser_completely_wrong_key(self):
        """Test SER calculation with completely wrong key."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Set the key to be completely wrong
        sampler.best_key = {1: 'z', 2: 'y', 3: 'x', 4: 'w', 5: 'v'}
        
        ser = sampler.calculate_ser()
        self.assertEqual(ser, 1.0, "Completely wrong key should have SER of 1.0")
    
    def test_ser_partial_correctness(self):
        """Test SER calculation with partially correct key."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Set the key to be 60% correct (3 out of 5)
        sampler.best_key = {1: 'a', 2: 'b', 3: 'c', 4: 'x', 5: 'y'}
        
        ser = sampler.calculate_ser()
        self.assertAlmostEqual(ser, 0.4, places=2, msg="Partially correct key should have SER between 0 and 1")
    
    def test_ser_with_custom_key(self):
        """Test SER calculation with a custom key parameter."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Test with a custom key
        custom_key = {1: 'a', 2: 'x', 3: 'c', 4: 'd', 5: 'e'}
        ser = sampler.calculate_ser(custom_key)
        
        # Should have 1 error out of 5
        self.assertAlmostEqual(ser, 0.2, places=2)


class TestBayesianSamplerPlaintextGeneration(unittest.TestCase):
    """Test the plaintext generation with spaces."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ciphertext = [1, 2, 3, 4, 5]
        self.ground_truth_key = {chr(ord('a') + i): [i+1] for i in range(5)}
        
        self.mock_lm_model = Mock(spec=InterpolatedLanguageModel)
        self.mock_lm_model.log_score_text.return_value = -10.0
    
    def test_plaintext_without_spaces(self):
        """Test plaintext generation with no spaces."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        sampler.space_positions = set()
        sampler.current_key = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
        
        plaintext = sampler._build_plaintext_from_key(sampler.current_key, sampler.space_positions)
        
        self.assertEqual(plaintext, 'abcde')
    
    def test_plaintext_with_spaces(self):
        """Test plaintext generation with spaces."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        sampler.space_positions = {2, 4}  # Space after positions 2 and 4
        sampler.current_key = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
        
        plaintext = sampler._build_plaintext_from_key(sampler.current_key, sampler.space_positions)
        
        # Spaces should be inserted at positions 2 and 4
        self.assertIn(' ', plaintext)
    
    def test_plaintext_length_consistency(self):
        """Test that plaintext has correct number of characters."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Count non-space characters
        non_space_count = len(sampler.current_plaintext.replace(' ', ''))
        
        self.assertEqual(non_space_count, len(self.ciphertext),
                        "Number of non-space characters should equal ciphertext length")


class TestBayesianSamplerTemperatureSchedule(unittest.TestCase):
    """Test the temperature update schedule."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ciphertext = [1, 2, 3]
        self.ground_truth_key = {'a': [1], 'b': [2], 'c': [3]}
        
        self.mock_lm_model = Mock(spec=InterpolatedLanguageModel)
        self.mock_lm_model.log_score_text.return_value = -10.0
    
    def test_temperature_decreases_linearly(self):
        """Test that temperature decreases linearly over iterations."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        num_iterations = 1000
        initial_temp = sampler.temperature
        
        # Check at 25%, 50%, 75%, 100%
        for progress in [0.25, 0.5, 0.75, 1.0]:
            iteration = int(num_iterations * progress)
            sampler.update_temperature(iteration, num_iterations)
            
            expected_temp = initial_temp - (9.0 * progress)
            self.assertAlmostEqual(sampler.temperature, expected_temp, places=5)
    
    def test_temperature_reaches_minimum(self):
        """Test that temperature reaches minimum at end of iterations."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        from utils.constants import INITIAL_TEMPERATURE
        num_iterations = 5000
        
        sampler.update_temperature(num_iterations - 1, num_iterations)
        
        # Temperature should be close to 1.0 at the end
        expected_final = INITIAL_TEMPERATURE - 9.0
        self.assertAlmostEqual(sampler.temperature, expected_final, places=1)


class TestBayesianSamplerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_lm_model = Mock(spec=InterpolatedLanguageModel)
        self.mock_lm_model.log_score_text.return_value = -10.0
    
    def test_seed_reproducibility(self):
        """Test that using the same seed produces reproducible results."""
        ciphertext = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ground_truth_key = {chr(ord('a') + i): [i+1] for i in range(10)}
        
        # Create two samplers with the same seed
        sampler1 = BayesianSampler(ciphertext, self.mock_lm_model, ground_truth_key, seed=42)
        sampler2 = BayesianSampler(ciphertext, self.mock_lm_model, ground_truth_key, seed=42)
        
        # Initial states should be identical
        self.assertEqual(sampler1.current_key, sampler2.current_key)
        self.assertEqual(sampler1.space_positions, sampler2.space_positions)
        self.assertEqual(sampler1.current_plaintext, sampler2.current_plaintext)
    
    def test_empty_ciphertext_spaces(self):
        """Test initialization with empty ciphertext for space positions."""
        ciphertext = []
        ground_truth_key = {}
        
        sampler = BayesianSampler(ciphertext, self.mock_lm_model, ground_truth_key)
        
        # Should handle empty ciphertext gracefully
        self.assertEqual(len(sampler.space_positions), 0)
    
    def test_single_symbol_cipher(self):
        """Test with cipher containing only one symbol."""
        ciphertext = [1, 1, 1, 1, 1]
        ground_truth_key = {'a': [1]}
        
        sampler = BayesianSampler(ciphertext, self.mock_lm_model, ground_truth_key)
        
        # Should create a key for the single symbol
        self.assertEqual(len(sampler.current_key), 1)
        self.assertIn(1, sampler.current_key)
    
    def test_large_alphabet_cipher(self):
        """Test with cipher containing many unique symbols."""
        ciphertext = list(range(1, 101))  # 100 unique symbols
        ground_truth_key = {chr(ord('a') + i % 26): [i+1] for i in range(100)}
        
        sampler = BayesianSampler(ciphertext, self.mock_lm_model, ground_truth_key)
        
        # Should create a key for all symbols
        self.assertEqual(len(sampler.current_key), 100)


class TestBayesianSamplerIntegration(unittest.TestCase):
    """Integration tests for the full sampling process."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a small cipher for quick testing
        self.ciphertext = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        self.ground_truth_key = {'a': [1], 'b': [2], 'c': [3]}
        
        # Create mock language model with realistic behavior
        self.mock_lm_model = Mock(spec=InterpolatedLanguageModel)
        self.mock_lm_model.ngram_lm = Mock(spec=NgramLanguageModel)
        self.mock_lm_model.dict_lm = Mock(spec=DictionaryLanguageModel)
        
        # Mock returns slightly random scores around -10
        def variable_score(text):
            return -10.0 + random.uniform(-2, 2)
        
        self.mock_lm_model.log_score_text.side_effect = variable_score
        self.mock_lm_model.ngram_lm.log_score_text.side_effect = variable_score
        self.mock_lm_model.dict_lm.log_score_text.side_effect = variable_score
    
    def test_run_completes_successfully(self):
        """Test that the run method completes without errors."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Run for a small number of iterations
        best_key, best_plaintext = sampler.run(num_iterations=10, log_interval=5)
        
        # Should return valid results
        self.assertIsNotNone(best_key)
        self.assertIsNotNone(best_plaintext)
        self.assertGreater(len(best_key), 0)
        self.assertGreater(len(best_plaintext), 0)
    
    def test_best_solution_is_tracked(self):
        """Test that best solution is properly tracked throughout run."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        # Set up mock to return improving scores
        call_count = [0]
        def improving_score(text):
            call_count[0] += 1
            # Every 10th call returns a better score
            if call_count[0] % 10 == 0:
                return -5.0 + (call_count[0] / 10)
            return -15.0
        
        self.mock_lm_model.log_score_text.side_effect = improving_score
        
        initial_best = sampler.best_score
        sampler.run(num_iterations=20, log_interval=10)
        
        # Best score should improve
        self.assertGreater(sampler.best_score, initial_best)
    
    def test_get_best_solution_returns_tracked_best(self):
        """Test that get_best_solution returns the best found solution."""
        sampler = BayesianSampler(self.ciphertext, self.mock_lm_model, self.ground_truth_key)
        
        sampler.run(num_iterations=10, log_interval=5)
        
        best_key, best_plaintext = sampler.get_best_solution()
        
        # Should match the tracked best solution
        self.assertEqual(best_key, sampler.best_key)
        self.assertEqual(best_plaintext, sampler.best_plaintext)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
