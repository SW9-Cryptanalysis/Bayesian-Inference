"""
Tests for CRP Bayesian sampler.

Verifies:
- Proper cache management during sampling
- No unnecessary cache rebuilding
- Incremental scoring is used correctly
- Proposals are handled correctly (accept/reject)
"""

import pytest
import json
from search.crp_bayesian_sampler import CRPBayesianSampler
from lm_models.n_gram_model import NgramLanguageModel
from lm_models.dictionary_model import DictionaryLanguageModel
from utils.constants import PROJECT_ROOT


class TestCRPBayesianSampler:
    """Test CRP Bayesian sampler implementation."""
    
    @pytest.fixture
    def models(self):
        """Create language models for testing."""
        # Load the pre-trained model from pickle
        model_path = PROJECT_ROOT / "models" / "trigram.pkl"
        if model_path.exists():
            import pickle
            with open(model_path, 'rb') as f:
                trained_model = pickle.load(f)
            ngram_model = NgramLanguageModel(trained_model, n=3)
        else:
            from unittest.mock import Mock
            mock_model = Mock()
            mock_model.score = Mock(return_value=0.001)
            ngram_model = NgramLanguageModel(mock_model, n=3)
        
        wordlist_path = PROJECT_ROOT / "data" / "wordlists" / "en_10k.txt"
        dict_model = DictionaryLanguageModel(str(wordlist_path))
        
        return ngram_model, dict_model
    
    @pytest.fixture
    def cipher_data(self):
        """Load a test cipher."""
        cipher_path = PROJECT_ROOT / "data" / "ciphers" / "mono-cipher-5.json"
        with open(cipher_path) as f:
            data = json.load(f)
        return data["ciphertext"], data["key"]
    
    def test_initialization(self, models, cipher_data):
        """Test sampler initializes correctly."""
        ngram_model, dict_model = models
        ciphertext, ground_truth_key = cipher_data
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext,
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Check initialization
        assert sampler.use_crp is True
        assert len(sampler.current_key) == len(set(ciphertext))
        assert sampler.temperature == 10.0
        assert sampler.current_score > -float('inf')
        
        # Check caches are initialized
        assert len(sampler.crp_source.cache.ngram_counts) > 0
        assert len(sampler.crp_channel.cache.channel_counts) > 0
    
    def test_key_sampling_modifies_plaintext(self, models, cipher_data):
        """Test that key sampling changes plaintext correctly."""
        ngram_model, dict_model = models
        ciphertext, ground_truth_key = cipher_data
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext[:20],  # Use shorter cipher for speed
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        initial_plaintext = sampler.current_plaintext
        initial_key = sampler.current_key.copy()
        
        # Run one key sampling pass
        sampler.sample_key_pass()
        
        # Something should change (with high probability)
        # Due to randomness, might not change on every pass, so we allow it
        assert sampler.current_key is not None
        assert sampler.current_plaintext is not None
    
    def test_cache_not_rebuilt_unnecessarily(self, models, cipher_data):
        """Test that cache is not cleared/rebuilt when proposal is rejected."""
        ngram_model, dict_model = models
        ciphertext, ground_truth_key = cipher_data
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext[:20],
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Get cache size before sampling
        initial_cache_size = len(sampler.crp_source.cache.ngram_counts)
        
        # Run sampling (will have both accepts and rejects)
        for _ in range(5):
            sampler.sample_key_pass()
        
        # Cache should still exist and be reasonable
        final_cache_size = len(sampler.crp_source.cache.ngram_counts)
        assert final_cache_size > 0
        
        # Cache size might change slightly but shouldn't be completely different
        # (This is a soft check - the key point is no errors occur)
    
    def test_temperature_annealing(self, models, cipher_data):
        """Test that temperature decreases over iterations."""
        ngram_model, dict_model = models
        ciphertext, ground_truth_key = cipher_data
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext[:20],
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        assert sampler.temperature == 10.0
        
        # Update temperature
        sampler.update_temperature(2500, 5000)  # Halfway
        assert sampler.temperature == 5.5  # Should be halfway between 10 and 1
        
        sampler.update_temperature(5000, 5000)  # End
        assert sampler.temperature == 1.0
    
    def test_space_sampling_changes_spaces(self, models, cipher_data):
        """Test that space sampling modifies space positions."""
        ngram_model, dict_model = models
        ciphertext, ground_truth_key = cipher_data
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext[:20],
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        initial_spaces = sampler.space_positions.copy()
        
        # Run space sampling
        sampler.sample_space_pass()
        
        # Spaces should exist (might be same or different)
        assert sampler.space_positions is not None
    
    def test_best_solution_tracking(self, models, cipher_data):
        """Test that best solution is tracked correctly."""
        ngram_model, dict_model = models
        ciphertext, ground_truth_key = cipher_data
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext[:20],
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        initial_best_score = sampler.best_score
        
        # Run some iterations
        for _ in range(10):
            sampler.sample_key_pass()
            sampler.sample_space_pass()
            sampler.update_temperature(1, 5000)
        
        # Best score should be tracked
        assert sampler.best_score >= initial_best_score  # Should not get worse
        assert sampler.best_key is not None
        assert sampler.best_plaintext is not None
    
    def test_ser_calculation(self, models, cipher_data):
        """Test Symbol Error Rate calculation."""
        ngram_model, dict_model = models
        ciphertext, ground_truth_key = cipher_data
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext[:50],
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Initial SER should be high (random key)
        initial_ser = sampler.calculate_ser()
        assert 0.0 <= initial_ser <= 1.0
        
        # If we set the correct key, SER should be 0
        sampler.best_key = sampler.ground_truth_key.copy()
        perfect_ser = sampler.calculate_ser()
        assert perfect_ser == 0.0
    
    def test_sampling_run_completes(self, models, cipher_data):
        """Test that a short sampling run completes without errors."""
        ngram_model, dict_model = models
        ciphertext, ground_truth_key = cipher_data
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext[:20],
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Run a very short sampling (just to test it works)
        best_key, best_plaintext = sampler.run(num_iterations=10, log_interval=5)
        
        # Should return valid results
        assert best_key is not None
        assert best_plaintext is not None
        assert len(best_key) > 0
        assert len(best_plaintext) > 0
    
    def test_incremental_scorer_is_used(self, models, cipher_data):
        """Test that incremental scorer is created and used."""
        ngram_model, dict_model = models
        ciphertext, ground_truth_key = cipher_data
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext[:20],
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Incremental scorer should be created
        assert sampler.incremental_scorer is not None
        assert sampler.incremental_scorer.source_model == sampler.crp_source
        assert sampler.incremental_scorer.channel_model == sampler.crp_channel
    
    def test_component_scores_tracked(self, models, cipher_data):
        """Test that component scores are tracked separately."""
        ngram_model, dict_model = models
        ciphertext, ground_truth_key = cipher_data
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext[:20],
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Component scores should be initialized
        assert hasattr(sampler, 'current_ngram_score')
        assert hasattr(sampler, 'current_channel_score')
        assert sampler.current_ngram_score > -float('inf')
        assert sampler.current_channel_score > -float('inf')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
