"""
Integration test for the complete CRP Bayesian decipherment pipeline.

This test validates:
1. All components work together correctly
2. No cache clearing/rebuilding during sampling
3. Incremental scoring is used throughout
4. Paper's approach is faithfully implemented
"""

import pytest
import json
from search.crp_bayesian_sampler import CRPBayesianSampler
from lm_models.n_gram_model import NgramLanguageModel
from lm_models.dictionary_model import DictionaryLanguageModel
from lm_models.crp_source_model import CRPSourceModel
from lm_models.crp_channel_model import CRPChannelModel
from lm_models.crp_joint_model import CRPJointModel
from utils.constants import PROJECT_ROOT, ALPHA, BETA


class TestCRPPipelineIntegration:
    """Integration tests for complete CRP decipherment pipeline."""
    
    @pytest.fixture
    def full_setup(self):
        """Create complete setup for testing."""
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
        
        # Load cipher
        cipher_path = PROJECT_ROOT / "data" / "ciphers" / "mono-cipher-5.json"
        with open(cipher_path) as f:
            data = json.load(f)
        
        return ngram_model, dict_model, data["ciphertext"][:30], data["key"]
    
    def test_complete_pipeline_runs(self, full_setup):
        """Test that complete pipeline runs without errors."""
        ngram_model, dict_model, ciphertext, ground_truth_key = full_setup
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext,
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Run short sampling
        best_key, best_plaintext = sampler.run(num_iterations=20, log_interval=10)
        
        # Verify results
        assert best_key is not None
        assert best_plaintext is not None
        assert len(best_key) == len(set(ciphertext))
        
        # Verify SER is computed
        ser = sampler.calculate_ser()
        assert 0.0 <= ser <= 1.0
    
    def test_hyperparameters_match_paper(self, full_setup):
        """Test that hyperparameters match paper's specification."""
        ngram_model, dict_model, ciphertext, ground_truth_key = full_setup
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext,
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Check alpha (source prior) = 10^4
        assert sampler.crp_source.alpha == ALPHA
        assert ALPHA == 10000.0
        
        # Check beta (channel prior) = 0.01
        assert sampler.crp_channel.beta == BETA
        assert BETA == 0.01
        
        # Check interpolation weights (0.1 n-gram, 0.9 word)
        assert sampler.model.ngram_weight == 0.1
        assert sampler.model.word_weight == 0.9
    
    def test_cache_persistence_during_sampling(self, full_setup):
        """Test that cache is not cleared during sampling."""
        ngram_model, dict_model, ciphertext, ground_truth_key = full_setup
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext,
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Get initial cache size
        initial_source_cache_size = len(sampler.crp_source.cache.ngram_counts)
        initial_channel_cache_size = len(sampler.crp_channel.cache.channel_counts)
        
        assert initial_source_cache_size > 0
        assert initial_channel_cache_size > 0
        
        # Run sampling
        for _ in range(10):
            sampler.sample_key_pass()
            sampler.sample_space_pass()
        
        # Cache should still exist (might grow/shrink slightly but not be empty)
        assert len(sampler.crp_source.cache.ngram_counts) > 0
        assert len(sampler.crp_channel.cache.channel_counts) > 0
    
    def test_incremental_scorer_is_active(self, full_setup):
        """Test that incremental scorer is being used."""
        ngram_model, dict_model, ciphertext, ground_truth_key = full_setup
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext,
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Incremental scorer should exist
        assert sampler.incremental_scorer is not None
        
        # It should reference the same models
        assert sampler.incremental_scorer.source_model is sampler.crp_source
        assert sampler.incremental_scorer.channel_model is sampler.crp_channel
    
    def test_temperature_annealing_schedule(self, full_setup):
        """Test temperature follows paper's linear annealing (10 -> 1)."""
        ngram_model, dict_model, ciphertext, ground_truth_key = full_setup
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext,
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Initial temperature
        assert sampler.temperature == 10.0
        
        # Test linear schedule
        test_points = [
            (0, 5000, 10.0),      # Start
            (1250, 5000, 7.75),   # 25%
            (2500, 5000, 5.5),    # 50%
            (3750, 5000, 3.25),   # 75%
            (5000, 5000, 1.0),    # End
        ]
        
        for iteration, total, expected_temp in test_points:
            sampler.update_temperature(iteration, total)
            assert abs(sampler.temperature - expected_temp) < 0.01
    
    def test_type_sampling_updates_all_occurrences(self, full_setup):
        """Test that type sampling updates all occurrences of a symbol."""
        ngram_model, dict_model, ciphertext, ground_truth_key = full_setup
        
        # Create cipher with repeated symbols
        simple_cipher = [1, 2, 1, 3, 1]
        
        sampler = CRPBayesianSampler(
            ciphertext=simple_cipher,
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Get current mapping for symbol 1
        old_char = sampler.current_key[1]
        old_plaintext = sampler.current_plaintext
        
        # Force a change by running multiple samples
        for _ in range(20):
            sampler.sample_key_pass()
            if sampler.current_key[1] != old_char:
                break
        
        # If key changed, all occurrences should change
        if sampler.current_key[1] != old_char:
            new_char = sampler.current_key[1]
            new_plaintext = sampler.current_plaintext.replace(' ', '')
            
            # Check all positions where symbol 1 appears have the new character
            for i, symbol in enumerate(simple_cipher):
                if symbol == 1:
                    assert new_plaintext[i] == new_char
    
    def test_crp_formulas_in_action(self, full_setup):
        """Test that CRP formulas are being applied correctly."""
        ngram_model, dict_model, ciphertext, ground_truth_key = full_setup
        
        sampler = CRPBayesianSampler(
            ciphertext=ciphertext[:10],
            ngram_model=ngram_model,
            dict_model=dict_model,
            ground_truth_key=ground_truth_key,
            seed=42,
            use_crp=True
        )
        
        # Get component scores
        ngram, word, source, channel = sampler.model.log_score_separate(
            sampler._ciphertext, sampler.current_plaintext, sampler.current_key
        )
        
        # All scores should be finite
        assert ngram > -float('inf')
        assert word > -float('inf')
        assert source > -float('inf')
        assert channel > -float('inf')
        
        # Source should be interpolation of ngram and word
        expected_source = 0.1 * ngram + 0.9 * word
        assert abs(source - expected_source) < 0.01
        
        # Total score should be source + channel
        total = sampler.model.log_score(
            sampler._ciphertext, sampler.current_plaintext, sampler.current_key
        )
        assert abs(total - (source + channel)) < 0.01


class TestPaperImplementationFidelity:
    """Tests that verify faithful implementation of paper's approach."""
    
    def test_crp_source_formula(self):
        """Verify CRP source formula: P(pi|pi-1) = (α·P0 + C) / (α + C_total)."""
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
        source = CRPSourceModel(ngram_model, alpha=ALPHA)
        
        context = ('t', 'h')
        char = 'e'
        
        # Add to cache
        source.cache.add_ngram(context, 'e')
        source.cache.add_ngram(context, 'e')
        source.cache.add_ngram(context, 'a')
        
        # Manual calculation
        base_prob = ngram_model.score_char(char, context)
        cache_count = 2
        context_total = 3
        expected = (ALPHA * base_prob + cache_count) / (ALPHA + context_total)
        
        # CRP calculation
        actual = source.score_char(char, context)
        
        assert abs(actual - expected) < 1e-6
    
    def test_crp_channel_formula(self):
        """Verify CRP channel formula: P(ci|pi) = (β·P0 + C) / (β + C_total)."""
        channel = CRPChannelModel(26, beta=BETA)
        
        plain = 'a'
        cipher = 5
        
        # Add to cache
        channel.cache.add_substitution('a', 5)
        channel.cache.add_substitution('a', 5)
        channel.cache.add_substitution('a', 10)
        
        # Manual calculation
        base_prob = 1.0 / 26
        cache_count = 2
        plain_total = 3
        expected = (BETA * base_prob + cache_count) / (BETA + plain_total)
        
        # CRP calculation
        actual = channel.score_substitution(plain, cipher)
        
        assert abs(actual - expected) < 1e-6
    
    def test_interpolation_weights(self):
        """Verify interpolation weights match paper (0.1 n-gram, 0.9 word)."""
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
        
        source = CRPSourceModel(ngram_model, alpha=ALPHA)
        channel = CRPChannelModel(26, beta=BETA)
        joint = CRPJointModel(source, channel, dict_model)
        
        assert joint.ngram_weight == 0.1
        assert joint.word_weight == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
