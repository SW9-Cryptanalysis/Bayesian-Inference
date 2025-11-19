"""
Tests for CRP source and channel models.

Verifies:
- CRP formula correctness
- Cache behavior
- Incremental scoring
- Exchangeability property
"""

import pytest
import math
from lm_models.crp_source_model import CRPSourceModel
from lm_models.crp_channel_model import CRPChannelModel
from lm_models.n_gram_model import NgramLanguageModel
from utils.constants import PROJECT_ROOT, ALPHA, BETA


class TestCRPSourceModel:
    """Test CRP source model implementation."""
    
    @pytest.fixture
    def ngram_model(self):
        """Create a simple n-gram model for testing."""
        # Load the pre-trained model from pickle
        model_path = PROJECT_ROOT / "models" / "trigram.pkl"
        if model_path.exists():
            import pickle
            with open(model_path, 'rb') as f:
                trained_model = pickle.load(f)
            return NgramLanguageModel(trained_model, n=3)
        else:
            # Create a simple mock model for testing if pickle doesn't exist
            from unittest.mock import Mock
            mock_model = Mock()
            mock_model.score = Mock(return_value=0.001)  # Small probability
            return NgramLanguageModel(mock_model, n=3)
    
    @pytest.fixture
    def source_model(self, ngram_model):
        """Create CRP source model."""
        return CRPSourceModel(ngram_model, alpha=ALPHA)
    
    def test_initialization(self, source_model, ngram_model):
        """Test model initializes correctly."""
        assert source_model.alpha == ALPHA
        assert source_model.n == ngram_model.n
        assert source_model.base_model == ngram_model
        assert len(source_model.cache.ngram_counts) == 0
    
    def test_crp_formula_with_empty_cache(self, source_model):
        """Test CRP formula reduces to base distribution when cache is empty."""
        context = ('<s>', '<s>')
        char = 't'
        
        # With empty cache, should be: (α * P0 + 0) / (α + 0) = P0
        base_prob = source_model.base_model.score_char(char, context)
        crp_prob = source_model.score_char(char, context)
        
        assert abs(crp_prob - base_prob) < 1e-6
    
    def test_crp_formula_with_cache(self, source_model):
        """Test CRP formula incorporates cache counts correctly."""
        context = ('t', 'h')
        char = 'e'
        
        # Get base probability
        base_prob = source_model.base_model.score_char(char, context)
        
        # Add to cache multiple times
        source_model.cache.add_ngram(context, char)
        source_model.cache.add_ngram(context, char)
        source_model.cache.add_ngram(context, 'a')  # Different char for same context
        
        # CRP formula: (α * P0 + 2) / (α + 3)
        cache_count = 2
        context_total = 3
        expected_prob = (ALPHA * base_prob + cache_count) / (ALPHA + context_total)
        
        crp_prob = source_model.score_char(char, context)
        
        assert abs(crp_prob - expected_prob) < 1e-6
    
    def test_cache_builds_incrementally(self, source_model):
        """Test cache is built incrementally during text scoring."""
        text = "the cat"
        
        # Score text (builds cache)
        source_model.log_score_text(text)
        
        # Cache should contain n-grams from the text
        assert len(source_model.cache.ngram_counts) > 0
        
        # Check specific n-grams are present
        # For "the cat" with padding, we expect n-grams like ('t', 'h') -> 'e'
        context = ('t', 'h')
        assert source_model.cache.get_count(context, 'e') > 0
    
    def test_cache_clear(self, source_model):
        """Test cache clearing works correctly."""
        source_model.log_score_text("hello world")
        assert len(source_model.cache.ngram_counts) > 0
        
        source_model.clear_cache()
        assert len(source_model.cache.ngram_counts) == 0
    
    def test_score_consistency(self, source_model):
        """Test scoring is consistent with same input."""
        text = "testing consistency"
        
        score1 = source_model.log_score_text(text)
        score2 = source_model.log_score_text(text)
        
        # Should get same score (cache is rebuilt each time in log_score_text)
        assert abs(score1 - score2) < 1e-6


class TestCRPChannelModel:
    """Test CRP channel model implementation."""
    
    @pytest.fixture
    def channel_model(self):
        """Create CRP channel model."""
        num_symbols = 26
        return CRPChannelModel(num_symbols, beta=BETA)
    
    def test_initialization(self, channel_model):
        """Test model initializes correctly."""
        assert channel_model.beta == BETA
        assert channel_model.num_cipher_symbols == 26
        assert channel_model.base_prob == 1.0 / 26
        assert len(channel_model.cache.channel_counts) == 0
    
    def test_uniform_base_distribution(self, channel_model):
        """Test base distribution is uniform."""
        # All cipher symbols should have equal base probability
        expected_prob = 1.0 / 26
        assert abs(channel_model.base_prob - expected_prob) < 1e-6
    
    def test_crp_formula_with_empty_cache(self, channel_model):
        """Test CRP formula with empty cache."""
        plaintext_char = 'a'
        cipher_symbol = 5
        
        # With empty cache: (β * P0 + 0) / (β + 0) = P0
        prob = channel_model.score_substitution(plaintext_char, cipher_symbol)
        expected_prob = channel_model.base_prob
        
        assert abs(prob - expected_prob) < 1e-6
    
    def test_crp_formula_with_cache(self, channel_model):
        """Test CRP formula incorporates cache correctly."""
        plaintext_char = 'a'
        cipher_symbol = 5
        
        # Add to cache
        channel_model.cache.add_substitution('a', 5)
        channel_model.cache.add_substitution('a', 5)
        channel_model.cache.add_substitution('a', 10)  # Different symbol for 'a'
        
        # CRP formula: (β * P0 + 2) / (β + 3)
        cache_count = 2
        plaintext_total = 3
        expected_prob = (BETA * channel_model.base_prob + cache_count) / (BETA + plaintext_total)
        
        prob = channel_model.score_substitution(plaintext_char, cipher_symbol)
        
        assert abs(prob - expected_prob) < 1e-6
    
    def test_deterministic_substitution_behavior(self, channel_model):
        """Test that model favors deterministic substitution with low beta."""
        plaintext_char = 'a'
        cipher_symbol = 5
        
        # Add many observations of same substitution
        for _ in range(100):
            channel_model.cache.add_substitution('a', 5)
        
        # This substitution should have high probability
        prob_consistent = channel_model.score_substitution('a', 5)
        
        # Different substitution should have low probability
        prob_inconsistent = channel_model.score_substitution('a', 10)
        
        # With low beta, consistent substitution is strongly favored
        assert prob_consistent > prob_inconsistent * 10
    
    def test_cache_builds_from_pairs(self, channel_model):
        """Test cache building from cipher-plaintext pairs."""
        ciphertext = [1, 2, 3, 1, 2]
        plaintext = "hello"
        
        channel_model.build_cache_from_pairs(ciphertext, plaintext)
        
        # Check cache has correct counts
        assert channel_model.cache.get_count('h', 1) == 1
        assert channel_model.cache.get_count('e', 2) == 1
        assert channel_model.cache.get_count('l', 3) == 1
        assert channel_model.cache.get_count('l', 1) == 1  # Second 'l' maps to 1
        assert channel_model.cache.get_count('o', 2) == 1


class TestCRPFormulas:
    """Test the mathematical correctness of CRP formulas."""
    
    def test_alpha_effect_on_source(self):
        """Test that high alpha favors base distribution."""
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
        
        # High alpha (favors base)
        high_alpha_model = CRPSourceModel(ngram_model, alpha=10000)
        
        # Low alpha (favors cache)
        low_alpha_model = CRPSourceModel(ngram_model, alpha=1)
        
        # Add same observations to both caches
        context = ('t', 'h')
        for _ in range(10):
            high_alpha_model.cache.add_ngram(context, 'e')
            low_alpha_model.cache.add_ngram(context, 'e')
        
        # Get probabilities
        high_alpha_prob = high_alpha_model.score_char('e', context)
        low_alpha_prob = low_alpha_model.score_char('e', context)
        base_prob = ngram_model.score_char('e', context)
        
        # High alpha should be closer to base distribution
        assert abs(high_alpha_prob - base_prob) < abs(low_alpha_prob - base_prob)
    
    def test_beta_effect_on_channel(self):
        """Test that low beta favors sparse/deterministic substitution."""
        num_symbols = 26
        
        # Low beta (favors sparsity)
        low_beta_model = CRPChannelModel(num_symbols, beta=0.01)
        
        # High beta (favors base)
        high_beta_model = CRPChannelModel(num_symbols, beta=100)
        
        # Add same observations
        for _ in range(10):
            low_beta_model.cache.add_substitution('a', 5)
            high_beta_model.cache.add_substitution('a', 5)
        
        # Score the observed substitution
        low_beta_prob = low_beta_model.score_substitution('a', 5)
        high_beta_prob = high_beta_model.score_substitution('a', 5)
        
        # Low beta should give higher probability to observed substitution
        assert low_beta_prob > high_beta_prob


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
