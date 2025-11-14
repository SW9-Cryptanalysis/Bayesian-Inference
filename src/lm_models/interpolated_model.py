import logging
from lm_models.n_gram_model import NgramLanguageModel
from lm_models.dictionary_model import DictionaryLanguageModel

logger = logging.getLogger(__name__)

class InterpolatedLanguageModel:
    """
    Combines the NgramLanguageModel and DictionaryLanguageModel
    using log-linear interpolation, as described in the paper.
    
    This model calculates the final P(p) probability for a
    plaintext hypothesis `p`.
    """
    
    def __init__(self, ngram_lm: NgramLanguageModel, dict_lm: DictionaryLanguageModel):
        self.ngram_lm = ngram_lm
        self.dict_lm = dict_lm
        
        # Set interpolation weights based on the paper
        self.ngram_lm_weight = 0.1
        self.word_lm_weight = 0.9
        
    def log_score_text(self, text: str) -> float:
        """
        Calculates the final interpolated log-score for the plaintext.
        
        This is the main scoring function your MCMC sampler will use
        to evaluate how "good" a plaintext hypothesis is.
        
        Args:
            text: The plaintext hypothesis (e.g., "I LIKE KILLING PEOPLE")
            
        Returns:
            float: The final, combined log-score. Higher is better.
        """
        
        ngram_log_score = self.ngram_lm.log_score_text(text)
        dict_log_score = self.dict_lm.log_score_text(text)
        
        # Calculate the weighted sum in log-space
        final_score = (self.ngram_lm_weight * ngram_log_score) + \
                      (self.word_lm_weight * dict_log_score)
                      
        return final_score

    def test_models(self, text: str) -> None:
        """
        Utility function to log individual model scores for debugging.
        
        Args:
            text: The plaintext hypothesis to score.
        
        Returns:
            None
        """
        ngram_log_score = self.ngram_lm.log_score_text(text)
        dict_log_score = self.dict_lm.log_score_text(text)
        logger.info("\n")
        logger.info("--- Model Scores ---")
        logger.info(f'"{text}"')
        
        logger.info(f"N-gram LM log-score: {ngram_log_score}")
        logger.info(f"Dictionary LM log-score: {dict_log_score}")
        logger.info(f"Interpolated final log-score: {self.log_score_text(text)}")