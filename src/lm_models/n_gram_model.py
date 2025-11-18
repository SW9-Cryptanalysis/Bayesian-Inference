import math
import pickle
import os
import logging
from datetime import datetime
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from text_fetching.fetcher import Fetcher

logger = logging.getLogger(__name__)


class NgramLanguageModel:
    """Character-level n-gram language model for scoring plaintext hypotheses.
    
    This model assigns probabilities P(p) to plaintext sequences based on
    character n-gram statistics learned from English text.
    """
    
    def __init__(self, model, n):
        """Initialize the language model wrapper.
        
        Args:
            model: Trained NLTK language model (e.g., Laplace)
            n: The n-gram order (e.g., 3 for trigrams)
        """
        self.model = model
        self.n = n
    
    def score_text(self, text: str) -> float:
        """Calculate the probability P(p) for a plaintext sequence.
        
        This is the score used for interpolation with dictionary models.
        Returns a probability between 0 and 1.
        
        Args:
            text: The plaintext string to score
            
        Returns:
            float: Probability P(p) of the text (0 to 1)
        """
        log_prob = self.model.logscore(list(text))
        # Convert log probability to actual probability
        # Note: This can be very small for long texts
        probability = math.exp(log_prob)
        return probability
    
    def log_score_text(self, text: str) -> float:
        """
        Calculates the TOTAL log probability for a plaintext sequence.
        """
        # NLTK's model.logscore already calculates the total log prob
        # of the sequence, handling padding and context correctly.
        try:
            # We need to tokenize into a list of characters,
            # just as the model was trained.
            return self.model.logscore(list(text))
        except Exception as e:
            logger.warning(f"Error scoring text: {e}. Returning -inf.")
            return -float('inf')
    
    def perplexity_text(self, text: str) -> float:
        """Calculate perplexity for a plaintext sequence.
        
        Lower perplexity indicates more probable/natural text.
        
        Args:
            text: The plaintext string to score
            
        Returns:
            float: Perplexity (lower is better)
        """
        return self.model.perplexity(list(text))
    
    def score_char(self, char: str, context: tuple) -> float:
        """Get probability of a character given its context.
        
        Args:
            char: The character to score
            context: Tuple of previous characters (length = n-1)
            
        Returns:
            float: Probability P(char | context)
        """
        return self.model.score(char, context)


def train_ngram_model(n: int = 3) -> NgramLanguageModel:
    tokenized_data = []
    fetcher = Fetcher()
    
    # --- 1. Fetch Gutenberg Book Data ---
    logger.info("Fetching books from Project Gutenberg...")
    for book_text in fetcher.fetch_book_text():
        logger.info(f"Book ID: {fetcher.book_id}")
        logger.info(f"Book length: {len(book_text)} characters")
        logger.debug(f"First 100 characters: {book_text[:100]}")
        
        chunk_size = 500  # Characters per chunk
        
        for i in range(0, len(book_text), chunk_size):
            chunk = book_text[i:i+chunk_size].strip()
            if len(chunk) > 50:
                # Character-level tokenization: convert each chunk to a list of characters
                tokenized_data.append(list(chunk))
        
        logger.info(f"Total chunks so far: {len(tokenized_data)}")
        
        if len(tokenized_data) >= 100_000: # 100.000 chunks = 50 million characters
            break

    logger.info(f"Number of chunks: {len(tokenized_data)}")
    if tokenized_data:
        logger.debug(f"Example tokenized chunk (first 50 chars): {tokenized_data[0][:50]}")
        logger.info(f"Average chunk length: {sum(len(chunk) for chunk in tokenized_data) / len(tokenized_data):.0f} characters")

    # --- 2. Preprocessing & Training ---
    train_data, vocab = padded_everygram_pipeline(n, tokenized_data)

    # Initialize the model
    # Laplace smoothing adds one count to each n-gram to prevent zero probabilities.
    model = Laplace(n)

    # Train the model
    logger.info(f"Training {n}-gram model...")
    model.fit(train_data, vocab)

    # --- 3. Using the Trained Model ---
    logger.info(f"\n--- {n}-gram LM Trained ---")
    logger.info(f"Vocabulary size: {len(model.vocab)}")  # 27 chars (a-z + space) + 3 special tokens (<s>, </s>, <UNK>)
    
    lm = NgramLanguageModel(model, n)
    
    test_ngram_model(lm)
    
    # Save the trained model
    save_model(lm, n, len(tokenized_data))
    
    return lm


def test_ngram_model(lm: NgramLanguageModel):
    # Test with common English text patterns
    plaintext_hypothesis_1 = "the quick brown fox jumps over the lazy dog"
    log_prob_1 = lm.log_score_text(plaintext_hypothesis_1)
    logger.info(f"\nScore: {log_prob_1:8.2f} | Text: '{plaintext_hypothesis_1}'")
    
    plaintext_hypothesis_2 = "and the and the and the and the and the and"
    log_prob_2 = lm.log_score_text(plaintext_hypothesis_2)
    logger.info(f"Score: {log_prob_2:8.2f} | Text: '{plaintext_hypothesis_2}'")
    
    plaintext_hypothesis_3 = "asdf eafm rafr lsdo mamdmr psd rkm gra lkt"
    log_prob_3 = lm.log_score_text(plaintext_hypothesis_3)
    logger.info(f"Score: {log_prob_3:8.2f} | Text: '{plaintext_hypothesis_3}'")
    
    plaintext_hypothesis_4 = "there are several editions of this ebook in"
    log_prob_4 = lm.log_score_text(plaintext_hypothesis_4)
    logger.info(f"Score: {log_prob_4:8.2f} | Text: '{plaintext_hypothesis_4}'")
    
    plaintext_hypothesis_5 = "This is a real sentence but it is very long"
    log_prob_5 = lm.log_score_text(plaintext_hypothesis_5)
    logger.info(f"Score: {log_prob_5:8.2f} | Text: '{plaintext_hypothesis_5}'")


def save_model(lm: NgramLanguageModel, n: int, num_chunks: int) -> str:
    """Save the trained n-gram language model to disk.
    
    Args:
        lm: The trained NgramLanguageModel to save
        n: The n-gram order (e.g., 3 for trigrams)
        num_chunks: Number of chunks used for training
        
    Returns:
        str: Path to the saved model file
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Generate filename with timestamp and metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"models/ngram_model_n{n}_chunks{num_chunks}_{timestamp}.pkl"
    
    # Save the model using pickle
    logger.info(f"\nSaving model to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(lm, f)
    
    logger.info(f"Model saved successfully!")
    logger.info(f"Model details:")
    logger.info(f"  - N-gram order: {n}")
    logger.info(f"  - Training chunks: {num_chunks:,}")
    logger.info(f"  - Vocabulary size: {len(lm.model.vocab)}")
    logger.info(f"  - File: {filename}")
    
    return filename


def load_ngram_model(filepath: str) -> NgramLanguageModel:
    """Load a saved n-gram language model from disk.
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        NgramLanguageModel: The loaded model
    """
    logger.info(f"Loading model from {filepath}...")
    with open(filepath, 'rb') as f:
        lm = pickle.load(f)
    
    logger.info(f"Model loaded successfully!")
    logger.info(f"  - N-gram order: {lm.n}")
    logger.info(f"  - Vocabulary size: {len(lm.model.vocab)}")
    
    return lm