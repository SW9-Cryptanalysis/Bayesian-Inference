import logging
import pathlib
from lm_models.n_gram_model import load_ngram_model
from lm_models.dictionary_model import DictionaryLanguageModel
from lm_models.interpolated_model import InterpolatedLanguageModel
from search.bayesian_sampler import BayesianSampler
from utils.load_cipher import load_cipher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == '__main__':
    ngram_model = load_ngram_model('models/ngram_model_n3_chunks101956_20251118_115213.pkl')
    dictionary_model = DictionaryLanguageModel(r'data\wordlists\en_10k.txt')
    interpolated_model = InterpolatedLanguageModel(ngram_model, dictionary_model)
    
    # Load cipher data including ground truth key
    cipher, ground_truth_key = load_cipher(pathlib.Path('c_400_30.json'))
    
    searcher = BayesianSampler(cipher, interpolated_model, ground_truth_key)
    
    searcher.run()