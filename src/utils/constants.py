import pathlib

TOTAL_ITERATIONS = 5000
INITIAL_TEMPERATURE = 10.0
PLAINTEXT_LENGTH_TO_SHOW = 100
LOG_INTERVAL = 100
PLAINTEXT_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

# Space sampling configuration
MIN_SPACE_PROPOSALS = 100   # Minimum proposals per pass
MAX_SPACE_PROPOSALS = 500  # Maximum proposals per pass
SPACE_PROPOSAL_RATIO = 0.25  # Propose 25% of cipher length

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
EXAMPLE_CIPHERS_PATH = DATA_PATH / "ciphers"
HOMOPHONIC_CIPHER_PREFIX = "cipher-"