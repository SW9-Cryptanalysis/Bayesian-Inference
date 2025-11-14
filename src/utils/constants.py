import pathlib

TOTAL_ITERATIONS = 5000
INITIAL_TEMPERATURE = 10.0


PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
EXAMPLE_CIPHERS_PATH = DATA_PATH / "ciphers"
HOMOPHONIC_CIPHER_PREFIX = "cipher-"