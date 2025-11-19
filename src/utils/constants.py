import pathlib

TOTAL_ITERATIONS = 5000
INITIAL_TEMPERATURE = 10.0

# CRP Dirichlet priors (from paper)
ALPHA = 10000.0  # Source model prior (high value favors base LM)
BETA = 0.01      # Channel model prior (low value favors sparse/deterministic substitution)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
EXAMPLE_CIPHERS_PATH = DATA_PATH / "ciphers"
HOMOPHONIC_CIPHER_PREFIX = "cipher-"
