import json
import pathlib
import os
from utils.constants import EXAMPLE_CIPHERS_PATH

def load_cipher(filepath: pathlib.Path) -> tuple[list[int], dict[str, list[int]]]:
    """Read a cipher from a JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns:
        Tuple of (ciphertext, ground_truth_key) where:
        - ciphertext is a list of integers
        - ground_truth_key maps plaintext letters to lists of cipher symbols

    """
    cipher_path = os.path.join(EXAMPLE_CIPHERS_PATH, filepath)
    with open(cipher_path) as f:
        data = json.load(f)
    ciphertext_str = data["ciphertext"]
    ciphertext = [int(num) for num in ciphertext_str.split()]
    ground_truth_key = data["key"]
    return ciphertext, ground_truth_key
