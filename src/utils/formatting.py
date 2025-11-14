from unidecode import unidecode
import re
from num2words import num2words
from parameter_validator import parameter_validator, strongly_typed
from gutenberg_cleaner import simple_cleaner


def numbers_to_words(text: str) -> str:
	"""Convert all numbers in the input text to their word representations.

	Args:
			text (str): The input text containing numbers.

	Returns:
		str: The text with numbers converted to words.

	"""

	def replace_number(match: re.Match) -> str:
		number_str = match.group()
		number_float = float(number_str)
		return num2words(number_float)

	return re.sub(r"\d+(\.\d+)?+", replace_number, text)


@parameter_validator(text=strongly_typed)
def format_text(text: str) -> str:
	"""Format text by filtering to alphabetic characters and spaces, converting to lowercase.

	Args:
			text (str): A slice of text from a book

	Returns:
			str: The text slice converted to lowercase
				keeping only alphabetic characters and spaces.

	"""
	text = simple_cleaner(text)
	text = numbers_to_words(text)
	text = numbers_to_words(text)
	text = unidecode(text.lower())
	text = re.sub(r"[A-Z]", "", text)
	# Keep only lowercase letters a-z and spaces
	return "".join([c for c in text if c.isalpha() or c == ' '])
