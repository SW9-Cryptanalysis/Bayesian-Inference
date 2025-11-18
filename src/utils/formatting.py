from unidecode import unidecode
import re
from num2words import num2words
from parameter_validator import parameter_validator, strongly_typed


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
    """
    Cleans and formats raw text from Project Gutenberg.
    
    1. Removes Gutenberg headers/footers.
    2. Converts numbers to words.
    3. Normalizes to ASCII and converts to lowercase.
    4. Filters to keep *only* a-z and spaces.
    5. Collapses multiple spaces into one.
    """
    
    # 1. Manually remove Gutenberg headers/footers
    #    This is more reliable than an external library.
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    
    try:
        start_index = text.index(start_marker)
        start_index = text.index('\n', start_index) # Find the newline after the marker
    except ValueError:
        start_index = 0 # If marker not found, start from the beginning

    try:
        end_index = text.index(end_marker)
    except ValueError:
        end_index = len(text) # If marker not found, go to the end
        
    text = text[start_index:end_index]

    # 2. Convert numbers to words (e.g., "101" -> "one hundred one")
    text = numbers_to_words(text)
    
    # 3. Normalize to ASCII (e.g., "naïve" -> "naive") and lowercase
    text = unidecode(text.lower())
    
    # 4. Filter to keep ONLY lowercase letters and spaces
    #    This one line replaces both of your broken filter lines.
    text = re.sub(r"[^a-z ]", "", text)
    
    # 5. Collapse multiple spaces into a single space
    text = re.sub(r" +", " ", text).strip()
    
    return text
