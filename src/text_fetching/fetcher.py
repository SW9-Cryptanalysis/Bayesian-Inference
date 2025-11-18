import logging
import requests
from typing import Any
from collections.abc import Generator
from utils.formatting import format_text
from utils.files import save_book, book_is_cached, get_cached_book

logger = logging.getLogger(__name__)

GUTENDEX_BASE_URL = "https://gutendex.com/books"


class Fetcher:
	"""A class to fetch and slice random books from Project Gutenberg.

	Fetched via the Gutendex API. It supports caching fetched books locally
	to avoid redundant network requests.

	Attributes:
		BOOK_IDS (list): A predefined list of Project Gutenberg book IDs.
		book_id (str): The ID of the currently selected book.
		is_cached (bool): Indicates if the book is already cached locally.

	Methods:
	-------
		fetch_random_book_text(): Fetches the full text of a random book.
		get_random_book_slice(book_text, min_len, max_len): Extracts a random slice
			from the provided book text.

	"""

	BOOK_IDS = [
		"84",  # Frankenstein
		"2701",  # Moby Dick
		"1342",  # Pride and Prejudice
		"2641",  # A Room with a View
		"145",  # Middlemarch
		"37106",  # Little Women; Or, Meg, Jo, Beth, and Amy
		"7241",  # Fables of La Fontaine
		"67979",  # The Blue Castle
		"43",  # The Strange Case of Dr. Jekyll and Mr. Hyde
		"1260",  # Jane Eyre
		"16389",  # The Enchanted April
		"394",  # Cranford
		"6761",  # The Adventures of Ferdinand Count Fathom
		"345",  # Dracula
		"1259",  # Twenty Years After
		"2160",  # The Expedition of Humphry Clinker
		"4085",  # The Adventures of Roderick Random
		"5197",  # My Life - Volume 1
		"6593",  # History of Tom Jones, a Foundling
		"1232",  # The Prince
		"3207",  # Leviathan
		"2554",  # Crime and Punishment
		"1080",  # A Modest Proposal
		"174",  # The Picture of Dorian Gray
		"98",  # A Tale of Two Cities
		"25344",  # The Scarlet Letter
		"7370",  # Second Treatise of Government
		"2148",  # The Works of Edgar Allan Poe - Volume 2
		"76",  # The Adventures of Huckleberry Finn
		"1952",  # The Yellow Wallpaper
		"2591",  # Grimm's Fairy Tales
		"2600",  # War and Peace
		"41",  # The Legend of Sleepy Hollow
		"844",  # The Importance of Being Earnest: A Trivial Comedy for Serious People
		"46",  # A Christmas Carol in Prose; Being a Ghost Story of Christmas
		"1661",  # The Adventures of Sherlock Holmes
		"3296",  # The Confessions of St. Augustine
		"408",  # The Souls of Black Folk
		"5200",  # Metamorphosis
		"26184",  # Simple Sabotage Field Manual
		"205",  # Walden, and On The Duty Of Civil Disobedience
		"1497",  # The Republic
		"1998",  # Thus Spake Zarathustra
		"23",  # Narrative of the Life of Frederick Douglass, an American Slave
		"768",  # Wuthering Heights
		"28054",  # The Brothers Karamazov
		"2542",  # A Doll's House
		"45",  # Anne of Green Gables
		"34901",  # On Liberty
		"219",  # Heart of Darkness
		"20203",  # Autobiography of Benjamin Franklin
		"76939",  # The laws of contrast of color
		"1184",  # The Count of Monte Cristo
		"15399",  # The Interesting Narrative of the Life of Olaudah Equiano...
		"1400",  # Great Expectations
		"74",  # The Adventures of Tom Sawyer
		"36034",  # White Nights and Other Stories
		"815",  # Democracy in America
		"4300",  # Ulysses
		"1023",  # Bleak House
		"4363",  # Beyond Good and Evil
		"2852",  # The Hound of the Baskervilles
		"34450",  # The Nature of Animal Light
		"36",  # War of the Worlds
		"55",  # The Wonderful Wizard of Oz
		"3300",  # An Inquiry into the Nature and Causes of the Wealth of Nations
		"135",  # les Misérables
		"2680",  # Meditations
		"829",  # Gulliver's Travels into Several Remote Nations of the World
		"120",  # Treasure Island
		"12",  # Through the Looking-Glass
		"16",  # Peter Pan
		"60976",  # Rip Van Winkle
		"140",  # The Jungle
		"1399",  # Anna Karenina
		"56517",  # The Philosophy of Auguste Comte
		"52621",  # Society in America, Vol. 1
		"1228",  # On the Origin of Species by Means of Natural Selection
		"18269",  # Pascal's Penseés
		"2814",  # Dubliners
		"10554",  # Right Ho, Jeeves
		"10007",  # Carmilla
		"33944",  # How to Observe: Morals and Manners
	]

	BOOK_IDS_VALIDATION = [
		"11",  # Alice's Adventures in Wonderland
		"236",  # The Jungle Book
		"4351",  # The English Constitution
		"64317",  # The Great Gatsby
		"8438",  # The Ethics of Aristotle
		"26659",  # The Will to Believe, and Other Essays in Popular Philosophy
	]

	def __init__(self, validation: bool = False) -> None:
		"""Initialize the BookFetcher with a random book ID and cache status.

		Args:
			validation (bool): If True, use a predefined set of book IDs for validation.

		"""
		if validation:
			self.book_id = self.BOOK_IDS_VALIDATION[0]
		else:
			self.book_id = self.BOOK_IDS[0]
		self.is_cached = book_is_cached(self.book_id)

	def fetch_book_text(self) -> Generator[Any, Any, Any]:
		"""Generator that yields formatted text for each book in BOOK_IDS.
		
		Checks cache first, otherwise fetches from Gutendex API.
		Skips books that fail to fetch and continues to the next one.

		Yields:
			str: The formatted text of each book from Project Gutenberg.

		"""
		url = GUTENDEX_BASE_URL

		for book_id in self.BOOK_IDS:
			self.book_id = book_id
			self.is_cached = book_is_cached(self.book_id)

			if self.is_cached:
				yield get_cached_book(self.book_id)
				continue

			try:
				r = requests.get(f"{url}/{self.book_id}", timeout=10)
				r.raise_for_status()
				book_metadata = r.json()

				# Now fetch the actual text
				formats = book_metadata.get("formats", {})

				text_url = None
				# Try different text format keys based on what we see in the API response
				for fmt in [
					"text/plain; charset=utf-8",
					"text/plain; charset=us-ascii",
					"text/plain",
				]:
					if fmt in formats:
						text_url = formats[fmt]
						break

				if not text_url:
					logger.warning(f"No suitable text format found for book ID {self.book_id}, skipping...")
					continue

				text_response = requests.get(text_url, timeout=10)
				text_response.raise_for_status()
				book_text = text_response.text

				formatted_text = format_text(book_text)
				save_book(self.book_id, formatted_text)

				yield formatted_text

			except requests.RequestException as e:
				logger.warning(f"Error fetching book {self.book_id}: {e}, skipping...")
				continue
