import os
import json
import re
from parameter_validator import parameter_validator, non_blank_string


@parameter_validator(book_id=non_blank_string, book_text=non_blank_string)
def save_book(book_id: str, book_text: str) -> None:
	"""Save the fetched book text to a local file for caching.

	Args:
		book_id (str): The ID of the book.
		book_text (str): The entire contents of a book in string format.

	"""
	try:
		os.makedirs("books", exist_ok=True)
	except OSError as e:
		raise RuntimeError(f"Error creating books directory: {e}") from e

	try:
		with open(f"books/{book_id}.txt", "w", encoding="utf-8") as f:
			f.write(book_text)
	except (OSError, PermissionError) as e:
		raise RuntimeError(f"Error saving book {book_id}: {e}") from e


def book_is_cached(book_id: str) -> bool:
	"""Check if a book with the given ID exists in the local cache.

	Args:
		book_id (str): The ID of the book.

	Returns:
		bool: True if the book file exists, False otherwise.

	"""
	return os.path.exists(f"books/{book_id}.txt")


def get_cached_book(book_id: str) -> str:
	"""Retrieve the cached book text from a local file.

	Args:
		book_id (str): The ID of the book.

	Returns:
		str: The entire contents of the cached book in string format.

	Raises:
		FileNotFoundError: If the book file does not exist.

	"""
	if not os.path.exists(f"books/{book_id}.txt"):
		raise FileNotFoundError(f"Book with ID {book_id} is not cached.")
	with open(f"books/{book_id}.txt", encoding="utf-8") as f:
		return f.read()
