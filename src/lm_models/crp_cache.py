"""
Chinese Restaurant Process (CRP) cache for tracking n-gram and channel counts.

This module implements the cache mechanism C^{i-1}_1 described in the paper,
which tracks counts of events (n-grams or channel substitutions) that have
occurred before position i in the current derivation.
"""

from collections import defaultdict
from typing import Dict, Tuple


class CRPCache:
    """Tracks n-gram counts for the Chinese Restaurant Process.
    
    Maintains counts C^{i-1}_1(context, char) representing how many times
    we've seen 'char' following 'context' in the current derivation.
    
    The cache is used in the CRP scoring formula:
    P(char | context) = (α * P0(char|context) + C(context, char)) / (α + C(context))
    """
    
    def __init__(self, n: int):
        """Initialize the cache.
        
        Args:
            n: The n-gram order (e.g., 3 for trigrams means context is 2 chars)
        """
        self.n = n
        # Maps (context_tuple) -> {char: count}
        self.ngram_counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def add_ngram(self, context: Tuple[str, ...], char: str) -> None:
        """Add an n-gram observation to the cache.
        
        Args:
            context: Tuple of previous characters (length n-1)
            char: The character following the context
        """
        self.ngram_counts[context][char] += 1
    
    def remove_ngram(self, context: Tuple[str, ...], char: str) -> None:
        """Remove an n-gram observation from the cache.
        
        Used when updating a derivation to remove old n-grams.
        
        Args:
            context: Tuple of previous characters (length n-1)
            char: The character following the context
        """
        if context in self.ngram_counts:
            if char in self.ngram_counts[context]:
                self.ngram_counts[context][char] -= 1
                if self.ngram_counts[context][char] <= 0:
                    del self.ngram_counts[context][char]
            if not self.ngram_counts[context]:
                del self.ngram_counts[context]
    
    def get_count(self, context: Tuple[str, ...], char: str) -> int:
        """Get the count for a specific n-gram.
        
        Args:
            context: Tuple of previous characters
            char: The character following the context
            
        Returns:
            int: Number of times this n-gram has been observed
        """
        return self.ngram_counts[context].get(char, 0)
    
    def get_context_total(self, context: Tuple[str, ...]) -> int:
        """Get the total count for all characters following a context.
        
        This is the denominator in the CRP formula: C(context) = sum of all C(context, char)
        
        Args:
            context: Tuple of previous characters
            
        Returns:
            int: Total count for this context
        """
        if context not in self.ngram_counts:
            return 0
        return sum(self.ngram_counts[context].values())
    
    def clear(self) -> None:
        """Clear all counts from the cache."""
        self.ngram_counts.clear()
    
    def copy(self) -> 'CRPCache':
        """Create a deep copy of this cache.
        
        Returns:
            CRPCache: A new cache with the same counts
        """
        new_cache = CRPCache(self.n)
        for context, char_counts in self.ngram_counts.items():
            for char, count in char_counts.items():
                new_cache.ngram_counts[context][char] = count
        return new_cache


class ChannelCache:
    """Tracks channel substitution counts for the CRP channel model.
    
    Maintains counts C^{i-1}_1(plaintext_char, cipher_symbol) representing
    how many times we've seen 'cipher_symbol' substituting for 'plaintext_char'
    in the current derivation.
    
    Used in the channel scoring formula:
    P(cipher | plain) = (β * P0(cipher|plain) + C(plain, cipher)) / (β + C(plain))
    """
    
    def __init__(self):
        """Initialize the channel cache."""
        # Maps plaintext_char -> {cipher_symbol: count}
        self.channel_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    
    def add_substitution(self, plaintext_char: str, cipher_symbol: int) -> None:
        """Add a channel substitution observation to the cache.
        
        Args:
            plaintext_char: The plaintext character
            cipher_symbol: The cipher symbol it maps to
        """
        self.channel_counts[plaintext_char][cipher_symbol] += 1
    
    def remove_substitution(self, plaintext_char: str, cipher_symbol: int) -> None:
        """Remove a channel substitution observation from the cache.
        
        Args:
            plaintext_char: The plaintext character
            cipher_symbol: The cipher symbol it maps to
        """
        if plaintext_char in self.channel_counts:
            if cipher_symbol in self.channel_counts[plaintext_char]:
                self.channel_counts[plaintext_char][cipher_symbol] -= 1
                if self.channel_counts[plaintext_char][cipher_symbol] <= 0:
                    del self.channel_counts[plaintext_char][cipher_symbol]
            if not self.channel_counts[plaintext_char]:
                del self.channel_counts[plaintext_char]
    
    def get_count(self, plaintext_char: str, cipher_symbol: int) -> int:
        """Get the count for a specific substitution.
        
        Args:
            plaintext_char: The plaintext character
            cipher_symbol: The cipher symbol
            
        Returns:
            int: Number of times this substitution has been observed
        """
        return self.channel_counts[plaintext_char].get(cipher_symbol, 0)
    
    def get_plaintext_total(self, plaintext_char: str) -> int:
        """Get the total count for all cipher symbols substituting a plaintext char.
        
        Args:
            plaintext_char: The plaintext character
            
        Returns:
            int: Total count for this plaintext character
        """
        if plaintext_char not in self.channel_counts:
            return 0
        return sum(self.channel_counts[plaintext_char].values())
    
    def clear(self) -> None:
        """Clear all counts from the cache."""
        self.channel_counts.clear()
    
    def copy(self) -> 'ChannelCache':
        """Create a deep copy of this cache.
        
        Returns:
            ChannelCache: A new cache with the same counts
        """
        new_cache = ChannelCache()
        for plaintext_char, cipher_counts in self.channel_counts.items():
            for cipher_symbol, count in cipher_counts.items():
                new_cache.channel_counts[plaintext_char][cipher_symbol] = count
        return new_cache
