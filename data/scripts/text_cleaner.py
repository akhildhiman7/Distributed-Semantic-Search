"""
High-quality text cleaning utilities for arXiv data.
Handles LaTeX, HTML, special characters, and whitespace normalization.
"""
import re
import html
from typing import Optional
import unicodedata


class TextCleaner:
    """Production-grade text cleaning for academic papers."""
    
    # Regex patterns (compiled for performance)
    LATEX_PATTERNS = [
        (re.compile(r'\\[a-zA-Z]+\{([^}]*)\}'), r'\1'),  # \command{text} -> text
        (re.compile(r'\$([^$]+)\$'), r'\1'),              # $math$ -> math
        (re.compile(r'\\[a-zA-Z]+'), ''),                 # \command -> empty
        (re.compile(r'\$+'), ''),                         # remaining $ symbols
        (re.compile(r'\\\\'), ' '),                       # line breaks
        (re.compile(r'[_^]'), ' '),                       # subscripts/superscripts
    ]
    
    HTML_PATTERN = re.compile(r'<[^>]+>')
    URL_PATTERN = re.compile(r'https?://[^\s]+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    
    # Multiple spaces/newlines
    WHITESPACE_PATTERN = re.compile(r'\s+')
    
    # Special unicode characters that should be normalized
    UNICODE_REPLACEMENTS = {
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2026': '...',  # ellipsis
    }
    
    @classmethod
    def clean_latex(cls, text: str) -> str:
        """Remove LaTeX commands and formatting."""
        for pattern, replacement in cls.LATEX_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
    
    @classmethod
    def clean_html(cls, text: str) -> str:
        """Remove HTML tags and decode entities."""
        # Decode HTML entities first
        text = html.unescape(text)
        # Remove tags
        text = cls.HTML_PATTERN.sub('', text)
        return text
    
    @classmethod
    def remove_urls(cls, text: str) -> str:
        """Remove URLs and email addresses."""
        text = cls.URL_PATTERN.sub('', text)
        text = cls.EMAIL_PATTERN.sub('', text)
        return text
    
    @classmethod
    def normalize_unicode(cls, text: str) -> str:
        """Normalize unicode characters to ASCII equivalents."""
        # Replace specific unicode characters
        for unicode_char, replacement in cls.UNICODE_REPLACEMENTS.items():
            text = text.replace(unicode_char, replacement)
        
        # Normalize to NFKD form and remove combining characters
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(char for char in text if not unicodedata.combining(char))
        
        return text
    
    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """Normalize whitespace: collapse multiple spaces, remove leading/trailing."""
        text = cls.WHITESPACE_PATTERN.sub(' ', text)
        return text.strip()
    
    @classmethod
    def clean_text(
        cls,
        text: str,
        strip_latex: bool = True,
        strip_html: bool = True,
        remove_urls: bool = True,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
    ) -> str:
        """
        Apply full cleaning pipeline to text.
        
        Args:
            text: Input text to clean
            strip_latex: Remove LaTeX commands
            strip_html: Remove HTML tags
            remove_urls: Remove URLs and emails
            normalize_unicode: Normalize unicode to ASCII
            normalize_whitespace: Collapse and trim whitespace
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Apply transformations in order
        if strip_html:
            text = cls.clean_html(text)
        
        if strip_latex:
            text = cls.clean_latex(text)
        
        if remove_urls:
            text = cls.remove_urls(text)
        
        if normalize_unicode:
            text = cls.normalize_unicode(text)
        
        if normalize_whitespace:
            text = cls.normalize_whitespace(text)
        
        return text
    
    @classmethod
    def create_search_text(cls, title: str, abstract: str, separator: str = ". ") -> str:
        """
        Create combined search text from title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            separator: Text to join title and abstract
            
        Returns:
            Combined text for embedding generation
        """
        # Clean both fields
        title_clean = cls.clean_text(title)
        abstract_clean = cls.clean_text(abstract)
        
        # Ensure title ends with punctuation for natural flow
        if title_clean and title_clean[-1] not in '.!?':
            title_clean += '.'
        
        # Combine with separator
        combined = f"{title_clean}{separator}{abstract_clean}".strip()
        
        return combined
    
    @classmethod
    def validate_text(
        cls,
        text: str,
        min_length: int = 10,
        max_length: int = 10000,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate text meets quality criteria.
        
        Args:
            text: Text to validate
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length
            
        Returns:
            (is_valid, error_message)
        """
        if not text:
            return False, "Empty text"
        
        length = len(text)
        
        if length < min_length:
            return False, f"Text too short: {length} < {min_length}"
        
        if length > max_length:
            return False, f"Text too long: {length} > {max_length}"
        
        # Check for meaningful content (not just whitespace/punctuation)
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count < min_length // 2:
            return False, f"Insufficient alphabetic characters: {alpha_count}"
        
        return True, None
