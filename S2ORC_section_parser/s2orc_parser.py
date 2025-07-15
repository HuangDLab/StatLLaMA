# s2orc_parser.py

import json
import re
import logging
from typing import List, Dict, Iterator
from tqdm import tqdm
import os

class S2orcParser:
    """
    A class to parse and filter academic papers from S2ORC JSON Lines dataset files.
    """
    def __init__(self, keywords: List[str]):
        """
        Initializes the parser with a set of keywords to filter for in section titles.

        Args:
            keywords (List[str]): A list of lower-cased keywords to search for.
        """
        if not all(isinstance(k, str) for k in keywords):
            raise TypeError("All keywords must be strings.")
            
        self.keywords = set(k.lower() for k in keywords)
        
        self.section_pattern = re.compile(
            r'\n([A-Z][A-Z\s\d\.-]{4,})\n\n'  # A title with at least 5 chars, allowing digits, dots, hyphens
            r'([\s\S]+?)'                     # The content of the section (non-greedy)
            r'(?=\n[A-Z][A-Z\s\d\.-]{4,}\n\n|\Z)', # Lookahead for the next title or end of string
            re.IGNORECASE
        )
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _stream_papers_from_file(self, filepath: str) -> Iterator[Dict]:
        """
        Streams paper data line by line from a JSON Lines file (even without a .jsonl extension).
        This is memory-efficient for large files.

        Args:
            filepath (str): Path to the S2ORC data file.

        Yields:
            Iterator[Dict]: A dictionary representing a single paper's raw JSON data.
        """
        logging.info(f"Streaming data from {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip(): # Ensure the line is not empty
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            logging.warning(f"Skipping malformed JSON line in {filepath}: {line[:100]}...")
                            continue
        except FileNotFoundError:
            logging.error(f"File not found: {filepath}")
            return

    def parse_and_filter_file(self, filepath: str) -> List[Dict]:
        """
        Parses a single S2ORC file, filtering for papers that contain relevant section titles.

        Args:
            filepath (str): The path to the S2ORC source file.

        Returns:
            A list of dictionaries, each representing a paper with its extracted sections.
        """
        filtered_papers = []
        # Use tqdm to show progress for iterating over the file stream
        file_stream = self._stream_papers_from_file(filepath)
        
        # We can't easily get the total number of lines for tqdm without reading the file twice,
        # so we'll just let it run without a total.
        for paper_data in tqdm(file_stream, desc=f"Parsing {os.path.basename(filepath)}"):
            # Normalize keys for compatibility (e.g., 'ArXiv' vs 'arxiv')
            external_ids = paper_data.get("external_ids", {}) or paper_data.get("externalids", {})
            arxiv_id = external_ids.get("ArXiv") or external_ids.get("arxiv")
            
            # The main content is typically in a single 'text' field within 'content'
            text_content = paper_data.get("text") # Simpler access pattern

            if not (arxiv_id and text_content and isinstance(text_content, str)):
                continue

            # Find all potential sections in one go
            all_matches = list(self.section_pattern.finditer(text_content))
            
            has_relevant_subtitle = any(
                any(keyword in match.group(1).strip().lower() for keyword in self.keywords)
                for match in all_matches
            )
            
            if has_relevant_subtitle:
                subtitles_dict = {}
                for match in all_matches:
                    title = match.group(1).strip()
                    content = match.group(2).strip()
                    if title: # Avoid empty titles
                        subtitles_dict[title] = content
                
                if subtitles_dict:
                    filtered_papers.append({
                        'arxiv_id': arxiv_id,
                        'subtitles': subtitles_dict
                    })

        logging.info(f"Finished processing {filepath}. Found {len(filtered_papers)} matching papers.")
        return filtered_papers

    def audit_subtitles_by_length(self, articles: List[Dict], threshold: int) -> List[Dict]:
        """
        Audits a list of parsed articles to find sections with content shorter than a threshold.
        This is useful for data quality checking.

        Args:
            articles (List[Dict]): A list of parsed paper dictionaries.
            threshold (int): The character length threshold.

        Returns:
            A list of dictionaries, each highlighting a section that failed the length check.
        """
        logging.info(f"Auditing {len(articles)} articles for section content shorter than {threshold} characters...")
        short_content_issues = []
        for article in tqdm(articles, desc="Auditing subtitles"):
            arxiv_id = article.get('arxiv_id', 'N/A')
            for subtitle, content in article.get('subtitles', {}).items():
                if any(keyword in subtitle.lower() for keyword in self.keywords):
                    if len(content) < threshold:
                        short_content_issues.append({
                            'arxiv_id': arxiv_id,
                            'matched_subtitle': subtitle,
                            'matched_content_length': len(content),
                            'matched_content_preview': content[:100] + '...' if len(content) > 100 else content
                        })
        logging.info(f"Audit complete. Found {len(short_content_issues)} sections with short content.")
        return short_content_issues