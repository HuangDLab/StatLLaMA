import requests
import re
import time
import logging
from bs4 import BeautifulSoup
from typing import List, Dict

class ArxivCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def fetch_paper_list(self, base_url: str, min_pages: int, max_pages: int) -> List[Dict]:
        """
        Fetches a list of paper metadata from a given arXiv search URL, paginating through results.

        Args:
            base_url (str): The base URL for the arXiv advanced search.
            min_pages (int): The starting page number to crawl (0-indexed).
            max_pages (int): The ending page number to crawl (exclusive).

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary contains metadata for a paper.
        """
        papers = []
        for page in range(min_pages, max_pages):
            # arXiv uses 'start' parameter for pagination, typically 200 results per page
            start_index = page * 200
            paged_url = f"{base_url}&start={start_index}"
            logging.info(f"Crawling page {page + 1}: {paged_url}")

            try:
                response = requests.get(paged_url, headers=self.headers, timeout=30)
                response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            except requests.exceptions.RequestException as e:
                logging.warning(f"Failed to access page {page + 1}: {e}. Skipping this page.")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('li', class_='arxiv-result')

            if not articles:
                logging.info(f"No more articles found on page {page + 1}. Stopping crawl.")
                break

            for article in articles:
                try:
                    title = article.find('p', class_='title').text.strip()
                    authors_p = article.find('p', class_='authors')
                    authors_text = authors_p.text.strip().replace("Authors:", "").strip()
                    authors = [author.strip() for author in authors_text.split(',')]
                    
                    # Handle expandable abstract
                    abstract_span = article.find('span', class_='abstract-full')
                    if abstract_span and abstract_span.text.strip():
                         abstract = abstract_span.text.strip().replace('\n', ' ').replace('△ Less', '').strip()
                    else:
                         abstract = article.find('p', class_='abstract').text.strip().replace('\n', ' ').replace('▽ More', '').strip()


                    pdf_link_tag = article.find('a', string='pdf')
                    arxiv_id = None
                    if pdf_link_tag and pdf_link_tag.get('href'):
                        # Regex to extract the arXiv ID (e.g., 2305.12345 or cs/0701150)
                        match = re.search(r'/(?:pdf|abs)/([\d\.]+|[a-zA-Z\-\.]+\/\d+)', pdf_link_tag['href'])
                        if match:
                            arxiv_id = match.group(1)

                    if arxiv_id:
                        papers.append({
                            'title': title,
                            'authors': authors,
                            'abstract': abstract,
                            'arxiv_id': arxiv_id
                        })
                except AttributeError as e:
                    logging.warning(f"Could not parse an article, likely due to missing tags: {e}. Skipping article.")
            
            # Be a polite crawler
            time.sleep(3)

        return papers
