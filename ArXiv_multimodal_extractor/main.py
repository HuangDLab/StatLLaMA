import json
import logging
from tqdm import tqdm
from arxiv_crawler import ArxivCrawler
from paper_parser import ArxivPaperParser

# Main Execution Logic
def main():
    # --- Configuration ---
    # URL from an advanced arXiv search for statistics papers within a specific date range.
    # Subject: stat.* (All statistics sub-categories)
    # Date range: 2020-07-05 to 2024-02-01
    BASE_URL = "https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=stat.*&terms-0-field=all&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date=2020-07-05&date-to_date=2024-02-01&date-date_type=submitted_date&abstracts=show&size=200&order=-announced_date_first"
    
    # Set the page range to crawl. Each page has 200 items.
    # For a full run, this would be a larger range, e.g., (0, 50)
    MIN_PAGES_TO_CRAWL = 38
    MAX_PAGES_TO_CRAWL = 40  # Set to a small range for testing purposes
    OUTPUT_FILENAME = 'arxiv_pretrain_data.json'

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Step 1: Crawl for Paper Metadata ---
    logging.info("Initializing arXiv crawler...")
    crawler = ArxivCrawler()
    papers_meta = crawler.fetch_paper_list(
        base_url=BASE_URL, 
        min_pages=MIN_PAGES_TO_CRAWL, 
        max_pages=MAX_PAGES_TO_CRAWL
    )
    logging.info(f"Crawling complete. Fetched metadata for {len(papers_meta)} papers.")

    if not papers_meta:
        logging.warning("No paper metadata was fetched. Exiting program.")
        return

    # --- Step 2: Initialize Paper Parser ---
    logging.info("Initializing paper parser...")
    parser = ArxivPaperParser() # Auto-detects CUDA

    # --- Step 3: Process Each Paper ---
    processed_papers = []
    for paper_meta in tqdm(papers_meta, desc="Parsing Papers"):
        arxiv_id = paper_meta.get('arxiv_id')
        abstract = paper_meta.get('abstract', '')

        if not arxiv_id:
            logging.warning(f"Skipping paper with missing arXiv ID: {paper_meta.get('title')}")
            continue
        
        logging.info(f"--- Processing paper: {arxiv_id} ---")
        parsed_sections = parser.parse_paper(arxiv_id, abstract)
        
        if parsed_sections:
            # Combine original metadata with parsed content
            full_paper_data = {**paper_meta, **parsed_sections}
            processed_papers.append(full_paper_data)
        else:
            logging.warning(f"Failed to parse paper {arxiv_id}, it will be excluded from the final output.")
    
    # --- Step 4: Filter for Complete Results ---
    final_papers = []
    for paper in processed_papers:
        # A simple validation rule: ensure the key sections we need have non-empty content.
        if paper.get('method') and paper.get('result') and paper.get('conclusion'):
            final_papers.append(paper)
    
    logging.info(f"After filtering for completeness, {len(final_papers)} papers remain.")

    # --- Step 5: Save Final Data to JSON ---
    logging.info(f"Saving final processed data to {OUTPUT_FILENAME}...")
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(final_papers, f, ensure_ascii=False, indent=4)
        logging.info(f"Successfully saved results to {OUTPUT_FILENAME}")
    except IOError as e:
        logging.error(f"Failed to write to file {OUTPUT_FILENAME}: {e}")

if __name__ == "__main__":
    main()