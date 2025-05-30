import os
import time
import random
import logging
import argparse
import requests
import datetime
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote
from playwright.sync_api import sync_playwright
from concurrent.futures import ThreadPoolExecutor, as_completed

def setup_parser():
    parser = argparse.ArgumentParser(description="Scrape 20 pages of electronic product data from Amazon with up to 100 reviews per product")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save CSV files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--region", type=str, default="in", choices=["com", "in"], help="Amazon region (default: in)")
    parser.add_argument("--use_playwright", action="store_true", help="Use Playwright for dynamic content (default: False)")
    return parser

def setup_logging(verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"amazon_scraper_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

def get_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

def fetch_page(url, headers, use_playwright=False, proxies=None):
    try:
        if use_playwright:
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    html = page.content()
                    browser.close()
                    return html
            except Exception as e:
                logging.warning(f"Playwright failed for {url}: {str(e)}. Falling back to requests.")
        
        response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
        if response.status_code != 200:
            logging.error(f"Failed to fetch {url}: Status code {response.status_code}")
            return None
        return response.text
    except Exception as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return None

def scrape_reviews(product_url, region, headers, use_playwright=False, proxies=None, asin=None):
    logging.debug(f"Scraping reviews for {product_url} (ASIN: {asin})")
    try:
        reviews = []
        max_reviews = 100  # Target number of reviews
        max_pages = 10     # Maximum review pages to scrape
        retries = 3        # Number of retries for failed requests

        # Fetch the product page to find the reviews link or confirm ASIN
        for attempt in range(retries):
            html = fetch_page(product_url, headers, use_playwright, proxies)
            if html:
                break
            logging.warning(f"Attempt {attempt + 1} failed for {product_url}. Retrying...")
            time.sleep(random.uniform(2, 5))
        else:
            logging.error(f"Failed to fetch {product_url} after {retries} attempts.")
            return "No reviews found"

        soup = BeautifulSoup(html, "html.parser")
        
        # Try to extract ASIN from the product page if not provided
        if not asin:
            asin_elem = (
                soup.select_one("input[name='ASIN']") or
                soup.select_one("input[name='asin']") or
                soup.select_one("[data-asin]") or
                soup.select_one("div[data-asin]")
            )
            asin = asin_elem.get("value") or asin_elem.get("data-asin") if asin_elem else None
            if not asin:
                logging.error(f"Could not extract ASIN for {product_url}")
                return "No reviews found"
            logging.debug(f"Extracted ASIN: {asin}")

        # Find the "See all reviews" link
        reviews_link = (
            soup.select_one("a[data-hook='see-all-reviews-link']") or 
            soup.select_one("a[href*='product-reviews']") or
            soup.select_one("a[href*='customer-reviews']") or 
            soup.select_one("a.reviews-link") or 
            soup.select_one("a[href*='reviews']") or 
            soup.select_one("a[class*='reviews']")
        )
        
        # Construct the base reviews URL
        if reviews_link and 'product-reviews' in reviews_link.get('href', ''):
            # Clean URL to remove unnecessary parameters
            base_reviews_url = f"https://www.amazon.{region}{reviews_link['href'].split('?')[0]}"
            logging.debug(f"Found reviews link: {base_reviews_url}")
        else:
            # Fallback: Use simple URL format
            base_reviews_url = f"https://www.amazon.{region}/product-reviews/{asin}"
            logging.debug(f"Constructed reviews URL using ASIN: {base_reviews_url}")

        # Scrape review pages concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_page = {
                executor.submit(fetch_page, f"{base_reviews_url}?pageNumber={i}", headers, use_playwright, proxies): i 
                for i in range(1, max_pages + 1)
            }
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                for attempt in range(retries):
                    try:
                        reviews_html = future.result()
                        if not reviews_html:
                            logging.debug(f"No HTML content for review page {page_num} of {product_url}")
                            break
                        
                        reviews_soup = BeautifulSoup(reviews_html, "html.parser")
                        review_divs = (
                            reviews_soup.select("div[data-hook='review']") or 
                            reviews_soup.select("div.review") or 
                            reviews_soup.select("div.a-section.review") or 
                            reviews_soup.select("div[class*='review']") or
                            reviews_soup.select("div.review-container") or
                            reviews_soup.select("div.a-row.a-spacing-small.review-data")
                        )
                        
                        if not review_divs:
                            logging.debug(f"No reviews found on page {page_num} for {product_url}")
                            break
                        
                        for review in review_divs:
                            review_text = (
                                review.select_one("span[data-hook='review-body']") or 
                                review.select_one("div.reviewText") or 
                                review.select_one("span.review-text") or 
                                review.select_one("span.review-text-content") or 
                                review.select_one("div.a-row.review-data") or 
                                review.select_one("span[class*='review-text']") or
                                review.select_one("span.a-size-base.review-text")
                            )
                            review_text = review_text.get_text(strip=True) if review_text else "No text"
                            if review_text != "No text" and len(reviews) < max_reviews:
                                reviews.append(review_text)
                        
                        logging.debug(f"Collected {len(reviews)} reviews from page {page_num} for {product_url}")
                        break  # Exit retry loop if successful
                    except Exception as e:
                        logging.warning(f"Attempt {attempt + 1} failed for review page {page_num} of {product_url}: {str(e)}")
                        # Try Playwright as fallback on last attempt
                        if attempt == retries - 1 and not use_playwright:
                            logging.debug(f"Retrying review page {page_num} with Playwright")
                            html = fetch_page(f"{base_reviews_url}?pageNumber={page_num}", headers, True, proxies)
                            if html:
                                reviews_soup = BeautifulSoup(html, "html.parser")
                                review_divs = (
                                    reviews_soup.select("div[data-hook='review']") or 
                                    reviews_soup.select("div.review") or 
                                    reviews_soup.select("div.a-section.review") or 
                                    reviews_soup.select("div[class*='review']") or
                                    reviews_soup.select("div.review-container") or
                                    reviews_soup.select("div.a-row.a-spacing-small.review-data")
                                )
                                for review in review_divs:
                                    review_text = (
                                        review.select_one("span[data-hook='review-body']") or 
                                        review.select_one("div.reviewText") or 
                                        review.select_one("span.review-text") or 
                                        review.select_one("span.review-text-content") or 
                                        review.select_one("div.a-row.review-data") or 
                                        review.select_one("span[class*='review-text']") or
                                        review.select_one("span.a-size-base.review-text")
                                    )
                                    review_text = review_text.get_text(strip=True) if review_text else "No text"
                                    if review_text != "No text" and len(reviews) < max_reviews:
                                        reviews.append(review_text)
                                logging.debug(f"Collected {len(reviews)} reviews from page {page_num} with Playwright")
                                break
                        time.sleep(random.uniform(2, 5))
                
                if len(reviews) >= max_reviews:
                    logging.debug(f"Reached maximum reviews ({max_reviews}) for {product_url}")
                    break
                if not review_divs and page_num > 1:  # Stop if no reviews found on a later page
                    logging.debug(f"Stopping review scrape for {product_url} at page {page_num}: no more reviews")
                    break
                time.sleep(random.uniform(2, 5))
        
        # Fallback: Check product page for reviews
        if not reviews:
            review_divs = (
                soup.select("div[data-hook='review']") or 
                soup.select("div.review") or 
                soup.select("div.a-section.review") or 
                soup.select("div[class*='review']") or
                soup.select("div.review-container") or
                soup.select("div.a-row.a-spacing-small.review-data")
            )
            for review in review_divs:
                review_text = (
                    review.select_one("span[data-hook='review-body']") or 
                    review.select_one("div.reviewText") or 
                    review.select_one("span.review-text") or 
                    review.select_one("span.review-text-content") or 
                    review.select_one("div.a-row.review-data") or 
                    review.select_one("span[class*='review-text']") or
                    review.select_one("span.a-size-base.review-text")
                )
                review_text = review_text.get_text(strip=True) if review_text else "No text"
                if review_text != "No text" and len(reviews) < max_reviews:
                    reviews.append(review_text)
                if len(reviews) >= max_reviews:
                    break
        
        # Log if no reviews were found
        if not reviews:
            review_section = soup.select_one("div#reviewsMedley") or soup.select_one("div[class*='reviews']")
            logging.debug(f"No reviews found for {product_url}. Review section: {str(review_section)[:500] if review_section else 'None'}")
        
        result = "; ".join(reviews) if reviews else "No reviews found"
        logging.info(f"Collected {len(reviews)} reviews for {product_url}")
        return result
    except Exception as e:
        logging.error(f"Error scraping reviews for {product_url}: {str(e)}")
        return "No reviews found"

def scrape_product(product, domain, region, headers, use_playwright, proxies):
    try:
        title_elem = (
            product.select_one("h2 a span") or 
            product.select_one("span.a-text-normal") or 
            product.select_one("div.s-title-instructions span") or 
            product.select_one("h2 span") or 
            product.select_one("span.s-title")
        )
        title = title_elem.get_text(strip=True) if title_elem else "N/A"
        
        price_elem = (
            product.select_one("span.a-price span.a-offscreen") or 
            product.select_one("span.a-price-whole") or 
            product.select_one("span.a-price") or 
            product.select_one("div.a-price") or 
            product.select_one("span.a-color-price") or 
            product.select_one("span.price")
        )
        price = price_elem.get_text(strip=True) if price_elem else "N/A"
        
        rating_elem = (
            product.select_one("span.a-icon-alt") or 
            product.select_one("span[aria-label*='out of 5 stars']") or 
            product.select_one("i.a-icon-star") or 
            product.select_one("span.a-icon-star") or 
            product.select_one("span.a-star")
        )
        rating = rating_elem.get_text(strip=True).split()[0] if rating_elem else "N/A"
        
        asin_elem = product.get("data-asin")
        asin = asin_elem if asin_elem else "N/A"
        
        url_elem = (
            product.select_one("h2 a") or 
            product.select_one("a.a-link-normal.s-no-outline") or 
            product.select_one("a.a-link-normal") or 
            product.select_one("a.s-title-instructions") or 
            product.select_one("a.s-title")
        )
        product_url = domain + url_elem["href"] if url_elem and url_elem.get("href") else "N/A"
        
        reviews = scrape_reviews(product_url, region, headers, use_playwright, proxies, asin=asin) if product_url != "N/A" and asin != "N/A" else "No reviews found"
        
        if title != "N/A" or price != "N/A":
            return {
                "Title": title,
                "Price": price,
                "Rating": rating,
                "ASIN": asin,
                "Reviews": reviews,
                "Category": ""  # Category will be set in scrape_amazon
            }
        return None
    except Exception as e:
        logging.error(f"Error processing product: {str(e)}")
        return None

def scrape_amazon(search_query, pages, region, output_file, verbose=False, use_playwright=False):
    base_url = f"https://www.amazon.{region}/s"
    domain = f"https://www.amazon.{region}"
    products = []
    headers = get_headers()
    proxies = None  # Add your proxy, e.g., {"http": "http://your_proxy:port", "https": "http://your_proxy:port"}
    
    def fetch_search_page(page):
        url = f"{base_url}?k={quote(search_query)}&page={page}"
        logging.info(f"Scraping {search_query} page {page}: {url}")
        html = fetch_page(url, headers, use_playwright, proxies)
        if not html:
            return []
        
        soup = BeautifulSoup(html, "html.parser")
        product_divs = (
            soup.select("div.s-result-item[data-component-type='s-search-result']") or 
            soup.select("div.s-main-slot div[data-component-type='s-search-result']") or 
            soup.select("div.s-result-item") or 
            soup.select("div.s-main-slot div") or 
            soup.select("div[data-component-type='s-search-result']")
        )
        
        if not product_divs:
            logging.warning(f"No products found for {search_query} on page {page}. Possible CAPTCHA or layout change.")
            return []
        
        page_products = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_product = {
                executor.submit(scrape_product, product, domain, region, headers, use_playwright, proxies): product 
                for product in product_divs
            }
            for future in as_completed(future_to_product):
                product_data = future.result()
                if product_data:
                    product_data["Category"] = search_query
                    page_products.append(product_data)
        
        logging.info(f"Scraped {len(page_products)} products for {search_query} on page {page}")
        return page_products
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_page = {executor.submit(fetch_search_page, page): page for page in range(1, pages + 1)}
        for future in as_completed(future_to_page):
            products.extend(future.result())
            time.sleep(random.uniform(2, 5))
    
    if products:
        # Remove URL and Page columns before saving
        df = pd.DataFrame(products)
        df = df[["Title", "Price", "Rating", "ASIN", "Reviews", "Category"]]  # Exclude URL and Page
        df.to_csv(output_file, index=False, encoding="utf-8")
        logging.info(f"Saved {len(products)} products for {search_query} to {output_file}")
    else:
        logging.warning(f"No products scraped for {search_query}")
    
    return products

def main():
    parser = setup_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    categories = {
        "Mobiles": "smartphones",
        "Headphones": "headphones",
        "TV": "led smart tv",  # Updated query
        "Smart Watches": "smartwatch"  # Updated query
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_products = []
    for category, query in categories.items():
        output_file = os.path.join(args.output_dir, f"{category.lower().replace(' ', '_')}_data.csv")
        logging.info(f"Starting scrape for {category} (query: {query})")
        try:
            products = scrape_amazon(query, 20, args.region, output_file, args.verbose, args.use_playwright)
            all_products.extend(products)
        except Exception as e:
            logging.error(f"Error processing category {category}: {str(e)}")
            continue
    
    if all_products:
        combined_file = os.path.join(args.output_dir, "all_electronics_data.csv")
        df = pd.DataFrame(all_products)
        df = df[["Title", "Price", "Rating", "ASIN", "Reviews", "Category"]]  # Exclude URL and Page
        df.to_csv(combined_file, index=False, encoding="utf-8")
        logging.info(f"Saved {len(all_products)} total products to {combined_file}")



if __name__ == "__main__":
    main()