import requests
from bs4 import BeautifulSoup
import pandas as pd
from config import get_random_headers, MAX_PRODUCTS_PER_CATEGORY
from scraper.utils import random_delay, clean_price, clean_rating
from datetime import datetime

def scrape_amazon_category(search_term, max_products=25):
    """
    Scrapes Amazon India with anti-blocking measures
    """
    products = []
    url = f"https://www.amazon.in/s?k={search_term.replace(' ', '+')}"
    
    try:
        # Random headers for each request
        headers = get_random_headers()
        
        # Make request with timeout
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code != 200:
            print(f"⚠️ Status code {resp.status_code} for {search_term}")
            return products
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Find product containers
        items = soup.find_all('div', {'data-component-type': 's-search-result'})[:max_products]
        
        if not items:
            # Fallback selector
            items = soup.find_all('div', {'class': 's-result-item'})[:max_products]
        
        for item in items:
            try:
                # Extract product details
                title_elem = item.find('h2')
                title = title_elem.text.strip() if title_elem else None
                
                price_elem = item.find('span', {'class': 'a-price-whole'})
                price_raw = price_elem.text.strip() if price_elem else None
                
                rating_elem = item.find('span', {'class': 'a-icon-alt'})
                rating_raw = rating_elem.text.strip() if rating_elem else None
                
                if title and price_raw:
                    products.append({
                        'product_name': title[:100],  # Limit length
                        'price_raw': price_raw,
                        'price': clean_price(price_raw),
                        'rating': clean_rating(rating_raw),
                        'source': 'Amazon India',
                        'category': search_term,
                        'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
            except Exception as e:
                continue
        
        print(f"✅ Scraped {len(products)} products from Amazon - {search_term}")
        
        # Polite delay before next request
        random_delay()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error scraping Amazon: {e}")
    except Exception as e:
        print(f"❌ Error scraping Amazon: {e}")
    
    return products

def scrape_all_categories(categories):
    """Scrape multiple categories with delays"""
    all_products = []
    
    for category in categories:
        print(f"🔍 Scraping category: {category}")
        products = scrape_amazon_category(category)
        all_products.extend(products)
        
        # Longer delay between categories
        random_delay()
    
    return pd.DataFrame(all_products)
