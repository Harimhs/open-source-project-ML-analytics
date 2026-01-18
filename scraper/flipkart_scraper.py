import requests
from bs4 import BeautifulSoup
import pandas as pd
from config import get_random_headers, MAX_PRODUCTS_PER_CATEGORY
from scraper.utils import random_delay, clean_price, clean_rating
from datetime import datetime

def scrape_flipkart_category(search_term, max_products=MAX_PRODUCTS_PER_CATEGORY):
    """
    Scrapes Flipkart with anti-blocking measures
    """
    products = []
    url = f"https://www.flipkart.com/search?q={search_term.replace(' ', '%20')}"
    
    try:
        headers = get_random_headers()
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code != 200:
            print(f"⚠️ Status code {resp.status_code} for {search_term}")
            return products
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Flipkart product containers (selectors may change)
        product_cards = soup.find_all('div', {'class': '_1AtVbE'})[:max_products]
        
        for card in product_cards:
            try:
                name_elem = card.find('a', {'class': 's1Q9rs'})
                name = name_elem.text.strip() if name_elem else None
                
                price_elem = card.find('div', {'class': '_30jeq3'})
                price_raw = price_elem.text.strip() if price_elem else None
                
                rating_elem = card.find('div', {'class': '_3LWZlK'})
                rating_raw = rating_elem.text.strip() if rating_elem else None
                
                if name and price_raw:
                    products.append({
                        'product_name': name[:100],
                        'price_raw': price_raw,
                        'price': clean_price(price_raw),
                        'rating': clean_rating(rating_raw),
                        'source': 'Flipkart',
                        'category': search_term,
                        'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
            except:
                continue
        
        print(f"✅ Scraped {len(products)} products from Flipkart - {search_term}")
        random_delay()
        
    except Exception as e:
        print(f"❌ Error scraping Flipkart: {e}")
    
    return products
