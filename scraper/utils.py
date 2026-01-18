import time
import random
from config import MIN_DELAY, MAX_DELAY

def random_delay():
    """Human-like delay between requests to avoid detection"""
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    time.sleep(delay)

def clean_price(price_str):
    """Extract numeric price from string like '₹1,299' -> 1299"""
    import re
    if not price_str or price_str == "N/A":
        return None
    
    # Remove currency symbols and commas
    cleaned = re.sub(r'[₹,\s]', '', price_str)
    try:
        return float(cleaned)
    except:
        return None

def clean_rating(rating_str):
    """Extract numeric rating from string like '4.5 out of 5' -> 4.5"""
    import re
    if not rating_str or rating_str == "N/A":
        return None
    
    match = re.search(r'(\d+\.?\d*)', rating_str)
    return float(match.group(1)) if match else None
