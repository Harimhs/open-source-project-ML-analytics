import random
from fake_useragent import UserAgent

# User Agents rotation to avoid detection
ua = UserAgent()

USER_AGENTS = [
    ua.chrome,
    ua.firefox,
    ua.safari,
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
]

def get_random_headers():
    """Returns randomized headers to mimic real browser"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

# Rate limiting settings
MIN_DELAY = 3  # Minimum seconds between requests
MAX_DELAY = 7  # Maximum seconds between requests

# Scraping limits
MAX_PRODUCTS_PER_CATEGORY = 15
