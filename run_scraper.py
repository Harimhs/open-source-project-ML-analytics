"""
Scraper for Custom T-Shirt Printing Business
Focus: Competitive pricing intelligence for printed/custom apparel
"""
import sys
sys.path.append('.')

from scraper.amazon_scraper import scrape_all_categories
import pandas as pd

# T-SHIRT HEAVY categories (your friend's main business)
categories = [
    # === MEN'S T-SHIRTS (Primary Focus) ===
    "mens plain tshirt",
    "mens printed tshirt",
    "mens graphic tshirt",
    "mens round neck tshirt",
    "mens polo tshirt",
    "mens v neck tshirt",
    "mens oversized tshirt",
    "mens full sleeve tshirt",
    "mens cotton tshirt",
    "mens black tshirt",
    "mens white tshirt",
    "mens sports tshirt",
    "mens casual tshirt",
    "mens premium tshirt",
    
    # === WOMEN'S T-SHIRTS (Secondary Focus) ===
    "womens plain tshirt",
    "womens printed tshirt",
    "womens graphic tshirt",
    "womens oversized tshirt",
    "womens crop top tshirt",
    "womens cotton tshirt",
    "womens v neck tshirt",
    "womens black tshirt",
    "womens white tshirt",
    
    # === OTHER PRINTABLE APPAREL ===
    "mens hoodies",
    "womens hoodies",
    "mens sweatshirt",
    "womens sweatshirt",
    
    # === ADDITIONAL CLOTHING (Complementary) ===
    "mens slim fit jeans",
    "mens casual shirts",
    "womens jeans",
    "womens casual top",
    "womens kurti",
    "womens dress"
]

print("🎨 Custom T-Shirt Printing - Price Intelligence Engine")
print("=" * 60)
print(f"📊 Scraping {len(categories)} product categories")
print(f"🎯 Primary Focus: T-Shirts (Plain & Printed)")
print("=" * 60)
print()

# Scrape with increased product limit for t-shirts
amazon_df = scrape_all_categories(categories)

# Save to CSV
amazon_df.to_csv('data/raw/competitor_prices.csv', index=False)

print("\n" + "=" * 60)
print(f"✅ SCRAPING COMPLETE!")
print(f"📦 Total Products Scraped: {len(amazon_df)}")
print(f"📁 Saved to: data/raw/competitor_prices.csv")
print()

# Show breakdown
print("📊 CATEGORY BREAKDOWN:")
print("-" * 60)
category_counts = amazon_df.groupby('category').size().sort_values(ascending=False)
for category, count in category_counts.head(15).items():
    print(f"  {category:.<45} {count:>3} products")

print()
print("💰 PRICING INSIGHTS:")
print("-" * 60)
print(f"  Price Range: ₹{amazon_df['price'].min():.0f} - ₹{amazon_df['price'].max():.0f}")
print(f"  Average Price: ₹{amazon_df['price'].mean():.0f}")
print(f"  Median Price: ₹{amazon_df['price'].median():.0f}")

# T-shirt specific insights
tshirt_data = amazon_df[amazon_df['category'].str.contains('tshirt', case=False)]
print()
print("👕 T-SHIRT SPECIFIC INSIGHTS:")
print("-" * 60)
print(f"  Total T-Shirt Products: {len(tshirt_data)}")
print(f"  T-Shirt Avg Price: ₹{tshirt_data['price'].mean():.0f}")
print(f"  T-Shirt Price Range: ₹{tshirt_data['price'].min():.0f} - ₹{tshirt_data['price'].max():.0f}")

mens_tshirts = tshirt_data[tshirt_data['category'].str.contains('mens')]
womens_tshirts = tshirt_data[tshirt_data['category'].str.contains('womens')]
print(f"  Men's T-Shirts: {len(mens_tshirts)} products (Avg: ₹{mens_tshirts['price'].mean():.0f})")
print(f"  Women's T-Shirts: {len(womens_tshirts)} products (Avg: ₹{womens_tshirts['price'].mean():.0f})")

print("\n" + "=" * 60)
print("🚀 Ready for ML model training & dashboard deployment!")
