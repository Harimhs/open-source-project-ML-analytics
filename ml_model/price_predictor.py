"""
Price Prediction Class for Real-time Predictions
"""
import pickle
import numpy as np
import pandas as pd
import re

class PricePredictor:
    def __init__(self, model_dir='data/models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.le_category = None
        self.le_brand = None
        self.features = None
        self.model_info = None
        self.df = None
        
    def extract_brand(self, product_name):
        """Extract brand name from product name (same logic as training)"""
        if pd.isna(product_name):
            return 'Unknown'
        
        known_brands = [
            'levis', 'levi', 'nike', 'adidas', 'puma', 'reebok',
            'tommy', 'gap', 'zara', 'h&m', 'uniqlo',
            'amazon', 'roadster', 'hrx', 'wrogn', 'highlander',
            'bewakoof', 'cultsport', 'urban', 'allen', 'solly',
            'london', 'polo', 'us polo', 'peter', 'van heusen',
            'arrow', 'flying', 'lee', 'wrangler', 'pepe',
            'jack', 'jones', 'being', 'human', 'spykar',
            'mufti', 'locomotive', 'basics', 'dennis', 'american',
            'symbol', 'tagas', 'kotty', 'urbano', 'leotude',
            'max', 'juneberry', 'jockey'
        ]
        
        product_lower = str(product_name).lower()
        
        for brand in known_brands:
            if brand in product_lower:
                return brand.title()
        
        first_word = str(product_name).split()[0] if str(product_name).split() else 'Unknown'
        return first_word
        
    def load_model(self):
        """Load trained model and artifacts"""
        with open(f'{self.model_dir}/price_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open(f'{self.model_dir}/model_info.pkl', 'rb') as f:
            self.model_info = pickle.load(f)
        
        with open(f'{self.model_dir}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f'{self.model_dir}/category_encoder.pkl', 'rb') as f:
            self.le_category = pickle.load(f)
        
        with open(f'{self.model_dir}/brand_encoder.pkl', 'rb') as f:
            self.le_brand = pickle.load(f)
        
        with open(f'{self.model_dir}/features.pkl', 'rb') as f:
            self.features = pickle.load(f)
        
        # Load cleaned data for market analysis
        self.df = pd.read_csv('data/processed/cleaned_prices.csv')
        
        print(f"✅ Loaded model: {self.model_info['name']}")
        print(f"✅ Features: {len(self.features)}")
        return self
    
    def predict_price(self, category, quality_rating=4.0, brand_name=None):
        """Predict optimal price for a product"""
        
        # Extract features from category
        is_tshirt = 1 if 'tshirt' in category.lower() else 0
        is_mens = 1 if 'mens' in category.lower() else 0
        is_womens = 1 if 'womens' in category.lower() else 0
        is_printed = 1 if 'printed' in category.lower() or 'graphic' in category.lower() else 0
        is_plain = 1 if 'plain' in category.lower() else 0
        
        # Brand handling
        if brand_name is None:
            brand_name = 'Generic'
        
        # Encode category
        try:
            category_encoded = self.le_category.transform([category])[0]
        except:
            category_encoded = 0
        
        # Encode brand
        try:
            brand_encoded = self.le_brand.transform([brand_name])[0]
        except:
            brand_encoded = 0
        
        # Brand frequency (from training data)
        brand_frequency = len(self.df[self.df['brand'] == brand_name])
        if brand_frequency == 0:
            brand_frequency = 1
        
        # Product name features
        product_name = f"{brand_name} {category}"
        name_length = len(product_name)
        word_count = len(product_name.split())
        
        has_premium_keyword = 1 if any(k in product_name.lower() for k in ['premium', 'luxury', 'designer', 'pro', 'elite']) else 0
        has_cotton = 1 if 'cotton' in product_name.lower() or 'organic' in product_name.lower() else 0
        has_fit_type = 1 if any(k in product_name.lower() for k in ['slim', 'regular', 'relaxed', 'oversized', 'skinny']) else 0
        
        has_rating = 1
        
        # Build feature vector (MUST match training order)
        feature_values = np.array([[
            category_encoded,
            brand_encoded,
            brand_frequency,
            quality_rating,
            has_rating,
            is_tshirt,
            is_mens,
            is_womens,
            is_printed,
            is_plain,
            name_length,
            word_count,
            has_premium_keyword,
            has_cotton,
            has_fit_type
        ]])
        
        # Scale if needed
        if self.model_info['needs_scaling']:
            feature_values = self.scaler.transform(feature_values)
        
        # Predict
        predicted_price = self.model.predict(feature_values)[0]
        
        # Get market statistics
        market_stats = self.get_market_stats(category)
        
        # Calculate position
        percentile = self.calculate_market_position(predicted_price, category)
        
        return {
            'predicted_price': round(max(predicted_price, 99), 2),  # Floor at ₹99
            'market_stats': market_stats,
            'percentile': percentile,
            'model_used': self.model_info['name'],
            'brand_used': brand_name
        }
    
    def get_market_stats(self, category):
        """Get market statistics for a category"""
        category_data = self.df[self.df['category'] == category]
        
        if len(category_data) == 0:
            similar_data = self.df[self.df['category'].str.contains(
                category.split()[0], case=False, na=False
            )]
            category_data = similar_data if len(similar_data) > 0 else self.df
        
        return {
            'avg_price': round(category_data['price'].mean(), 2),
            'min_price': round(category_data['price'].min(), 2),
            'max_price': round(category_data['price'].max(), 2),
            'median_price': round(category_data['price'].median(), 2),
            'product_count': len(category_data)
        }
    
    def calculate_market_position(self, price, category):
        """Calculate where price sits in market distribution"""
        category_data = self.df[self.df['category'] == category]
        
        if len(category_data) == 0:
            return 50.0
        
        lower_count = len(category_data[category_data['price'] < price])
        percentile = (lower_count / len(category_data)) * 100
        
        return round(percentile, 1)
    
    def get_top_brands(self):
        """Get top brands by frequency"""
        return self.df['brand'].value_counts().head(20).to_dict()


if __name__ == "__main__":
    print("🧪 Testing Price Predictor")
    print("=" * 60)
    
    predictor = PricePredictor()
    predictor.load_model()
    
    test_cases = [
        ("mens printed tshirt", 4.0, "Bewakoof"),
        ("womens plain tshirt", 4.5, "Amazon"),
        ("mens hoodies", 4.0, "Nike"),
    ]
    
    for category, rating, brand in test_cases:
        result = predictor.predict_price(category, rating, brand)
        print(f"\n📦 {brand} - {category} (Rating: {rating})")
        print(f"💰 Predicted Price: ₹{result['predicted_price']}")
        print(f"📊 Market Average: ₹{result['market_stats']['avg_price']}")
        print(f"🎯 Market Position: {result['percentile']}th percentile")
