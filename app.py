"""
Flask Backend for Price Intelligence Engine
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
from ml_model.price_predictor import PricePredictor

app = Flask(__name__)

# Load model and data
predictor = PricePredictor()
predictor.load_model()
df = pd.read_csv('data/processed/cleaned_prices.csv')

# Get unique values
categories = sorted(df['category'].unique().tolist())
brands = ['Your Brand'] + df['brand'].value_counts().head(20).index.tolist()

@app.route('/')
def home():
    return render_template('index.html', categories=categories, brands=brands)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    category = data.get('category')
    brand = data.get('brand')
    quality = float(data.get('quality', 4.0))
    
    # Get prediction
    result = predictor.predict_price(
        category,
        quality,
        None if brand == 'Your Brand' else brand
    )
    
    # Get category data for chart
    cat_data = df[df['category'] == category]
    price_distribution = cat_data['price'].tolist()
    
    return jsonify({
        'predicted_price': round(result['predicted_price'], 0),
        'market_min': round(result['market_stats']['min_price'], 0),
        'market_avg': round(result['market_stats']['avg_price'], 0),
        'market_max': round(result['market_stats']['max_price'], 0),
        'percentile': round(result['percentile'], 0),
        'product_count': result['market_stats']['product_count'],
        'price_distribution': price_distribution
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

