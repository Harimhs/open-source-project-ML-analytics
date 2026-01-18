"""
Simple Model Drift Detection
"""
import pandas as pd
import pickle
import json
from datetime import datetime

def check_drift():
    """Check if model performance is degrading"""
    
    # Load current data
    df = pd.read_csv('data/processed/cleaned_prices.csv')
    
    # Load model
    with open('data/models/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    
    # Calculate current statistics
    current_stats = {
        'timestamp': datetime.now().isoformat(),
        'total_products': len(df),
        'avg_price': float(df['price'].mean()),
        'price_std': float(df['price'].std()),
        'categories': df['category'].nunique(),
        'brands': df['brand'].nunique(),
        'model_name': model_info['name']
    }
    
    # Load previous stats (if exists)
    try:
        with open('data/drift_history.json', 'r') as f:
            history = json.load(f)
    except:
        history = []
    
    # Check for drift
    drift_detected = False
    alerts = []
    
    if len(history) > 0:
        last = history[-1]
        
        # Check price mean drift (>15% change)
        price_change = abs(current_stats['avg_price'] - last['avg_price']) / last['avg_price']
        if price_change > 0.15:
            drift_detected = True
            alerts.append(f"⚠️ Average price changed by {price_change*100:.1f}%")
        
        # Check data volume drift (>20% change)
        volume_change = abs(current_stats['total_products'] - last['total_products']) / last['total_products']
        if volume_change > 0.20:
            drift_detected = True
            alerts.append(f"⚠️ Product count changed by {volume_change*100:.1f}%")
    
    current_stats['drift_detected'] = drift_detected
    current_stats['alerts'] = alerts
    
    # Save to history
    history.append(current_stats)
    
    # Keep only last 30 days
    history = history[-30:]
    
    with open('data/drift_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Print report
    print("\n" + "="*60)
    print("📊 MODEL DRIFT CHECK")
    print("="*60)
    print(f"Timestamp: {current_stats['timestamp']}")
    print(f"Products: {current_stats['total_products']}")
    print(f"Avg Price: ₹{current_stats['avg_price']:.2f}")
    print(f"Categories: {current_stats['categories']}")
    print(f"Brands: {current_stats['brands']}")
    
    if drift_detected:
        print("\n⚠️  DRIFT DETECTED!")
        for alert in alerts:
            print(f"   {alert}")
        print("\n📝 Recommendation: Review model performance")
    else:
        print("\n✅ No significant drift detected")
    
    print("="*60)
    
    return drift_detected

if __name__ == "__main__":
    check_drift()
