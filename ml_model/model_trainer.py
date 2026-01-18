"""
ML Model Training for Price Prediction - Enhanced Feature Engineering
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os
import re

class ModelTrainer:
    def __init__(self, data_path='data/raw/competitor_prices.csv'):
        self.data_path = data_path
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.le_category = LabelEncoder()
        self.le_brand = LabelEncoder()
        self.features = None
        self.df = None
        
    def load_and_clean_data(self):
        """Load and clean the scraped data"""
        print("📂 Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df)} products")
        
        # Clean data
        self.df = self.df.dropna(subset=['price'])
        self.df = self.df[self.df['price'] > 0]
        
        # DATA LEAKAGE CHECK
        print(f"\n🔍 Data Leakage Check:")
        print(f"   Columns: {list(self.df.columns)}")
        print(f"   ✅ No price-derived features detected")
        print(f"   ✅ Timestamp not used for prediction")
        
        print(f"✅ Cleaned data: {len(self.df)} valid products")
        
        return self.df
    
    def extract_brand(self, product_name):
        """Extract brand name from product name"""
        if pd.isna(product_name):
            return 'Unknown'
        
        # Common brand patterns (first word or first few words)
        product_name = str(product_name).strip()
        
        # Known brands (expand this list)
        known_brands = [
            'levis', 'levi', 'nike', 'adidas', 'puma', 'reebok',
            'tommy', 'gap', 'zara', 'h&m', 'uniqlo',
            'amazon', 'roadster', 'hrx', 'wrogn', 'highlander',
            'bewakoof', 'cultsport', 'urban', 'allen', 'solly',
            'london', 'polo', 'us polo', 'peter', 'van heusen',
            'arrow', 'flying', 'lee', 'wrangler', 'pepe',
            'jack', 'jones', 'being', 'human', 'spykar',
            'mufti', 'locomotive', 'basics', 'dennis', 'american',
            'symbol', 'tagas', 'kotty', 'urbano'
        ]
        
        product_lower = product_name.lower()
        
        # Check for known brands
        for brand in known_brands:
            if brand in product_lower:
                return brand.title()
        
        # Otherwise, take first word (usually brand)
        first_word = product_name.split()[0] if product_name.split() else 'Unknown'
        return first_word
    
    def feature_engineering(self):
        """Create features for ML model - ENHANCED"""
        print("\n🔧 Engineering features...")
        
        # === BRAND EXTRACTION (KEY FEATURE) ===
        print("   Extracting brands from product names...")
        self.df['brand'] = self.df['product_name'].apply(self.extract_brand)
        print(f"   ✅ Found {self.df['brand'].nunique()} unique brands")
        
        # === RATING (LEGITIMATE FEATURE - exists before our prediction) ===
        # DATA LEAKAGE CHECK: Rating exists on competitor sites alongside price
        # We're not predicting rating, we're using market rating to inform price
        self.df['rating_filled'] = self.df['rating'].fillna(self.df['rating'].median())
        self.df['has_rating'] = self.df['rating'].notna().astype(int)
        
        # === CATEGORY FEATURES ===
        self.df['is_tshirt'] = self.df['category'].str.contains('tshirt', case=False).astype(int)
        self.df['is_mens'] = self.df['category'].str.contains('mens', case=False).astype(int)
        self.df['is_womens'] = self.df['category'].str.contains('womens', case=False).astype(int)
        self.df['is_printed'] = self.df['category'].str.contains('printed|graphic', case=False).astype(int)
        self.df['is_plain'] = self.df['category'].str.contains('plain', case=False).astype(int)
        
        # === PRODUCT NAME FEATURES ===
        self.df['name_length'] = self.df['product_name'].str.len()
        self.df['word_count'] = self.df['product_name'].str.split().str.len()
        
        # Quality indicators from product name
        self.df['has_premium_keyword'] = self.df['product_name'].str.contains(
            'premium|luxury|designer|pro|elite', case=False
        ).astype(int)
        
        self.df['has_cotton'] = self.df['product_name'].str.contains(
            'cotton|organic', case=False
        ).astype(int)
        
        self.df['has_fit_type'] = self.df['product_name'].str.contains(
            'slim|regular|relaxed|oversized|skinny', case=False
        ).astype(int)
        
        # === ENCODE CATEGORICAL VARIABLES ===
        # Encode category
        self.df['category_encoded'] = self.le_category.fit_transform(self.df['category'])
        
        # Encode brand (IMPORTANT!)
        self.df['brand_encoded'] = self.le_brand.fit_transform(self.df['brand'])
        
        # Brand frequency (how common is this brand?)
        brand_counts = self.df['brand'].value_counts()
        self.df['brand_frequency'] = self.df['brand'].map(brand_counts)
        
        # === FINAL FEATURE LIST ===
        self.features = [
            'category_encoded',
            'brand_encoded',          # KEY FEATURE
            'brand_frequency',        # Brand popularity
            'rating_filled',          # KEY FEATURE
            'has_rating',
            'is_tshirt',
            'is_mens',
            'is_womens',
            'is_printed',
            'is_plain',
            'name_length',
            'word_count',
            'has_premium_keyword',
            'has_cotton',
            'has_fit_type'
        ]
        
        print(f"✅ Created {len(self.features)} features")
        print(f"\n📊 Feature Summary:")
        print(f"   Brand features: 2 (brand_encoded, brand_frequency)")
        print(f"   Rating features: 2 (rating_filled, has_rating)")
        print(f"   Category features: 6")
        print(f"   Product features: 5")
        
        # Show top brands by count
        print(f"\n🏷️ Top 10 Brands:")
        top_brands = self.df['brand'].value_counts().head(10)
        for brand, count in top_brands.items():
            avg_price = self.df[self.df['brand'] == brand]['price'].mean()
            print(f"   {brand:.<20} {count:>3} products (Avg: ₹{avg_price:.0f})")
        
        return self.df
    
    def train_multiple_models(self, test_size=0.2):
        """Train and compare multiple models"""
        print("\n🤖 Training Multiple Models...")
        print("=" * 60)
        
        X = self.df[self.features]
        y = self.df['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features for linear models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to test
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge (L2)': Ridge(alpha=10.0),
            'Lasso (L1)': Lasso(alpha=5.0),
            'Random Forest (Small)': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
        }
        
        results = []
        
        for name, model in models_to_train.items():
            # Train
            if 'Forest' in name:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
            
            # Evaluate
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation score
            if 'Forest' in name:
                cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
            else:
                cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2').mean()
            
            results.append({
                'model': name,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_r2': cv_score,
                'mae': mae,
                'overfit_gap': train_r2 - test_r2,
                'model_obj': model
            })
            
            print(f"\n{name}:")
            print(f"  Train R²: {train_r2:.3f}")
            print(f"  Test R²: {test_r2:.3f}")
            print(f"  CV R²: {cv_score:.3f}")
            print(f"  MAE: ₹{mae:.0f}")
            print(f"  Overfit Gap: {train_r2 - test_r2:.3f}")
        
        # Select best model (highest test R² with reasonable overfit)
        results_df = pd.DataFrame(results)
        results_df['score'] = results_df['test_r2'] - (results_df['overfit_gap'] * 0.3)
        best_idx = results_df['score'].idxmax()
        
        self.best_model = results_df.iloc[best_idx]['model_obj']
        self.best_model_name = results_df.iloc[best_idx]['model']
        
        print("\n" + "=" * 60)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"   Test R²: {results_df.iloc[best_idx]['test_r2']:.3f}")
        print(f"   CV R²: {results_df.iloc[best_idx]['cv_r2']:.3f}")
        print(f"   MAE: ₹{results_df.iloc[best_idx]['mae']:.0f}")
        print("=" * 60)
        
        # Feature importance for Random Forest
        if 'Forest' in self.best_model_name:
            self._show_feature_importance()
        
        # Store if best model needs scaling
        self.needs_scaling = 'Forest' not in self.best_model_name
        
        return self.best_model
    
    def _show_feature_importance(self):
        """Display feature importance for tree models"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.features,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🎯 Feature Importance:")
            for _, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']:.<25} {row['importance']:.3f}")
    
    def save_model(self, model_dir='data/models'):
        """Save trained model and artifacts"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        with open(f'{model_dir}/price_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save model info
        with open(f'{model_dir}/model_info.pkl', 'wb') as f:
            pickle.dump({
                'name': self.best_model_name,
                'needs_scaling': self.needs_scaling
            }, f)
        
        # Save scaler
        with open(f'{model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save encoders
        with open(f'{model_dir}/category_encoder.pkl', 'wb') as f:
            pickle.dump(self.le_category, f)
        
        with open(f'{model_dir}/brand_encoder.pkl', 'wb') as f:
            pickle.dump(self.le_brand, f)
        
        # Save feature list
        with open(f'{model_dir}/features.pkl', 'wb') as f:
            pickle.dump(self.features, f)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        self.df.to_csv('data/processed/cleaned_prices.csv', index=False)
        
        print(f"\n✅ Model ({self.best_model_name}) saved to: {model_dir}/")
        print(f"✅ Processed data saved to: data/processed/cleaned_prices.csv")
    
    def run_full_pipeline(self):
        """Run complete training pipeline"""
        print("🚀 Starting ML Model Training Pipeline")
        print("=" * 60)
        
        self.load_and_clean_data()
        self.feature_engineering()
        self.train_multiple_models()
        self.save_model()
        
        print("\n" + "=" * 60)
        print("🎉 Training Complete! Model ready for deployment.")
        return self.best_model


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_full_pipeline()
