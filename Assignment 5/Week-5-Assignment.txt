objective :  code for House Price Prediction
Data Preprocessing and feature engineering
Resources :
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

Code :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        
    def load_data(self, train_path, test_path=None):
        """Load training and test data"""
        self.train_df = pd.read_csv(train_path)
        if test_path:
            self.test_df = pd.read_csv(test_path)
        print(f"Training data shape: {self.train_df.shape}")
        return self.train_df
    
    def explore_data(self):
        """Basic data exploration"""
        print("Missing Values:")
        missing = self.train_df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        print(missing.head(10))
        
        # Target distribution
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(self.train_df['SalePrice'], kde=True)
        plt.title('Sale Price Distribution')
        
        plt.subplot(1, 2, 2)
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
        correlations = self.train_df[numeric_cols].corr()['SalePrice'].sort_values(ascending=False)
        correlations.head(10).plot(kind='barh')
        plt.title('Top Correlations with Sale Price')
        plt.tight_layout()
        plt.show()
        
        return correlations
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        df = df.copy()
        
        # Fill categorical with 'None'
        categorical_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                           'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                           'MasVnrType']
        
        for col in categorical_none:
            if col in df.columns:
                df[col] = df[col].fillna('None')
        
        # Fill numerical with 0
        numerical_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
                         'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
        
        for col in numerical_zero:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill remaining with mode/median
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def create_features(self, df):
        """Feature engineering"""
        df = df.copy()
        
        # Total areas
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['TotalBathrooms'] = df['FullBath'] + df['HalfBath']*0.5 + df['BsmtFullBath'] + df['BsmtHalfBath']*0.5
        df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
        
        # Age features
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        
        # Quality mapping
        quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
        quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'KitchenQual', 'FireplaceQu', 'GarageQual']
        
        for col in quality_cols:
            if col in df.columns:
                df[col + '_num'] = df[col].map(quality_map)
        
        # Binary features
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
        
        return df
    
    def encode_categorical(self, df, fit_encoders=True):
        """Encode categorical features"""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    known_labels = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(lambda x: x if x in known_labels else 'Unknown')
                    if 'Unknown' not in known_labels:
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'Unknown')
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def remove_outliers(self, df):
        """Remove outliers using IQR method"""
        Q1 = df['SalePrice'].quantile(0.25)
        Q3 = df['SalePrice'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df['SalePrice'] < lower_bound) | (df['SalePrice'] > upper_bound)
        print(f"Removing {outliers_mask.sum()} outliers")
        
        return df[~outliers_mask]
    
    def transform_skewed_features(self, df):
        """Apply log transformation to skewed features"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'SalePrice']
        
        skewed_features = []
        for col in numeric_cols:
            if df[col].min() >= 0 and abs(skew(df[col])) > 0.75:
                skewed_features.append(col)
        
        for col in skewed_features:
            df[col] = np.log1p(df[col])
        
        # Transform target
        if 'SalePrice' in df.columns:
            df['SalePrice'] = np.log1p(df['SalePrice'])
        
        return df, skewed_features
    
    def preprocess_data(self):
        """Complete preprocessing pipeline"""
        print("Preprocessing data...")
        
        # Handle missing values
        self.train_df = self.handle_missing_values(self.train_df)
        if hasattr(self, 'test_df'):
            self.test_df = self.handle_missing_values(self.test_df)
        
        # Create features
        self.train_df = self.create_features(self.train_df)
        if hasattr(self, 'test_df'):
            self.test_df = self.create_features(self.test_df)
        
        # Remove outliers
        self.train_df = self.remove_outliers(self.train_df)
        
        # Transform skewed features
        self.train_df, self.skewed_features = self.transform_skewed_features(self.train_df)
        if hasattr(self, 'test_df'):
            for col in self.skewed_features:
                if col in self.test_df.columns:
                    self.test_df[col] = np.log1p(self.test_df[col])
        
        # Encode categorical
        self.train_df = self.encode_categorical(self.train_df, fit_encoders=True)
        if hasattr(self, 'test_df'):
            self.test_df = self.encode_categorical(self.test_df, fit_encoders=False)
        
        print(f"Final training shape: {self.train_df.shape}")
        
    def train_models(self):
        """Train and compare models"""
        # Prepare data
        X = self.train_df.drop(['SalePrice', 'Id'], axis=1, errors='ignore')
        y = self.train_df['SalePrice']
        
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Models to try
        models = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.001),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            results[name] = {'model': model, 'rmse': rmse, 'r2': r2}
            print(f"{name}: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        # Select best model
        best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
        self.model = results[best_model]['model']
        print(f"\nBest model: {best_model}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            plt.show()
        
        return results
    
    def make_predictions(self):
        """Generate predictions"""
        if hasattr(self, 'test_df'):
            X_test = self.test_df.drop(['Id'], axis=1, errors='ignore')
            X_test_scaled = self.scaler.transform(X_test)
            
            predictions = self.model.predict(X_test_scaled)
            predictions = np.expm1(predictions)  # Transform back from log
            
            # Save predictions
            submission = pd.DataFrame({
                'Id': self.test_df['Id'],
                'SalePrice': predictions
            })
            submission.to_csv('house_price_predictions.csv', index=False)
            print("Predictions saved to 'house_price_predictions.csv'")
            
            return predictions
        else:
            print("No test data available")
            return None
    
    def run_pipeline(self, train_path, test_path=None):
        """Run complete pipeline"""
        print("🏠 House Price Prediction Pipeline")
        print("=" * 40)
        
        # Load and explore data
        self.load_data(train_path, test_path)
        self.explore_data()
        
        # Preprocess
        self.preprocess_data()
        
        # Train models
        results = self.train_models()
        
        # Make predictions
        predictions = self.make_predictions()
        
        return results, predictions

# Usage
if __name__ == "__main__":
    predictor = HousePricePredictor()
    
    # Run pipeline
    # results, predictions = predictor.run_pipeline('train.csv', 'test.csv')
    
    print("\nTo use this code:")
    print("1. Download dataset from Kaggle")
    print("2. Run: predictor.run_pipeline('train.csv', 'test.csv')")
