# Install ONLY necessary packages (minimal set)
!pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly

# For advanced time series and ML (optional but recommended)
!pip install xgboost lightgbm statsmodels

# For similarity search
!pip install dtaidistance

# Kaggle hub for data loading
!pip install kagglehub

print("✅ All dependencies installed!")

# Core data manipulation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
import scipy.optimize as optimize

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Advanced ML (optional)
try:
    import xgboost as xgb
    import lightgbm as lgb
except:
    print("⚠️ XGBoost/LightGBM not available, using scikit-learn only")

# Time Series Analysis
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
except:
    print("⚠️ statsmodels not available, using basic time series")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# System & Utilities
import os
import json
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass  # Fixed: dataclasses (plural)

print("✅ All required libraries imported!")

class GoldDataLoader:
    def __init__(self):
        self.data = None
        self.features = None
        
    def load_kaggle_dataset(self, dataset_name: str = "asiluxresearch/xauusd-dalyop") -> pd.DataFrame:
        """Load dataset using Kaggle Hub with flexible file discovery"""
        try:
            print("📥 Downloading dataset using Kaggle Hub...")
            import kagglehub
            path = kagglehub.dataset_download(dataset_name)
            print(f"✅ Dataset downloaded to: {path}")
            
            # Discover CSV files in the downloaded path
            csv_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                raise ValueError(f"No CSV files found in {path}")
            
            print(f"📁 Found {len(csv_files)} CSV files. Attempting to load...")
            
            # Try each CSV file until we successfully load one
            for csv_file in csv_files:
                print(f"🔄 Trying: {os.path.basename(csv_file)}")
                try:
                    # Try with date parsing first
                    self.data = pd.read_csv(csv_file, parse_dates=['Date'])
                    print(f"✅ Successfully loaded: {os.path.basename(csv_file)}")
                    break
                except Exception as e:
                    print(f"❌ Failed to load {os.path.basename(csv_file)} with date parsing: {e}")
                    # Try without date parsing
                    try:
                        self.data = pd.read_csv(csv_file)
                        # Try to find and parse date column
                        date_columns = [col for col in self.data.columns if 'date' in col.lower() or 'time' in col.lower()]
                        if date_columns:
                            self.data[date_columns[0]] = pd.to_datetime(self.data[date_columns[0]])
                            self.data = self.data.set_index(date_columns[0])
                            print(f"✅ Loaded with date column detection: {os.path.basename(csv_file)}")
                            break
                    except Exception as e2:
                        print(f"❌ Failed to load {os.path.basename(csv_file)}: {e2}")
                        continue
            
            if self.data is None:
                raise ValueError("Could not load any CSV files from the dataset")
            
            # Ensure we have a datetime index
            if not isinstance(self.data.index, pd.DatetimeIndex):
                # Try to use first column as index if it looks like dates
                try:
                    self.data.index = pd.to_datetime(self.data.iloc[:, 0])
                    self.data = self.data.iloc[:, 1:]
                    print("✅ Set first column as datetime index")
                except:
                    print("⚠️ Could not parse dates, using integer index")
            
            # Sort by index
            self.data = self.data.sort_index()
            
            # Standardize column names
            column_mapping = {
                'close': 'Close', 'price': 'Close', 'value': 'Close', 'adj close': 'Close',
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'volume': 'Volume', 'vol': 'Volume',
                'xauusd': 'Close', 'gold': 'Close', 'xau': 'Close'
            }
            
            new_columns = []
            for col in self.data.columns:
                col_lower = str(col).lower().strip()
                new_columns.append(column_mapping.get(col_lower, col))
            
            self.data.columns = new_columns
            
            # Validate we have at least Close prices
            if 'Close' not in self.data.columns:
                # Try to identify price column
                price_columns = [col for col in self.data.columns if any(x in col.lower() for x in ['close', 'price', 'value'])]
                if price_columns:
                    self.data = self.data.rename(columns={price_columns[0]: 'Close'})
                    print(f"✅ Renamed '{price_columns[0]}' to 'Close'")
                else:
                    # Use the first numeric column as Close
                    numeric_columns = self.data.select_dtypes(include=[np.number]).columns
                    if len(numeric_columns) > 0:
                        self.data = self.data.rename(columns={numeric_columns[0]: 'Close'})
                        print(f"⚠️ Using '{numeric_columns[0]}' as Close price column")
                    else:
                        raise ValueError("No numeric columns found for price data")
            
            # Create missing OHLC columns if only Close exists
            if 'Open' not in self.data.columns:
                self.data['Open'] = self.data['Close']
                print("⚠️ Created 'Open' column from Close prices")
            if 'High' not in self.data.columns:
                self.data['High'] = self.data['Close'] 
                print("⚠️ Created 'High' column from Close prices")
            if 'Low' not in self.data.columns:
                self.data['Low'] = self.data['Close']
                print("⚠️ Created 'Low' column from Close prices")
            
            # Ensure weekly frequency
            self._ensure_weekly_frequency()
            
            # Validate data quality
            self._validate_data_quality()
            
            print(f"✅ Data loaded successfully!")
            print(f"   Period: {self.data.index.min().date()} to {self.data.index.max().date()}")
            print(f"   Total records: {len(self.data)}")
            print(f"   Columns: {list(self.data.columns)}")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def _ensure_weekly_frequency(self):
        """Ensure data is in weekly frequency (Friday close)"""
        if len(self.data) == 0:
            return
            
        # Check current frequency
        if len(self.data) > 1:
            sample_diff = (self.data.index[1] - self.data.index[0]).days
            
            if sample_diff <= 5:  # Daily or more frequent
                print("📊 Converting to weekly frequency (Friday close)...")
                weekly_data = self.data.resample('W-FRI').agg({
                    'Open': 'first',
                    'High': 'max', 
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum' if 'Volume' in self.data.columns else 'sum'
                }).dropna()
                
                self.data = weekly_data
                print(f"✅ Converted to weekly data: {len(self.data)} records")
            elif sample_diff > 7:  # Less frequent than weekly
                print("📊 Data appears to be less frequent than weekly - using as-is")
    
    def _validate_data_quality(self):
        """Validate data quality and handle issues"""
        required_cols = ['Open', 'High', 'Low', 'Close']
        
        # Check for NaN values
        nan_count = self.data[required_cols].isna().sum().sum()
        if nan_count > 0:
            print(f"⚠️  Found {nan_count} NaN values - filling forward")
            self.data[required_cols] = self.data[required_cols].fillna(method='ffill')
        
        # Check for zero/negative prices
        if (self.data['Close'] <= 0).any():
            print("⚠️  Found zero/negative prices - removing invalid records")
            self.data = self.data[self.data['Close'] > 0]
        
        # Check data length
        if len(self.data) < 260:  # Less than 5 years
            print("⚠️  Warning: Limited historical data (less than 5 years)")
        else:
            years = (self.data.index.max() - self.data.index.min()).days / 365.25
            print(f"📅 Data covers {years:.1f} years")
        
        print(f"✅ Data quality check passed: {len(self.data)} clean records")

# Initialize and load your Kaggle dataset
print("🔄 Loading XAUUSD data from Kaggle Hub...")
data_loader = GoldDataLoader()
gold_data = data_loader.load_kaggle_dataset("asiluxresearch/xauusd-dalyop")

if gold_data is not None:
    print(f"🎉 Successfully loaded gold data with {len(gold_data)} records")
    print(gold_data.head())
else:
    print("❌ Failed to load gold data. Please check the dataset path or structure.")
    
    # Cell: Check and Fix Data Orientation
print("🔍 CHECKING DATA ORIENTATION...")

# Check if we have USDXAU instead of XAUUSD
def check_and_fix_data_orientation(data):
    """Check if data is USDXAU (inverse) and convert to XAUUSD if needed"""
    
    # Check typical gold price ranges
    current_price = data['Close'].iloc[-1]
    avg_price = data['Close'].mean()
    
    print(f"   Current Price: ${current_price:,.2f}")
    print(f"   Average Price: ${avg_price:,.2f}")
    
    # Gold typically trades between $1,000 - $2,500 in recent years
    # If prices are very small (like 0.0004), it's likely USDXAU
    if avg_price < 0.01:  # Very small numbers suggest USDXAU
        print("⚠️  Detected USDXAU format (inverse prices) - converting to XAUUSD...")
        
        # Invert all price columns
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                data[col] = 1 / data[col]
        
        print(f"✅ Converted to XAUUSD format")
        print(f"   New Current Price: ${data['Close'].iloc[-1]:,.2f}")
        print(f"   New Average Price: ${data['Close'].mean():,.2f}")
    
    elif avg_price < 100:  # Suspiciously low for gold
        print("⚠️  Prices seem unusually low for gold - may be inverse data")
        response = input("   Convert to XAUUSD? (y/n): ")
        if response.lower() == 'y':
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in data.columns:
                    data[col] = 1 / data[col]
            print("✅ Converted to XAUUSD format")
    
    else:
        print("✅ Data appears to be in XAUUSD format (normal gold prices)")
    
    return data

# Apply the fix
gold_data = check_and_fix_data_orientation(gold_data)

print(f"\n📊 DATA SUMMARY AFTER ORIENTATION CHECK:")
print(f"   First 5 prices: {[f'${x:,.2f}' for x in gold_data['Close'].head().values]}")
print(f"   Last 5 prices:  {[f'${x:,.2f}' for x in gold_data['Close'].tail().values]}")
print(f"   Price range: ${gold_data['Close'].min():,.2f} - ${gold_data['Close'].max():,.2f}")

# If your data is USDXAU, run this cell to convert to XAUUSD
def convert_usdxau_to_xauusd(data):
    """Convert USDXAU data to XAUUSD format"""
    print("🔄 Converting USDXAU to XAUUSD...")
    
    original_prices = data['Close'].copy()
    
    # Invert all price columns
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in data.columns:
            data[col] = 1 / data[col]
    
    print(f"✅ Conversion complete:")
    print(f"   Before: ${original_prices.iloc[-1]:.6f} -> After: ${data['Close'].iloc[-1]:,.2f}")
    print(f"   Example: 0.0005 USDXAU = ${1/0.0005:,.0f} XAUUSD")
    
    return data

# Uncomment and run if you need to convert:
# gold_data = convert_usdxau_to_xauusd(gold_data)

# After fixing data orientation, re-initialize everything
print("🔄 Re-initializing forecasters with corrected data...")

# Re-initialize all components with the corrected data
years_forecaster = SimilarYearsForecaster(gold_data)
smc_analyzer = SMCPriceActionAnalyzer(gold_data) 
ml_enhancer = MLForecastEnhancer(gold_data)
comprehensive_forecaster = ComprehensiveGoldForecaster(years_forecaster, smc_analyzer, ml_enhancer)

# Show corrected data summary
print(f"\n📊 CORRECTED DATA SUMMARY:")
print(f"   Time period: {gold_data.index[0].strftime('%Y-%m-%d')} to {gold_data.index[-1].strftime('%Y-%m-%d')}")
print(f"   Current gold price: ${gold_data['Close'].iloc[-1]:,.2f}")
print(f"   All-time high: ${gold_data['Close'].max():,.2f}")
print(f"   Recent trend: {'📈 Bullish' if gold_data['Close'].iloc[-1] > gold_data['Close'].iloc[-52] else '📉 Bearish'}")

# Regenerate forecast with corrected data
print("\n🚀 GENERATING NEW FORECAST WITH CORRECTED DATA...")
final_forecast_corrected = generate_and_visualize_forecast()

@dataclass
class SimilarYear:
    year: int
    similarity_score: float
    next_year_return: float
    next_year_data: pd.DataFrame
    detail_scores: Dict

class SimilarYearsForecaster:
    def __init__(self, price_data: pd.DataFrame, min_years_history: int = 10):
        self.data = price_data
        self.min_years = min_years_history
        self.yearly_windows = None
        self.complete_years = None
        
        # Similarity weights (tuned for gold)
        self.similarity_weights = {
            'price_correlation': 0.25,
            'volatility_similarity': 0.20, 
            'trend_similarity': 0.15,
            'seasonal_pattern': 0.15,
            'drawdown_similarity': 0.10,
            'momentum_profile': 0.15
        }
    
    def detect_complete_years(self) -> List[int]:
        """Automatically detect which years have complete data"""
        complete_years = []
        
        for year in range(self.data.index.year.min(), self.data.index.year.max() + 1):
            year_data = self.data[self.data.index.year == year]
            
            # Check if year has at least 48 weeks of data (considered complete)
            if len(year_data) >= 48:
                complete_years.append(year)
        
        print(f"📅 Complete years detected: {complete_years}")
        self.complete_years = complete_years
        return complete_years
    
    def get_last_complete_year(self) -> int:
        """Get the last year with complete data"""
        if self.complete_years is None:
            self.detect_complete_years()
        
        if not self.complete_years:
            raise ValueError("No complete years found in dataset")
        
        return max(self.complete_years)
    
    def extract_calendar_years(self) -> Dict[int, pd.DataFrame]:
        """Extract complete calendar years from the data"""
        if self.complete_years is None:
            self.detect_complete_years()
            
        yearly_data = {}
        
        for year in self.complete_years:
            year_data = self.data[self.data.index.year == year]
            
            # Pad to 52 weeks if necessary
            if len(year_data) < 52:
                year_start = datetime(year, 1, 1)
                year_end = datetime(year, 12, 31)
                full_index = pd.date_range(start=year_start, end=year_end, freq='W-FRI')
                year_data = year_data.reindex(full_index)
                # Forward fill missing values
                year_data = year_data.fillna(method='ffill')
            
            # Take exactly 52 weeks
            yearly_data[year] = year_data.head(52)
        
        print(f"📅 Extracted {len(yearly_data)} complete years: {list(yearly_data.keys())}")
        self.yearly_windows = yearly_data
        return yearly_data
    
    def calculate_comprehensive_similarity(self, current_year: int, historical_year: int) -> Optional[Tuple[float, Dict]]:
        """Calculate multi-dimensional similarity between two years"""
        if self.yearly_windows is None:
            self.extract_calendar_years()
            
        if current_year not in self.yearly_windows or historical_year not in self.yearly_windows:
            return None
        
        current_data = self.yearly_windows[current_year]
        historical_data = self.yearly_windows[historical_year]
        
        # Ensure we have valid data
        if len(current_data) < 52 or len(historical_data) < 52:
            return None
        
        current_prices = current_data['Close'].values
        historical_prices = historical_data['Close'].values
        
        similarity_scores = {}
        
        try:
            # 1. Price Correlation
            current_returns = np.diff(np.log(current_prices))
            historical_returns = np.diff(np.log(historical_prices))
            correlation = np.corrcoef(current_returns, historical_returns)[0, 1]
            similarity_scores['price_correlation'] = max(0, (correlation + 1) / 2)
            
            # 2. Volatility Similarity  
            current_vol = np.std(current_returns)
            historical_vol = np.std(historical_returns)
            if current_vol > 0 and historical_vol > 0:
                vol_ratio = min(current_vol, historical_vol) / max(current_vol, historical_vol)
                similarity_scores['volatility_similarity'] = vol_ratio
            else:
                similarity_scores['volatility_similarity'] = 0
            
            # 3. Trend Similarity
            current_trend = self._calculate_trend_strength(current_prices)
            historical_trend = self._calculate_trend_strength(historical_prices)
            trend_similarity = 1 - abs(current_trend - historical_trend) / (abs(current_trend) + abs(historical_trend) + 1e-8)
            similarity_scores['trend_similarity'] = max(0, trend_similarity)
            
            # 4. Seasonal Pattern
            seasonal_sim = self._calculate_seasonal_similarity(current_prices, historical_prices)
            similarity_scores['seasonal_pattern'] = seasonal_sim
            
            # 5. Drawdown Similarity
            current_dd = self._calculate_drawdown_profile(current_prices)
            historical_dd = self._calculate_drawdown_profile(historical_prices)
            if current_dd > 0 or historical_dd > 0:
                drawdown_sim = 1 - abs(current_dd - historical_dd) / (abs(current_dd) + abs(historical_dd) + 1e-8)
                similarity_scores['drawdown_similarity'] = max(0, drawdown_sim)
            else:
                similarity_scores['drawdown_similarity'] = 1.0
            
            # 6. Momentum Profile
            momentum_sim = self._calculate_momentum_similarity(current_prices, historical_prices)
            similarity_scores['momentum_profile'] = momentum_sim
            
            # Composite weighted score
            composite_score = 0
            for metric, weight in self.similarity_weights.items():
                score = similarity_scores.get(metric, 0)
                if not np.isnan(score):
                    composite_score += score * weight
            
            return composite_score, similarity_scores
            
        except Exception as e:
            print(f"⚠️ Similarity calculation error for {historical_year}: {e}")
            return None
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate normalized trend strength"""
        if len(prices) < 2:
            return 0
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        return slope / prices[0]  # Normalize by initial price
    
    def _calculate_seasonal_similarity(self, prices1: np.ndarray, prices2: np.ndarray) -> float:
        """Compare quarterly performance patterns"""
        try:
            # Quarterly returns (approximate quarters)
            q_points = [0, 13, 26, 39, 51]  # Start of each quarter
            quarters1 = []
            quarters2 = []
            
            for i in range(4):
                start_idx = q_points[i]
                end_idx = q_points[i+1] if i < 3 else len(prices1)-1
                
                if end_idx < len(prices1) and start_idx < len(prices1):
                    q1_return = (prices1[end_idx] / prices1[start_idx]) - 1
                    q2_return = (prices2[end_idx] / prices2[start_idx]) - 1
                    quarters1.append(q1_return)
                    quarters2.append(q2_return)
            
            if len(quarters1) == 4 and len(quarters2) == 4:
                distance = np.linalg.norm(np.array(quarters1) - np.array(quarters2))
                max_distance = 2.0  # Reasonable maximum
                similarity = 1 - (distance / max_distance)
                return max(0, min(1, similarity))
            else:
                return 0.5  # Neutral if we can't calculate
                
        except:
            return 0.5
    
    def _calculate_drawdown_profile(self, prices: np.ndarray) -> float:
        """Calculate combined drawdown characteristics"""
        try:
            peak = np.maximum.accumulate(prices)
            drawdowns = (peak - prices) / peak
            max_dd = np.max(drawdowns)
            return max_dd
        except:
            return 0
    
    def _calculate_momentum_similarity(self, prices1: np.ndarray, prices2: np.ndarray) -> float:
        """Compare momentum characteristics using cosine similarity"""
        try:
            # Multiple momentum periods
            periods = [4, 13, 26]
            momentums1 = []
            momentums2 = []
            
            for p in periods:
                if len(prices1) > p and len(prices2) > p:
                    mom1 = (prices1[-1] / prices1[-p]) - 1
                    mom2 = (prices2[-1] / prices2[-p]) - 1
                    momentums1.append(mom1)
                    momentums2.append(mom2)
            
            if momentums1 and momentums2:
                # Cosine similarity
                dot_product = np.dot(momentums1, momentums2)
                norm1 = np.linalg.norm(momentums1)
                norm2 = np.linalg.norm(momentums2)
                
                if norm1 > 0 and norm2 > 0:
                    return max(0, dot_product / (norm1 * norm2))
            
            return 0.5  # Neutral default
            
        except:
            return 0.5
    
    def find_similar_years(self, current_year: int = None, top_n: int = 5, min_similarity: float = 0.6) -> List[SimilarYear]:
        """Find most similar historical years to current year"""
        if self.yearly_windows is None:
            self.extract_calendar_years()
        
        # If no current_year provided, use the last complete year
        if current_year is None:
            current_year = self.get_last_complete_year()
            print(f"🔍 Using last complete year for analysis: {current_year}")
        
        if current_year not in self.yearly_windows:
            # Try to find the closest available year
            available_years = list(self.yearly_windows.keys())
            if available_years:
                closest_year = max(available_years)
                print(f"⚠️  Year {current_year} not available, using closest complete year: {closest_year}")
                current_year = closest_year
            else:
                raise ValueError("No complete years available for analysis")
        
        similar_years = []
        
        print(f"🔍 Searching for years similar to {current_year}...")
        
        for historical_year in self.yearly_windows:
            if historical_year >= current_year:  # Only past years
                continue
            
            result = self.calculate_comprehensive_similarity(current_year, historical_year)
            if result is not None:
                similarity_score, detail_scores = result
                
                if similarity_score >= min_similarity:
                    # Get +1 year performance
                    next_year = historical_year + 1
                    if next_year in self.yearly_windows:
                        next_year_data = self.yearly_windows[next_year]
                        if len(next_year_data) > 0:
                            next_year_return = (next_year_data['Close'].iloc[-1] / next_year_data['Close'].iloc[0]) - 1
                            
                            similar_year = SimilarYear(
                                year=historical_year,
                                similarity_score=similarity_score,
                                next_year_return=next_year_return,
                                next_year_data=next_year_data,
                                detail_scores=detail_scores
                            )
                            similar_years.append(similar_year)
        
        # Sort by similarity score
        similar_years.sort(key=lambda x: x.similarity_score, reverse=True)
        
        print(f"✅ Found {len(similar_years)} similar years for {current_year}")
        return similar_years[:top_n]
    
    def generate_similar_years_forecast(self, current_year: int = None) -> Dict:
        """Generate forecast based on similar years +1 year performance"""
        similar_years = self.find_similar_years(current_year)
        
        if not similar_years:
            current_year = current_year or self.get_last_complete_year()
            return {"error": f"No similar years found for {current_year} (min similarity: 0.6)"}
        
        # Weighted average based on similarity scores
        weights = np.array([year.similarity_score for year in similar_years])
        weights = weights / weights.sum()
        
        next_year_returns = np.array([year.next_year_return for year in similar_years])
        weighted_return = np.average(next_year_returns, weights=weights)
        
        # Calculate confidence intervals
        return_std = np.std(next_year_returns)
        confidence_95 = [weighted_return - 1.96 * return_std, weighted_return + 1.96 * return_std]
        confidence_80 = [weighted_return - 1.28 * return_std, weighted_return + 1.28 * return_std]
        
        # Bullish probability
        bullish_prob = np.mean(next_year_returns > 0)
        
        # Determine forecast year (current_year + 1)
        forecast_year = (current_year or self.get_last_complete_year()) + 1
        
        forecast = {
            'current_year': current_year or self.get_last_complete_year(),
            'forecast_year': forecast_year,
            'similar_years_count': len(similar_years),
            'expected_return': weighted_return,
            'confidence_95': confidence_95,
            'confidence_80': confidence_80,
            'bullish_probability': bullish_prob,
            'similar_years_details': [
                {
                    'year': year.year,
                    'similarity_score': year.similarity_score,
                    'next_year_return': year.next_year_return,
                    'detail_scores': year.detail_scores
                } for year in similar_years
            ],
            'historical_returns_range': [min(next_year_returns), max(next_year_returns)],
            'weights_used': weights.tolist()
        }
        
        return forecast

# Initialize the similar years forecaster
print("🔄 Initializing Similar Years Forecaster...")
years_forecaster = SimilarYearsForecaster(gold_data)

# Show available years
complete_years = years_forecaster.detect_complete_years()
last_complete_year = years_forecaster.get_last_complete_year()
print(f"📅 Last complete year available: {last_complete_year}")
print(f"🎯 We will forecast for: {last_complete_year + 1}")

class SMCPriceActionAnalyzer:
    def __init__(self, price_data: pd.DataFrame):
        self.data = price_data
    
    def identify_smc_levels(self, window_weeks: int = 52) -> Dict:
        """Identify Smart Money Concepts levels from price action"""
        latest_data = self.data.tail(window_weeks).copy()
        
        smc_signals = {}
        
        # 1. Order Blocks (significant reversal zones)
        smc_signals['order_blocks'] = self._find_order_blocks(latest_data)
        
        # 2. Liquidity Levels (recent highs/lows)
        smc_signals['liquidity_levels'] = self._find_liquidity_levels(latest_data)
        
        # 3. Market Structure
        smc_signals['market_structure'] = self._analyze_market_structure(latest_data)
        
        # 4. Key Support/Resistance
        smc_signals['key_levels'] = self._find_key_support_resistance(latest_data)
        
        return smc_signals
    
    def _find_order_blocks(self, data: pd.DataFrame) -> List[Dict]:
        """Identify order blocks (large candles with rejection)"""
        order_blocks = []
        
        for i in range(2, len(data)-2):
            if i >= len(data):
                continue
                
            current = data.iloc[i]
            prev = data.iloc[i-1]
            next_candle = data.iloc[i+1]
            
            # Large candle criteria (2% or more)
            candle_size = abs(current['Close'] - current['Open']) / current['Open']
            if candle_size > 0.02:
                # Reversal confirmation
                bullish_reversal = (current['Close'] > current['Open'] and 
                                  next_candle['Close'] < current['Close'])
                bearish_reversal = (current['Close'] < current['Open'] and 
                                  next_candle['Close'] > current['Close'])
                
                if bullish_reversal or bearish_reversal:
                    order_blocks.append({
                        'date': data.index[i],
                        'type': 'bullish' if bullish_reversal else 'bearish',
                        'high': current['High'],
                        'low': current['Low'],
                        'strength': candle_size
                    })
        
        return order_blocks[-3:]  # Last 3 order blocks
    
    def _find_liquidity_levels(self, data: pd.DataFrame) -> Dict:
        """Identify recent liquidity levels (highs and lows)"""
        # Recent swing highs and lows
        window = min(20, len(data))
        
        recent_highs = data['High'].rolling(window, center=True).max()
        significant_highs = data[data['High'] == recent_highs]
        
        recent_lows = data['Low'].rolling(window, center=True).min()
        significant_lows = data[data['Low'] == recent_lows]
        
        return {
            'resistance': significant_highs['High'].tail(3).tolist(),
            'support': significant_lows['Low'].tail(3).tolist()
        }
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> str:
        """Analyze market structure (uptrend, downtrend, ranging)"""
        if len(data) < 10:
            return "insufficient_data"
        
        # Simple trend analysis using moving averages
        sma_20 = data['Close'].rolling(20).mean()
        sma_50 = data['Close'].rolling(50).mean()
        
        current_close = data['Close'].iloc[-1]
        
        if current_close > sma_20.iloc[-1] > sma_50.iloc[-1]:
            return "uptrend"
        elif current_close < sma_20.iloc[-1] < sma_50.iloc[-1]:
            return "downtrend"
        else:
            return "ranging"
    
    def _find_key_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Identify key support and resistance levels"""
        # Use recent highs and lows as key levels
        recent_high = data['High'].tail(26).max()
        recent_low = data['Low'].tail(26).min()
        
        # Psychological levels (round numbers)
        current_price = data['Close'].iloc[-1]
        psychological_levels = []
        
        for level in [1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]:
            if abs(level - current_price) / current_price < 0.1:  # Within 10%
                psychological_levels.append(level)
        
        return {
            'recent_high': recent_high,
            'recent_low': recent_low,
            'psychological_levels': psychological_levels[:3]  # Top 3 closest
        }
    
    def generate_smc_forecast_adjustment(self, similar_years_forecast: Dict) -> Dict:
        """Generate SMC-based adjustment to similar years forecast"""
        smc_levels = self.identify_smc_levels()
        current_price = self.data['Close'].iloc[-1]
        
        expected_return = similar_years_forecast.get('expected_return', 0)
        
        # Adjustment factors based on SMC analysis
        adjustment = 1.0
        reasoning = []
        
        # 1. Market structure adjustment
        structure = smc_levels['market_structure']
        if structure == 'uptrend':
            adjustment *= 1.1
            reasoning.append("Uptrend structure - slightly bullish adjustment")
        elif structure == 'downtrend':
            adjustment *= 0.9
            reasoning.append("Downtrend structure - slightly bearish adjustment")
        
        # 2. Proximity to key levels
        resistance_distance = min([abs(level - current_price) / current_price 
                                 for level in smc_levels['liquidity_levels']['resistance']], default=1.0)
        support_distance = min([abs(level - current_price) / current_price 
                              for level in smc_levels['liquidity_levels']['support']], default=1.0)
        
        if resistance_distance < 0.02:  # Within 2% of resistance
            adjustment *= 0.85
            reasoning.append("Near resistance - reducing bullish expectation")
        elif support_distance < 0.02:  # Within 2% of support
            adjustment *= 1.15
            reasoning.append("Near support - increasing bullish expectation")
        
        # 3. Recent order blocks
        recent_bullish_blocks = len([b for b in smc_levels['order_blocks'] if b['type'] == 'bullish'])
        recent_bearish_blocks = len([b for b in smc_levels['order_blocks'] if b['type'] == 'bearish'])
        
        if recent_bullish_blocks > recent_bearish_blocks:
            adjustment *= 1.05
            reasoning.append("More recent bullish order blocks")
        elif recent_bearish_blocks > recent_bullish_blocks:
            adjustment *= 0.95
            reasoning.append("More recent bearish order blocks")
        
        adjusted_forecast = {
            'original_expected_return': expected_return,
            'adjusted_expected_return': expected_return * adjustment,
            'adjustment_factor': adjustment,
            'smc_analysis': smc_levels,
            'adjustment_reasoning': reasoning,
            'current_price': current_price
        }
        
        return adjusted_forecast

# Initialize SMC analyzer
print("🔄 Initializing SMC & Price Action Analyzer...")
smc_analyzer = SMCPriceActionAnalyzer(gold_data)

class MLForecastEnhancer:
    def __init__(self, price_data: pd.DataFrame):
        self.data = price_data
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def create_technical_features(self) -> pd.DataFrame:
        """Create comprehensive technical features from price data (no external TA libraries)"""
        df = self.data.copy()
        
        # Price-based features
        df['returns_1w'] = df['Close'].pct_change(1)
        df['returns_4w'] = df['Close'].pct_change(4)
        df['returns_13w'] = df['Close'].pct_change(13)
        df['returns_52w'] = df['Close'].pct_change(52)
        
        # Volatility features (using basic rolling std)
        df['volatility_4w'] = df['returns_1w'].rolling(4).std()
        df['volatility_13w'] = df['returns_1w'].rolling(13).std()
        df['volatility_52w'] = df['returns_1w'].rolling(52).std()
        
        # Momentum features
        df['momentum_4w'] = df['Close'] / df['Close'].shift(4) - 1
        df['momentum_13w'] = df['Close'] / df['Close'].shift(13) - 1
        df['momentum_52w'] = df['Close'] / df['Close'].shift(52) - 1
        
        # Moving averages (basic pandas)
        df['sma_13w'] = df['Close'].rolling(13).mean()
        df['sma_52w'] = df['Close'].rolling(52).mean()
        df['ema_13w'] = df['Close'].ewm(span=13).mean()
        
        # Relative position
        df['position_vs_13w'] = (df['Close'] / df['sma_13w']) - 1
        df['position_vs_52w'] = (df['Close'] / df['sma_52w']) - 1
        
        # Support/Resistance (basic calculations)
        df['resistance_13w'] = df['High'].rolling(13).max()
        df['support_13w'] = df['Low'].rolling(13).min()
        
        # Avoid division by zero
        level_range = df['resistance_13w'] - df['support_13w']
        df['support_resistance_position'] = np.where(
            level_range > 0, 
            (df['Close'] - df['support_13w']) / level_range, 
            0.5
        )
        
        # Target: Next 52-week return
        df['target_52w_return'] = df['Close'].shift(-52) / df['Close'] - 1
        
        # Drop NaN values
        df_clean = df.dropna()
        
        return df_clean
    
    def prepare_ml_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for ML model"""
        feature_df = self.create_technical_features()
        
        # Select feature columns
        self.feature_columns = [
            'returns_1w', 'returns_4w', 'returns_13w', 'returns_52w',
            'volatility_4w', 'volatility_13w', 'volatility_52w',
            'momentum_4w', 'momentum_13w', 'momentum_52w',
            'position_vs_13w', 'position_vs_52w',
            'support_resistance_position'
        ]
        
        X = feature_df[self.feature_columns]
        y = feature_df['target_52w_return']
        
        return X, y
    
    def train_model(self, test_size: float = 0.2) -> Dict:
        """Train machine learning model for forecasting"""
        X, y = self.prepare_ml_data()
        
        # Time-series split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0)
        }
        
        best_model = None
        best_score = -np.inf
        results = {}
        
        for name, model in models.items():
            try:
                if name == 'Ridge':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                results[name] = {'r2': r2, 'mae': mae}
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    
                print(f"   {name}: R² = {r2:.4f}, MAE = {mae:.4f}")
                
            except Exception as e:
                print(f"   {name}: Error - {e}")
        
        self.model = best_model
        print(f"✅ Best model: {list(models.keys())[list(results.values()).index(max(results.values(), key=lambda x: x['r2']))]}")
        
        return results
    
    def generate_ml_forecast(self, similar_years_forecast: Dict, smc_forecast: Dict) -> Dict:
        """Generate ML-enhanced forecast"""
        if self.model is None:
            print("🔄 Training ML model...")
            self.train_model()
        
        # Get latest features
        feature_df = self.create_technical_features()
        latest_features = feature_df[self.feature_columns].iloc[-1:].copy()
        
        # ML prediction
        if isinstance(self.model, Ridge):
            latest_features_scaled = self.scaler.transform(latest_features)
            ml_prediction = self.model.predict(latest_features_scaled)[0]
        else:
            ml_prediction = self.model.predict(latest_features)[0]
        
        # Get component forecasts
        similar_return = similar_years_forecast.get('expected_return', 0)
        smc_adjusted_return = smc_forecast.get('adjusted_expected_return', 0)
        
        # Ensemble weighting
        weights = {
            'similar_years': 0.4,
            'smc_adjusted': 0.3, 
            'ml_prediction': 0.3
        }
        
        final_return = (
            weights['similar_years'] * similar_return +
            weights['smc_adjusted'] * smc_adjusted_return +
            weights['ml_prediction'] * ml_prediction
        )
        
        ml_enhanced_forecast = {
            'final_expected_return': final_return,
            'component_forecasts': {
                'similar_years': similar_return,
                'smc_adjusted': smc_adjusted_return, 
                'ml_prediction': ml_prediction
            },
            'weights_used': weights,
            'ml_model_type': type(self.model).__name__
        }
        
        return ml_enhanced_forecast

# Initialize ML enhancer
print("🔄 Initializing ML Forecast Enhancer...")
ml_enhancer = MLForecastEnhancer(gold_data)

class ComprehensiveGoldForecaster:
    def __init__(self, years_forecaster, smc_analyzer, ml_enhancer):
        self.years_forecaster = years_forecaster
        self.smc_analyzer = smc_analyzer
        self.ml_enhancer = ml_enhancer
        self.forecast_history = []
    
    def generate_comprehensive_forecast(self, current_year: int = None) -> Dict:
        """Generate comprehensive forecast using all three methodologies"""
        # Auto-detect the best year to use
        if current_year is None:
            current_year = self.years_forecaster.get_last_complete_year()
            print(f"🎯 Auto-selected analysis year: {current_year}")
        
        forecast_year = current_year + 1
        
        print(f"🎯 Generating Comprehensive Forecast")
        print(f"   Analysis Year: {current_year}")
        print(f"   Forecast Year: {forecast_year}")
        print("=" * 50)
        
        # 1. Similar Years Forecast
        print("1. Similar Years Analysis...")
        similar_years_forecast = self.years_forecaster.generate_similar_years_forecast(current_year)
        
        if 'error' in similar_years_forecast:
            return {"error": similar_years_forecast['error']}
        
        # 2. SMC Adjustment
        print("2. SMC & Price Action Analysis...")
        smc_forecast = self.smc_analyzer.generate_smc_forecast_adjustment(similar_years_forecast)
        
        # 3. ML Enhancement
        print("3. Machine Learning Enhancement...")
        final_forecast = self.ml_enhancer.generate_ml_forecast(similar_years_forecast, smc_forecast)
        
        # Combine results
        comprehensive_result = {
            'current_year': current_year,
            'forecast_year': forecast_year,
            'current_price': self.smc_analyzer.data['Close'].iloc[-1],
            'forecast_date': datetime.now().strftime("%Y-%m-%d"),
            'forecast_horizon': f'1 year ({forecast_year})',
            'data_status': self._get_data_status(),
            'final_forecast': final_forecast,
            'component_analysis': {
                'similar_years': similar_years_forecast,
                'smc_adjusted': smc_forecast
            }
        }
        
        # Store in history
        self.forecast_history.append(comprehensive_result)
        
        return comprehensive_result
    
    def _get_data_status(self) -> str:
        """Get status of current data"""
        latest_date = self.smc_analyzer.data.index[-1]
        current_year = latest_date.year
        current_week = latest_date.isocalendar()[1]
        
        if current_week >= 48:
            return f"Complete data through {latest_date.strftime('%Y-%m-%d')}"
        else:
            return f"Partial year {current_year} data ({current_week}/52 weeks)"
    
    def print_forecast_report(self, comprehensive_forecast: Dict):
        """Print detailed forecast report"""
        if 'error' in comprehensive_forecast:
            print(f"❌ Error: {comprehensive_forecast['error']}")
            return
        
        fc = comprehensive_forecast
        final = fc['final_forecast']
        current_price = fc['current_price']
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE GOLD FORECAST REPORT")
        print("="*80)
        print(f"Forecast Date: {fc['forecast_date']}")
        print(f"Analysis Year: {fc['current_year']} (complete data)")
        print(f"Forecast Year: {fc['forecast_year']}")
        print(f"Current Gold Price: ${current_price:,.2f}")
        print(f"Data Status: {fc['data_status']}")
        
        # Final Forecast
        expected_return = final['final_expected_return']
        expected_price = current_price * (1 + expected_return)
        price_change = expected_price - current_price
        
        print(f"\n📊 FINAL FORECAST:")
        print(f"Expected Return: {expected_return:+.2%}")
        print(f"Expected Price: ${expected_price:,.2f}")
        print(f"Price Change: ${price_change:+,.2f}")
        
        # Component Breakdown
        print(f"\n🔧 COMPONENT BREAKDOWN:")
        components = final['component_forecasts']
        print(f"  Similar Years: {components['similar_years']:+.2%}")
        print(f"  SMC Adjusted:  {components['smc_adjusted']:+.2%}")
        print(f"  ML Prediction: {components['ml_prediction']:+.2%}")
        
        # Weights
        print(f"\n⚖️  METHODOLOGY WEIGHTS:")
        weights = final['weights_used']
        for method, weight in weights.items():
            print(f"  {method.replace('_', ' ').title():<15} {weight:.0%}")
        
        # Similar Years Details
        similar_years = fc['component_analysis']['similar_years']
        print(f"\n📅 TOP SIMILAR YEARS FOUND:")
        for i, year in enumerate(similar_years['similar_years_details'][:3]):
            print(f"  {i+1}. {year['year']} "
                  f"(Similarity: {year['similarity_score']:.3f}, "
                  f"Next Year Return: {year['next_year_return']:+.2%})")
        
        # SMC Analysis
        smc = fc['component_analysis']['smc_adjusted']
        print(f"\n🎯 SMC ANALYSIS:")
        print(f"  Market Structure: {smc['smc_analysis']['market_structure'].upper()}")
        print(f"  Adjustment Factor: {smc['adjustment_factor']:.2f}x")
        if smc['adjustment_reasoning']:
            print(f"  Key Factors: {', '.join(smc['adjustment_reasoning'])}")
        
        # ML Model Info
        print(f"\n🤖 ML ENHANCEMENT:")
        print(f"  Model Type: {final['ml_model_type']}")
        
        # Confidence Assessment
        similar_years_data = fc['component_analysis']['similar_years']
        confidence_level = similar_years_data['similar_years_count']
        if confidence_level >= 4:
            confidence = "HIGH"
            emoji = "💎"
        elif confidence_level >= 2:
            confidence = "MEDIUM" 
            emoji = "📊"
        else:
            confidence = "LOW"
            emoji = "⚠️"
            
        print(f"\n{emoji} CONFIDENCE: {confidence}")
        print(f"   Based on {similar_years_data['similar_years_count']} similar years found")
        print(f"   Bullish Probability: {similar_years_data['bullish_probability']:.1%}")

# Initialize comprehensive forecaster
print("🔄 Initializing Comprehensive Gold Forecaster...")
comprehensive_forecaster = ComprehensiveGoldForecaster(years_forecaster, smc_analyzer, ml_enhancer)

def generate_and_visualize_forecast(target_year: int = None):
    """Generate forecast and create comprehensive visualizations"""
    
    print("🚀 GENERATING COMPREHENSIVE GOLD FORECAST")
    print("Methodology: Similar Years + SMC/Price Action + Machine Learning")
    print("=" * 60)
    
    # Let the system automatically handle year selection
    comprehensive_forecast = comprehensive_forecaster.generate_comprehensive_forecast(target_year)
    
    # Display detailed report
    comprehensive_forecaster.print_forecast_report(comprehensive_forecast)
    
    # Create visualizations
    if 'error' not in comprehensive_forecast:
        create_forecast_visualizations(comprehensive_forecast)
    
    return comprehensive_forecast

def create_forecast_visualizations(forecast: Dict):
    """Create comprehensive visualizations for the forecast"""
    
    # 1. Price Projection Chart
    fig1 = go.Figure()
    
    current_price = forecast['current_price']
    expected_return = forecast['final_forecast']['final_expected_return']
    expected_price = current_price * (1 + expected_return)
    
    # Historical price data (last 2 years)
    historical_data = gold_data.tail(104)  # 2 years of weekly data
    
    # Add historical price
    fig1.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#FFD700', width=2)
    ))
    
    # Add forecast point
    forecast_date = historical_data.index[-1] + pd.DateOffset(years=1)
    fig1.add_trace(go.Scatter(
        x=[historical_data.index[-1], forecast_date],
        y=[current_price, expected_price],
        mode='lines+markers',
        name=f'Forecast for {forecast["forecast_year"]}',
        line=dict(color='#00FF00', width=3, dash='dash'),
        marker=dict(size=10)
    ))
    
    # Add current price annotation
    fig1.add_annotation(
        x=historical_data.index[-1],
        y=current_price,
        text=f"Current: ${current_price:,.2f}",
        showarrow=True,
        arrowhead=2
    )
    
    # Add forecast annotation
    fig1.add_annotation(
        x=forecast_date,
        y=expected_price,
        text=f"Forecast: ${expected_price:,.2f}",
        showarrow=True,
        arrowhead=2
    )
    
    fig1.update_layout(
        title=f'Gold Price Forecast: Historical vs {forecast["forecast_year"]} Projection',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        showlegend=True
    )
    
    # 2. Component Breakdown Chart
    components = forecast['final_forecast']['component_forecasts']
    weights = forecast['final_forecast']['weights_used']
    
    fig2 = go.Figure()
    
    methods = list(components.keys())
    returns = [components[method] for method in methods]
    weight_values = [weights[method] for method in methods]
    
    # Bar chart for component returns
    fig2.add_trace(go.Bar(
        x=[m.replace('_', ' ').title() for m in methods],
        y=returns,
        name='Component Returns',
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        text=[f'{r:+.2%}' for r in returns],
        textposition='auto'
    ))
    
    fig2.update_layout(
        title='Forecast Component Breakdown',
        xaxis_title='Methodology',
        yaxis_title='Expected Return',
        template='plotly_dark',
        yaxis_tickformat='.1%'
    )
    
    # 3. Show all plots
    print("\n📊 VISUALIZATIONS:")
    fig1.show()
    fig2.show()
    
    return fig1, fig2

# Generate the forecast with automatic year handling
print("🎯 Starting automatic forecast generation...")
final_forecast = generate_and_visualize_forecast()  # No year specified - auto-detect

# Show what years we're working with
print(f"\n📅 ANALYSIS SUMMARY:")
print(f"   Last data point: {gold_data.index[-1].strftime('%Y-%m-%d')}")
print(f"   Analysis year: {final_forecast.get('current_year', 'N/A')}")
print(f"   Forecast year: {final_forecast.get('forecast_year', 'N/A')}")

def create_similar_years_comparison(forecast: Dict):
    """Create detailed comparison with similar years"""
    
    similar_years_data = forecast['component_analysis']['similar_years']
    
    if similar_years_data['similar_years_count'] == 0:
        print("No similar years to visualize")
        return
    
    # Get current analysis year data (not forecast year!)
    current_year = forecast['current_year']  # This is the year we analyzed (e.g., 2024)
    
    # Check if current_year exists in yearly_windows
    if current_year not in years_forecaster.yearly_windows:
        print(f"⚠️  Data for analysis year {current_year} not available for visualization")
        return
        
    current_data = years_forecaster.yearly_windows[current_year]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'Price Comparison: {current_year} vs Similar Years',
            'Similarity Scores & Weights',
            'Quarterly Performance Patterns',
            'Next Year Returns Distribution'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Price comparison
    weeks = list(range(52))
    fig.add_trace(
        go.Scatter(x=weeks, y=current_data['Close'].values, 
                  name=f'Analysis Year ({current_year})', line=dict(width=4)),
        row=1, col=1
    )
    
    # Add similar years
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for i, year_data in enumerate(similar_years_data['similar_years_details'][:3]):
        year = year_data['year']
        if year in years_forecaster.yearly_windows:
            year_prices = years_forecaster.yearly_windows[year]['Close'].values
            fig.add_trace(
                go.Scatter(x=weeks, y=year_prices, 
                          name=f'Similar Year {year}', line=dict(color=colors[i])),
                row=1, col=1
            )
    
    # 2. Similarity scores and weights
    years = [f"{yd['year']}" for yd in similar_years_data['similar_years_details']]
    similarities = [yd['similarity_score'] for yd in similar_years_data['similar_years_details']]
    weights = similar_years_data['weights_used']
    
    fig.add_trace(
        go.Bar(x=years, y=similarities, name='Similarity Score', marker_color='orange'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=years, y=weights, name='Weight', mode='lines+markers', 
                  line=dict(color='red'), yaxis='y2'),
        row=1, col=2, secondary_y=True
    )
    
    # 3. Quarterly performance patterns
    current_quarters = calculate_quarterly_returns(current_data['Close'].values)
    
    for i, year_data in enumerate(similar_years_data['similar_years_details'][:3]):
        year = year_data['year']
        if year in years_forecaster.yearly_windows:
            year_prices = years_forecaster.yearly_windows[year]['Close'].values
            year_quarters = calculate_quarterly_returns(year_prices)
            
            fig.add_trace(
                go.Scatter(x=['Q1', 'Q2', 'Q3', 'Q4'], y=year_quarters,
                          name=f'{year} Quarterly', line=dict(color=colors[i], dash='dot')),
                row=2, col=1
            )
    
    fig.add_trace(
        go.Scatter(x=['Q1', 'Q2', 'Q3', 'Q4'], y=current_quarters,
                  name=f'Analysis {current_year}', line=dict(width=4)),
        row=2, col=1
    )
    
    # 4. Next year returns distribution
    next_year_returns = [yd['next_year_return'] for yd in similar_years_data['similar_years_details']]
    
    fig.add_trace(
        go.Box(y=next_year_returns, name='Next Year Returns', boxpoints='all',
               marker_color='lightblue', line=dict(color='blue')),
        row=2, col=2
    )
    
    # Add forecast line
    forecast_return = forecast['final_forecast']['final_expected_return']
    fig.add_trace(
        go.Scatter(x=[0], y=[forecast_return], mode='markers',
                  name=f'{forecast["forecast_year"]} Forecast', 
                  marker=dict(size=15, color='red', symbol='star')),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text=f"Detailed Similar Years Analysis: {current_year} → {forecast['forecast_year']}",
        template='plotly_dark',
        showlegend=True
    )
    
    fig.show()

def calculate_quarterly_returns(prices: np.ndarray) -> List[float]:
    """Calculate quarterly returns from weekly prices"""
    quarters = []
    quarter_points = [0, 13, 26, 39, 51]  # Approximate quarter boundaries
    
    for i in range(4):
        start_idx = quarter_points[i]
        end_idx = quarter_points[i+1] if i < 3 else len(prices)-1
        quarter_return = (prices[end_idx] / prices[start_idx]) - 1
        quarters.append(quarter_return)
    
    return quarters

# Create detailed comparison (with error handling)
if 'error' not in final_forecast:
    print("\n🔍 CREATING DETAILED SIMILAR YEARS COMPARISON...")
    try:
        create_similar_years_comparison(final_forecast)
    except Exception as e:
        print(f"⚠️  Could not create detailed comparison: {e}")
        print("📊 Creating simplified visualization instead...")
        create_simplified_comparison(final_forecast)
        
        def create_simplified_comparison(forecast: Dict):
    """Create a simplified comparison when detailed data isn't available"""
    
    similar_years_data = forecast['component_analysis']['similar_years']
    
    if similar_years_data['similar_years_count'] == 0:
        print("No similar years to visualize")
        return
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Similar Years Performance',
            'Forecast Components'
        ]
    )
    
    # 1. Similar years returns
    years = [f"{yd['year']}" for yd in similar_years_data['similar_years_details']]
    returns = [yd['next_year_return'] for yd in similar_years_data['similar_years_details']]
    similarities = [yd['similarity_score'] for yd in similar_years_data['similar_years_details']]
    
    fig.add_trace(
        go.Bar(x=years, y=returns, name='Next Year Returns',
               marker_color=['green' if r > 0 else 'red' for r in returns],
               text=[f'{r:+.2%}' for r in returns],
               textposition='auto'),
        row=1, col=1
    )
    
    # Add average line
    avg_return = np.mean(returns)
    fig.add_hline(y=avg_return, line_dash="dash", line_color="blue", 
                  annotation_text=f"Average: {avg_return:+.2%}",
                  row=1, col=1)
    
    # 2. Forecast components
    components = forecast['final_forecast']['component_forecasts']
    methods = list(components.keys())
    component_returns = [components[method] for method in methods]
    
    fig.add_trace(
        go.Bar(x=[m.replace('_', ' ').title() for m in methods],
               y=component_returns,
               name='Component Returns',
               marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
               text=[f'{r:+.2%}' for r in component_returns],
               textposition='auto'),
        row=1, col=2
    )
    
    # Add final forecast line
    final_return = forecast['final_forecast']['final_expected_return']
    fig.add_hline(y=final_return, line_dash="dash", line_color="red", 
                  annotation_text=f"Final: {final_return:+.2%}",
                  row=1, col=2)
    
    fig.update_layout(
        height=400,
        title_text=f"Simplified Analysis: {forecast['current_year']} → {forecast['forecast_year']}",
        template='plotly_dark',
        showlegend=False
    )
    
    fig.show()
    
    def export_forecast_results(forecast: Dict, filename: str = "gold_forecast_results"):
    """Export forecast results to files with proper type handling"""
    
    if 'error' in forecast:
        print("Cannot export - forecast generation failed")
        return
    
    # Create results directory
    os.makedirs('/kaggle/working/forecast_results', exist_ok=True)
    
    # 1. Save forecast as JSON
    json_path = f'/kaggle/working/forecast_results/{filename}.json'
    
    # Enhanced function to convert numpy types to Python native for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy and pandas types to JSON-serializable formats"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif hasattr(obj, 'dtype'):  # Catch any other numpy types
            return float(obj) if np.issubdtype(obj.dtype, np.floating) else int(obj)
        return obj
    
    def make_json_serializable(data):
        """Recursively make data JSON serializable"""
        if isinstance(data, dict):
            return {k: make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [make_json_serializable(item) for item in data]
        elif isinstance(data, tuple):
            return [make_json_serializable(item) for item in data]
        else:
            return convert_to_serializable(data)
    
    try:
        serializable_forecast = make_json_serializable(forecast)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_forecast, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON exported: {json_path}")
        
    except Exception as e:
        print(f"❌ Error exporting JSON: {e}")
        # Create a minimal JSON with just the key data
        minimal_forecast = {
            'current_year': int(forecast.get('current_year', 0)),
            'forecast_year': int(forecast.get('forecast_year', 0)),
            'current_price': float(forecast.get('current_price', 0)),
            'forecast_date': forecast.get('forecast_date', ''),
            'expected_return': float(forecast['final_forecast']['final_expected_return']) if 'final_forecast' in forecast else 0,
            'expected_price': float(forecast.get('current_price', 0) * (1 + forecast['final_forecast']['final_expected_return'])) if 'final_forecast' in forecast else 0
        }
        
        with open(json_path, 'w') as f:
            json.dump(minimal_forecast, f, indent=2)
        print(f"✅ Minimal JSON exported: {json_path}")
    
    # 2. Save summary as text
    txt_path = f'/kaggle/working/forecast_results/{filename}_summary.txt'
    
    try:
        with open(txt_path, 'w') as f:
            f.write("GOLD PRICE FORECAST SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Forecast Date: {forecast.get('forecast_date', 'N/A')}\n")
            f.write(f"Analysis Year: {forecast.get('current_year', 'N/A')}\n")
            f.write(f"Forecast Year: {forecast.get('forecast_year', 'N/A')}\n")
            f.write(f"Current Price: ${forecast.get('current_price', 0):,.2f}\n")
            
            if 'final_forecast' in forecast:
                expected_return = forecast['final_forecast']['final_expected_return']
                expected_price = forecast['current_price'] * (1 + expected_return)
                f.write(f"Expected Return: {expected_return:+.2%}\n")
                f.write(f"Expected Price: ${expected_price:,.2f}\n")
            
            if 'component_analysis' in forecast and 'similar_years' in forecast['component_analysis']:
                similar_count = forecast['component_analysis']['similar_years']['similar_years_count']
                bullish_prob = forecast['component_analysis']['similar_years']['bullish_probability']
                f.write(f"Similar Years Used: {similar_count}\n")
                f.write(f"Bullish Probability: {bullish_prob:.1%}\n")
        
        print(f"✅ Summary exported: {txt_path}")
        
    except Exception as e:
        print(f"❌ Error exporting summary: {e}")
    
    # 3. Save similar years data as CSV
    try:
        if ('component_analysis' in forecast and 
            'similar_years' in forecast['component_analysis'] and 
            forecast['component_analysis']['similar_years']['similar_years_count'] > 0):
            
            similar_years_data = forecast['component_analysis']['similar_years']['similar_years_details']
            
            # Convert to DataFrame with proper type handling
            clean_data = []
            for year_data in similar_years_data:
                clean_year = {
                    'year': int(year_data['year']),
                    'similarity_score': float(year_data['similarity_score']),
                    'next_year_return': float(year_data['next_year_return'])
                }
                # Add detail scores if they exist
                if 'detail_scores' in year_data:
                    for key, value in year_data['detail_scores'].items():
                        clean_year[f'detail_{key}'] = float(value) if isinstance(value, (int, float, np.number)) else str(value)
                clean_data.append(clean_year)
            
            df_similar = pd.DataFrame(clean_data)
            csv_path = f'/kaggle/working/forecast_results/{filename}_similar_years.csv'
            df_similar.to_csv(csv_path, index=False)
            print(f"✅ Similar years data exported: {csv_path}")
            
    except Exception as e:
        print(f"❌ Error exporting similar years data: {e}")
    
    # 4. Save component breakdown as CSV
    try:
        if 'final_forecast' in forecast:
            components = forecast['final_forecast']['component_forecasts']
            weights = forecast['final_forecast']['weights_used']
            
            component_data = []
            for method, return_val in components.items():
                component_data.append({
                    'methodology': method.replace('_', ' ').title(),
                    'expected_return': float(return_val),
                    'weight': float(weights[method])
                })
            
            df_components = pd.DataFrame(component_data)
            components_path = f'/kaggle/working/forecast_results/{filename}_components.csv'
            df_components.to_csv(components_path, index=False)
            print(f"✅ Component breakdown exported: {components_path}")
            
    except Exception as e:
        print(f"❌ Error exporting component data: {e}")
    
    print(f"\n📁 All exports saved to: /kaggle/working/forecast_results/")

# Create summary and export results
if 'error' not in final_forecast:
    print("\n" + "="*80)
    print("📋 FINAL FORECAST SUMMARY & EXPORT")
    print("="*80)
    
    create_forecast_summary(final_forecast)
    export_forecast_results(final_forecast, "gold_forecast")
else:
    print("❌ Cannot export - forecast generation failed")
    
    def export_forecast_results_simple(forecast: Dict, filename: str = "gold_forecast_results"):
    """Simple export function that handles basic data types only"""
    
    if 'error' in forecast:
        print("Cannot export - forecast generation failed")
        return
    
    # Create results directory
    os.makedirs('/kaggle/working/forecast_results', exist_ok=True)
    
    # 1. Simple text summary
    txt_path = f'/kaggle/working/forecast_results/{filename}_summary.txt'
    
    try:
        with open(txt_path, 'w') as f:
            f.write("GOLD PRICE FORECAST SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            f.write(f"Forecast Date: {forecast.get('forecast_date', 'N/A')}\n")
            f.write(f"Analysis Year: {forecast.get('current_year', 'N/A')}\n")
            f.write(f"Forecast Year: {forecast.get('forecast_year', 'N/A')}\n")
            f.write(f"Current Price: ${forecast.get('current_price', 0):,.2f}\n\n")
            
            # Forecast results
            if 'final_forecast' in forecast:
                final = forecast['final_forecast']
                expected_return = final['final_expected_return']
                expected_price = forecast['current_price'] * (1 + expected_return)
                
                f.write("FINAL FORECAST:\n")
                f.write(f"Expected Return: {expected_return:+.2%}\n")
                f.write(f"Expected Price: ${expected_price:,.2f}\n")
                f.write(f"Price Change: ${expected_price - forecast['current_price']:+,.2f}\n\n")
                
                # Components
                f.write("COMPONENT BREAKDOWN:\n")
                for method, return_val in final['component_forecasts'].items():
                    f.write(f"  {method.replace('_', ' ').title()}: {return_val:+.2%}\n")
                
                f.write(f"\nML Model: {final.get('ml_model_type', 'N/A')}\n")
            
            # Similar years info
            if 'component_analysis' in forecast and 'similar_years' in forecast['component_analysis']:
                similar = forecast['component_analysis']['similar_years']
                f.write(f"\nSIMILAR YEARS ANALYSIS:\n")
                f.write(f"Years Found: {similar['similar_years_count']}\n")
                f.write(f"Bullish Probability: {similar['bullish_probability']:.1%}\n")
                
                if similar['similar_years_count'] > 0:
                    f.write(f"\nTOP SIMILAR YEARS:\n")
                    for i, year_data in enumerate(similar['similar_years_details'][:3]):
                        f.write(f"  {i+1}. {year_data['year']} "
                               f"(Similarity: {year_data['similarity_score']:.3f}, "
                               f"Return: {year_data['next_year_return']:+.2%})\n")
        
        print(f"✅ Summary exported: {txt_path}")
        
    except Exception as e:
        print(f"❌ Error exporting summary: {e}")
    
    # 2. Export similar years as CSV (simple)
    try:
        if ('component_analysis' in forecast and 
            'similar_years' in forecast['component_analysis'] and 
            forecast['component_analysis']['similar_years']['similar_years_count'] > 0):
            
            similar_data = forecast['component_analysis']['similar_years']['similar_years_details']
            
            # Simple data extraction
            rows = []
            for year_data in similar_data:
                row = {
                    'year': int(year_data['year']),
                    'similarity_score': float(year_data['similarity_score']),
                    'next_year_return': float(year_data['next_year_return'])
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            csv_path = f'/kaggle/working/forecast_results/{filename}_similar_years.csv'
            df.to_csv(csv_path, index=False)
            print(f"✅ Similar years exported: {csv_path}")
            
    except Exception as e:
        print(f"❌ Error exporting similar years: {e}")
    
    print(f"\n📁 Exports saved to: /kaggle/working/forecast_results/")

# Use the simple version if needed
if 'error' not in final_forecast:
    print("\n" + "="*80)
    print("📋 FINAL FORECAST SUMMARY & EXPORT")
    print("="*80)
    
    create_forecast_summary(final_forecast)
    
    # Try the comprehensive export first, fall back to simple if it fails
    try:
        export_forecast_results(final_forecast, "gold_forecast")
    except Exception as e:
        print(f"⚠️  Comprehensive export failed: {e}")
        print("🔄 Trying simple export...")
        export_forecast_results_simple(final_forecast, "gold_forecast_simple")
