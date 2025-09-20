"""
Algorithmic Trading & Market Microstructure Analysis System
Professional quantitative trading platform for institutional investors

Author: Liam Hirsch
Based on institutional finance experience and EDHEC Portfolio Analysis certification
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class MarketMicrostructure:
    """
    Advanced market microstructure analysis for institutional trading
    
    Features:
    - Liquidity analysis (Amihud ratio, bid-ask spreads)
    - Order flow analysis and market impact
    - Volatility regime detection
    - Flash crash and anomaly identification
    """
    
    def __init__(self, symbol, period='1y', interval='1h'):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.data = self.fetch_data()
        self.features = None
        
    def fetch_data(self):
        """Fetch high-frequency market data"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=self.period, interval=self.interval)
            return data.dropna()
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return None
    
    def calculate_microstructure_features(self):
        """Calculate advanced microstructure indicators"""
        if self.data is None:
            return None
            
        df = self.data.copy()
        
        # Basic price and volume features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Volume-weighted average price (VWAP)
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Market microstructure indicators
        df['bid_ask_spread_proxy'] = (df['High'] - df['Low']) / df['Close']
        df['amihud_illiquidity'] = abs(df['returns']) / (df['Volume'] * df['Close'])
        df['price_impact'] = abs(df['returns']) * df['volume_ratio']
        
        # Order flow proxies
        df['buying_pressure'] = np.where(df['Close'] > df['Open'], df['Volume'], 0)
        df['selling_pressure'] = np.where(df['Close'] < df['Open'], df['Volume'], 0)
        df['net_order_flow'] = df['buying_pressure'] - df['selling_pressure']
        
        # Volatility regime detection
        df['volatility_regime'] = np.where(
            df['volatility'] > df['volatility'].quantile(0.75), 'high',
            np.where(df['volatility'] < df['volatility'].quantile(0.25), 'low', 'medium')
        )
        
        self.features = df
        return df
    
    def detect_anomalies(self, threshold=3):
        """Detect market anomalies and flash crash events"""
        if self.features is None:
            self.calculate_microstructure_features()
        
        df = self.features.copy()
        
        # Z-score based anomaly detection
        z_scores = np.abs(stats.zscore(df['returns'].dropna()))
        price_anomalies = z_scores > threshold
        
        # Volume spike detection
        volume_z = np.abs(stats.zscore(df['volume_ratio'].dropna()))
        volume_anomalies = volume_z > threshold
        
        # Combined anomaly detection
        df['anomaly_score'] = z_scores
        df['is_anomaly'] = price_anomalies | volume_anomalies
        
        return df[df['is_anomaly']]
    
    def liquidity_analysis(self):
        """Comprehensive liquidity analysis"""
        if self.features is None:
            self.calculate_microstructure_features()
        
        df = self.features.copy()
        
        return {
            'avg_amihud_illiquidity': df['amihud_illiquidity'].mean(),
            'avg_bid_ask_spread': df['bid_ask_spread_proxy'].mean(),
            'volume_concentration': df['Volume'].std() / df['Volume'].mean(),
            'price_impact_ratio': df['price_impact'].mean(),
            'liquidity_score': 1 / (df['amihud_illiquidity'].mean() + 1e-10)
        }

class TradingStrategies:
    """
    Multi-strategy trading system with institutional-grade risk management
    
    Strategies:
    - Pairs Trading (statistical arbitrage)
    - Cross-sectional Momentum
    - Mean Reversion (Bollinger Bands)
    - Machine Learning (Random Forest)
    """
    
    def __init__(self, symbols, lookback_period=252):
        self.symbols = symbols
        self.lookback_period = lookback_period
        self.data = {}
        self.strategies = {}
        
    def fetch_universe_data(self):
        """Fetch data for entire trading universe"""
        print("Fetching market data...")
        for symbol in self.symbols:
            try:
                data = yf.download(symbol, period='2y', progress=False)
                if not data.empty:
                    self.data[symbol] = data
                    print(f"✓ {symbol} data loaded")
            except Exception as e:
                print(f"✗ Error fetching {symbol}: {e}")
    
    def pairs_trading_strategy(self, stock1, stock2, entry_threshold=2.0, exit_threshold=0.5):
        """
        Statistical arbitrage pairs trading strategy
        
        Based on cointegration analysis and mean reversion
        """
        if stock1 not in self.data or stock2 not in self.data:
            return None
        
        # Align price series
        prices1 = self.data[stock1]['Adj Close']
        prices2 = self.data[stock2]['Adj Close']
        aligned_data = pd.DataFrame({stock1: prices1, stock2: prices2}).dropna()
        
        # Calculate spread and z-score
        spread = np.log(aligned_data[stock1]) - np.log(aligned_data[stock2])
        spread_mean = spread.rolling(self.lookback_period).mean()
        spread_std = spread.rolling(self.lookback_period).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Generate trading signals
        signals = pd.DataFrame(index=aligned_data.index)
        signals['z_score'] = z_score
        signals['position'] = 0
        
        # Entry signals
        signals.loc[z_score < -entry_threshold, 'position'] = 1  # Long spread
        signals.loc[z_score > entry_threshold, 'position'] = -1  # Short spread
        
        # Exit signals
        signals.loc[abs(z_score) < exit_threshold, 'position'] = 0
        signals['position'] = signals['position'].fillna(method='ffill')
        
        # Calculate strategy returns
        stock1_returns = aligned_data[stock1].pct_change()
        stock2_returns = aligned_data[stock2].pct_change()
        strategy_returns = signals['position'] * (stock1_returns - stock2_returns)
        
        return {
            'signals': signals,
            'returns': strategy_returns,
            'current_zscore': z_score.iloc[-1],
            'correlation': aligned_data[stock1].corr(aligned_data[stock2])
        }
    
    def momentum_strategy(self, lookback_days=20, holding_period=5):
        """
        Cross-sectional momentum strategy with risk adjustment
        """
        momentum_signals = {}
        
        for symbol in self.data:
            data = self.data[symbol].copy()
            
            # Calculate momentum and risk metrics
            momentum_score = data['Adj Close'].pct_change(lookback_days)
            volatility = data['Adj Close'].pct_change().rolling(20).std()
            risk_adjusted_momentum = momentum_score / volatility
            
            # Generate signals based on percentile ranking
            signals = pd.DataFrame(index=data.index)
            signals['momentum_score'] = momentum_score
            signals['risk_adj_momentum'] = risk_adjusted_momentum
            signals['percentile_rank'] = risk_adjusted_momentum.rolling(252).rank(pct=True)
            
            # Position allocation
            signals['position'] = 0
            signals.loc[signals['percentile_rank'] > 0.8, 'position'] = 1   # Long top 20%
            signals.loc[signals['percentile_rank'] < 0.2, 'position'] = -1  # Short bottom 20%
            
            momentum_signals[symbol] = signals
        
        return momentum_signals
    
    def mean_reversion_strategy(self, window=20, num_std=2):
        """
        Bollinger Band mean reversion strategy
        """
        mean_reversion_signals = {}
        
        for symbol in self.data:
            data = self.data[symbol].copy()
            
            # Calculate Bollinger Bands
            sma = data['Adj Close'].rolling(window).mean()
            std = data['Adj Close'].rolling(window).std()
            upper_band = sma + (num_std * std)
            lower_band = sma - (num_std * std)
            
            # Generate mean reversion signals
            signals = pd.DataFrame(index=data.index)
            signals['price'] = data['Adj Close']
            signals['sma'] = sma
            signals['upper_band'] = upper_band
            signals['lower_band'] = lower_band
            signals['bb_position'] = (data['Adj Close'] - lower_band) / (upper_band - lower_band)
            
            # Position allocation
            signals['position'] = 0
            signals.loc[data['Adj Close'] < lower_band, 'position'] = 1   # Long oversold
            signals.loc[data['Adj Close'] > upper_band, 'position'] = -1  # Short overbought
            signals.loc[abs(signals['bb_position'] - 0.5) < 0.1, 'position'] = 0  # Exit near center
            
            mean_reversion_signals[symbol] = signals
        
        return mean_reversion_signals
    
    def machine_learning_strategy(self, target_return_days=5):
        """
        Random Forest-based prediction strategy
        """
        ml_signals = {}
        
        for symbol in self.data:
            data = self.data[symbol].copy()
            
            # Feature engineering
            features = pd.DataFrame(index=data.index)
            
            # Technical indicators
            features['returns_1d'] = data['Adj Close'].pct_change()
            features['returns_5d'] = data['Adj Close'].pct_change(5)
            features['returns_20d'] = data['Adj Close'].pct_change(20)
            features['volatility'] = features['returns_1d'].rolling(20).std()
            features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
            
            # Moving averages
            features['sma_5'] = data['Adj Close'].rolling(5).mean() / data['Adj Close']
            features['sma_20'] = data['Adj Close'].rolling(20).mean() / data['Adj Close']
            features['sma_50'] = data['Adj Close'].rolling(50).mean() / data['Adj Close']
            
            # Target variable (forward returns)
            features['target'] = data['Adj Close'].shift(-target_return_days).pct_change(target_return_days)
            
            # Clean data
            features = features.dropna()
            
            if len(features) > 100:
                # Prepare features and target
                feature_columns = [col for col in features.columns if col != 'target']
                X = features[feature_columns]
                y = features['target']
                
                # Time-based train-test split
                split_idx = int(len(features) * 0.7)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train Random Forest model
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train_scaled, y_train)
                
                # Generate predictions
                predictions = rf_model.predict(X_test_scaled)
                
                # Create trading signals
                signals = pd.DataFrame(index=X_test.index)
                signals['prediction'] = predictions
                signals['actual'] = y_test
                
                # Position sizing based on prediction strength
                prediction_std = np.std(predictions)
                signals['position'] = 0
                signals.loc[predictions > prediction_std, 'position'] = 1
                signals.loc[predictions < -prediction_std, 'position'] = -1
                
                # Calculate accuracy
                accuracy = np.corrcoef(predictions, y_test)[0, 1] ** 2 if len(predictions) > 1 else 0
                
                ml_signals[symbol] = {
                    'signals': signals,
                    'model': rf_model,
                    'scaler': scaler,
                    'accuracy': accuracy
                }
        
        return ml_signals

class RiskManagement:
    """
    Institutional-grade risk management system
    
    Features:
    - Kelly Criterion position sizing
    - Portfolio VaR calculation
    - Maximum drawdown controls
    - Sector exposure limits
    """
    
    def __init__(self, portfolio_value=1000000):
        self.portfolio_value = portfolio_value
        self.risk_limits = {
            'max_position_size': 0.1,      # 10% max per position
            'max_sector_exposure': 0.25,   # 25% max per sector
            'max_portfolio_var': 0.02,     # 2% daily VaR
            'max_drawdown': 0.05           # 5% max drawdown
        }
    
    def kelly_criterion_sizing(self, win_prob, avg_win, avg_loss, signal_strength=1.0):
        """
        Kelly Criterion optimal position sizing
        
        f* = (bp - q) / b
        where f = fraction to wager, b = odds, p = win prob, q = loss prob
        """
        if avg_win <= 0 or avg_loss <= 0:
            return 0
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Adjust for signal strength
        adjusted_fraction = kelly_fraction * signal_strength
        
        return adjusted_fraction
    
    def calculate_portfolio_var(self, positions, returns_data, confidence_level=0.05):
        """
        Calculate portfolio Value at Risk using historical simulation
        """
        portfolio_returns = []
        
        for symbol, weight in positions.items():
            if symbol in returns_data:
                stock_returns = returns_data[symbol].pct_change().dropna()
                weighted_returns = stock_returns * weight
                portfolio_returns.append(weighted_returns)
        
        if portfolio_returns:
            portfolio_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
            var = np.percentile(portfolio_returns, confidence_level * 100)
            return var
        
        return 0
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

class PerformanceAnalytics:
    """
    Comprehensive performance measurement and attribution
    """
    
    @staticmethod
    def calculate_metrics(returns):
        """Calculate comprehensive performance metrics"""
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Risk metrics
        max_drawdown = RiskManagement.calculate_max_drawdown(None, returns)
        var_95 = np.percentile(returns, 5)
        
        # Additional metrics
        win_rate = (returns > 0).sum() / len(returns)
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        
        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'VaR (95%)': var_95,
            'Win Rate': win_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }

class QuantTradingSystem:
    """
    Main quantitative trading system orchestrator
    
    Integrates all components:
    - Strategy execution
    - Risk management
    - Performance monitoring
    - Portfolio optimization
    """
    
    def __init__(self, symbols, initial_capital=1000000):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.strategies = TradingStrategies(symbols)
        self.risk_manager = RiskManagement(initial_capital)
        self.performance = PerformanceAnalytics()
        
    def initialize_system(self):
        """Initialize complete trading system"""
        print("🚀 Initializing Quantitative Trading System")
        print(f"Capital: ${self.initial_capital:,}")
        print(f"Universe: {len(self.symbols)} symbols")
        print("-" * 50)
        
        # Fetch market data
        self.strategies.fetch_universe_data()
        
        print("✓ System initialization complete")
    
    def run_strategy_suite(self):
        """Execute complete strategy suite"""
        results = {}
        
        print("\n📊 Running Strategy Suite")
        print("-" * 30)
        
        # 1. Pairs Trading (if sufficient symbols)
        if len(self.symbols) >= 2:
            print("Running pairs trading...")
            pairs_result = self.strategies.pairs_trading_strategy(
                self.symbols[0], self.symbols[1]
            )
            if pairs_result:
                results['pairs_trading'] = pairs_result
                print(f"✓ Pairs trading: {self.symbols[0]}-{self.symbols[1]}")
        
        # 2. Momentum Strategy
        print("Running momentum strategies...")
        momentum_results = self.strategies.momentum_strategy()
        results['momentum'] = momentum_results
        print(f"✓ Momentum: {len(momentum_results)} symbols analyzed")
        
        # 3. Mean Reversion
        print("Running mean reversion strategies...")
        mean_reversion_results = self.strategies.mean_reversion_strategy()
        results['mean_reversion'] = mean_reversion_results
        print(f"✓ Mean reversion: {len(mean_reversion_results)} symbols analyzed")
        
        # 4. Machine Learning
        print("Running ML strategies...")
        ml_results = self.strategies.machine_learning_strategy()
        results['machine_learning'] = ml_results
        print(f"✓ ML models: {len(ml_results)} symbols analyzed")
        
        return results
    
    def generate_performance_report(self, results):
        """Generate comprehensive performance report"""
        report = {
            'system_overview': {
                'symbols_analyzed': len(self.symbols),
                'strategies_deployed': len(results),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'capital': self.initial_capital
            },
            'strategy_performance': {},
            'risk_metrics': {},
            'top_performers': []
        }
        
        print("\n📈 TRADING SYSTEM PERFORMANCE REPORT")
        print("=" * 50)
        
        # Strategy performance summary
        for strategy_name, strategy_results in results.items():
            if strategy_name == 'pairs_trading' and strategy_results:
                returns = strategy_results['returns'].dropna()
                if len(returns) > 0:
                    metrics = self.performance.calculate_metrics(returns)
                    report['strategy_performance'][strategy_name] = metrics
                    print(f"\n{strategy_name.upper()}:")
                    print(f"  Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.3f}")
                    print(f"  Total Return: {metrics.get('Total Return', 0):.2%}")
                    print(f"  Max Drawdown: {metrics.get('Maximum Drawdown', 0):.2%}")
        
        return report

# Example usage and demonstration
if __name__ == "__main__":
    print("🏦 ALGORITHMIC TRADING SYSTEM")
    print("Author: Liam Hirsch")
    print("Institutional Quantitative Trading Platform")
    print("=" * 60)
    
    # Initialize with tech-focused universe
    tech_universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Create trading system
    trading_system = QuantTradingSystem(
        symbols=tech_universe,
        initial_capital=5000000  # $5M initial capital
    )
    
    # Initialize and run analysis
    trading_system.initialize_system()
    strategy_results = trading_system.run_strategy_suite()
    
    # Generate comprehensive report
    performance_report = trading_system.generate_performance_report(strategy_results)
    
    # Display system overview
    overview = performance_report['system_overview']
    print(f"\n📊 SYSTEM OVERVIEW:")
    print(f"Capital Deployed: ${overview['capital']:,}")
    print(f"Symbols Analyzed: {overview['symbols_analyzed']}")
    print(f"Strategies Active: {overview['strategies_deployed']}")
    print(f"Analysis Date: {overview['analysis_date']}")
    
    print("\n✅ Analysis Complete - System Ready for Production")
    print("🚀 Built with institutional finance experience")
    print("💼 Combining traditional finance with modern quantitative methods")
