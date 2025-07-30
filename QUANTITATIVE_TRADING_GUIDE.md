# Quantitative Trading System Guide

## Tổng quan

Hệ thống Quantitative Trading được thiết kế để cung cấp các công cụ phân tích định lượng tiên tiến cho trading bot. Hệ thống bao gồm các thành phần chính sau:

### 1. Risk Management (Quản lý rủi ro)
- **VaR Calculator**: Tính toán Value at Risk với nhiều phương pháp
- **Dynamic Position Sizer**: Tối ưu hóa kích thước vị thế
- **Risk Manager**: Quản lý rủi ro tổng thể

### 2. Statistical Validation (Xác thực thống kê)
- **Statistical Signal Validator**: Xác thực tín hiệu giao dịch
- **Hypothesis Testing**: Kiểm định giả thuyết thống kê
- **Performance Metrics**: Các chỉ số hiệu suất

### 3. Portfolio Optimization (Tối ưu hóa danh mục)
- **Markowitz Optimization**: Tối ưu hóa mean-variance
- **Black-Litterman Model**: Mô hình Black-Litterman
- **Risk Parity**: Phân bổ rủi ro đồng đều
- **Maximum Sharpe Ratio**: Tối đa hóa tỷ lệ Sharpe

### 4. Market Microstructure Analysis (Phân tích cấu trúc thị trường)
- **Bid-Ask Spread Analysis**: Phân tích spread
- **Order Flow Imbalance**: Mất cân bằng luồng lệnh
- **Market Depth Analysis**: Phân tích độ sâu thị trường
- **Price Impact Estimation**: Ước tính tác động giá

### 5. Advanced Backtesting Engine (Engine backtesting nâng cao)
- **Realistic Market Simulation**: Mô phỏng thị trường thực tế
- **Transaction Costs**: Chi phí giao dịch
- **Risk Management Integration**: Tích hợp quản lý rủi ro
- **Performance Analytics**: Phân tích hiệu suất

### 6. Factor Models (Mô hình factor)
- **PCA-based Factor Analysis**: Phân tích factor dựa trên PCA
- **Multi-Factor Models**: Mô hình đa factor
- **Factor Attribution**: Phân bổ factor
- **Factor Timing**: Thời điểm factor

## Cách sử dụng

### 1. Khởi tạo hệ thống

```python
from src.quantitative import QuantitativeTradingSystem

# Cấu hình hệ thống
config = {
    'confidence_level': 0.95,
    'max_position_size': 0.02,
    'risk_free_rate': 0.02,
    'optimization_method': 'markowitz',
    'n_factors': 5
}

# Khởi tạo hệ thống
system = QuantitativeTradingSystem(config)
```

### 2. Phân tích cơ hội giao dịch

```python
# Dữ liệu thị trường
market_data = {
    'orderbook': orderbook_data,
    'trades': trade_data,
    'returns': returns_data,
    'portfolio_data': portfolio_returns,
    'factor_data': factor_returns
}

# Tín hiệu giao dịch
signal_data = {
    'signal_strength': 0.8,
    'signal_type': 'momentum',
    'position_size': 0.01,
    'confidence': 0.7
}

# Phân tích toàn diện
analysis_results = system.analyze_trading_opportunity(market_data, signal_data)
```

### 3. Tối ưu hóa danh mục

```python
# Tối ưu hóa danh mục
portfolio_results = system.optimize_portfolio(returns_data, method='markowitz')

# Kết quả bao gồm:
# - Optimal weights
# - Expected return
# - Portfolio volatility
# - Sharpe ratio
```

### 4. Backtesting chiến lược

```python
# Định nghĩa chiến lược
def my_strategy(row, historical_data, params):
    # Logic chiến lược
    return {'action': 'buy', 'position_size': 0.1}

# Chạy backtest
backtest_results = system.run_backtest(
    strategy_function=my_strategy,
    historical_data=historical_data,
    strategy_params={'param1': 0.1}
)
```

### 5. Phân tích factor

```python
# Xây dựng mô hình factor
factor_results = system.factor_model.build_factor_model(returns_data)

# Phân tích timing factor
timing_results = factor_results['factor_timing']

# Tạo tín hiệu giao dịch từ factor
signals = system.factor_model.generate_factor_signals(timing_results)
```

## Các thành phần chi tiết

### Risk Manager

```python
from src.quantitative import RiskManager

risk_manager = RiskManager(confidence_level=0.95, max_position_size=0.02)

# Tính VaR
var_results = risk_manager.var_calculator.calculate_var(
    returns=returns,
    position_size=10000,
    method='all'
)

# Tối ưu hóa kích thước vị thế
position_results = risk_manager.position_sizer.calculate_position_size(
    signal_strength=0.8,
    volatility=0.02,
    correlation=0.3,
    var_limit=0.02
)
```

### Statistical Validator

```python
from src.quantitative import StatisticalSignalValidator

validator = StatisticalSignalValidator(min_p_value=0.05, min_t_stat=2.0)

# Xác thực tín hiệu
validation_results = validator.validate_signal(signal_data, returns)

# Kiểm định giả thuyết
hypothesis_results = validator.perform_hypothesis_test(returns)
```

### Portfolio Optimizer

```python
from src.quantitative import PortfolioOptimizer

optimizer = PortfolioOptimizer(risk_free_rate=0.02, target_return=0.10)

# Tối ưu hóa với các phương pháp khác nhau
methods = ['markowitz', 'black_litterman', 'risk_parity', 'max_sharpe']

for method in methods:
    results = optimizer.optimize_portfolio(returns_data, method=method)
```

### Market Microstructure Analyzer

```python
from src.quantitative import MarketMicrostructureAnalyzer

analyzer = MarketMicrostructureAnalyzer(min_tick_size=0.0001)

# Phân tích cấu trúc thị trường
analysis_results = analyzer.analyze_market_structure(orderbook_data, trade_data)

# Các chỉ số quan trọng:
# - Bid-ask spread
# - Order flow imbalance
# - Market depth
# - Price impact
# - Liquidity metrics
```

### Advanced Backtesting Engine

```python
from src.quantitative import AdvancedBacktestingEngine

engine = AdvancedBacktestingEngine(
    initial_capital=100000,
    commission_rate=0.001,
    slippage_rate=0.0005
)

# Chạy backtest
backtest_results = engine.run_backtest(
    strategy_function=my_strategy,
    historical_data=historical_data,
    risk_management={'max_drawdown': 0.15, 'var_limit': 0.02}
)
```

### Factor Model

```python
from src.quantitative import FactorModel

factor_model = FactorModel(n_factors=5, min_eigenvalue=1.0)

# Xây dựng mô hình factor
factor_results = factor_model.build_factor_model(returns_data)

# Phân tích timing
timing_results = factor_results['factor_timing']

# Tạo tín hiệu giao dịch
signals = factor_model.generate_factor_signals(timing_results)
```

## Cấu hình nâng cao

### Risk Management Configuration

```python
risk_config = {
    'confidence_level': 0.95,
    'max_position_size': 0.02,
    'var_limit': 0.02,
    'max_drawdown': 0.15,
    'correlation_limit': 0.7
}
```

### Portfolio Optimization Constraints

```python
constraints = {
    'min_weight': 0.0,
    'max_weight': 0.3,
    'target_return': 0.10,
    'max_volatility': 0.15
}
```

### Backtesting Parameters

```python
backtest_config = {
    'initial_capital': 100000,
    'commission_rate': 0.001,
    'slippage_rate': 0.0005,
    'risk_free_rate': 0.02,
    'rebalance_frequency': 'daily'
}
```

## Monitoring và Reporting

### System Summary

```python
# Lấy tổng quan hệ thống
summary = system.get_system_summary()

print(f"Total analyses: {summary['total_analyses']}")
print(f"Average Sharpe ratio: {summary['backtesting_summary']['avg_sharpe_ratio']}")
print(f"Recommendation success rate: {summary['recommendation_success_rate']}")
```

### Export Analysis Report

```python
# Xuất báo cáo phân tích
report = system.export_analysis_report()

# Báo cáo bao gồm:
# - Market analysis
# - Risk analysis
# - Statistical validation
# - Portfolio analysis
# - Factor analysis
# - Trading recommendation
```

## Best Practices

### 1. Risk Management
- Luôn sử dụng VaR để giới hạn rủi ro
- Đa dạng hóa danh mục
- Theo dõi drawdown liên tục
- Sử dụng position sizing động

### 2. Statistical Validation
- Kiểm tra tính hợp lệ thống kê của tín hiệu
- Sử dụng multiple timeframes
- Validate out-of-sample
- Monitor performance metrics

### 3. Portfolio Optimization
- Rebalance định kỳ
- Sử dụng rolling windows
- Monitor factor exposures
- Adjust for transaction costs

### 4. Market Microstructure
- Monitor bid-ask spreads
- Analyze order flow patterns
- Consider market impact
- Use liquidity metrics

### 5. Factor Analysis
- Identify persistent factors
- Monitor factor timing
- Diversify factor exposures
- Rebalance factor weights

## Troubleshooting

### Common Issues

1. **Import Errors**: Đảm bảo đã cài đặt tất cả dependencies
2. **Data Issues**: Kiểm tra format dữ liệu đầu vào
3. **Optimization Failures**: Điều chỉnh constraints
4. **Performance Issues**: Sử dụng caching và optimization

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode for detailed logs
system = QuantitativeTradingSystem(config)
```

## Performance Optimization

### Caching

```python
# Sử dụng caching cho tính toán nặng
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(data):
    # Expensive computation
    pass
```

### Parallel Processing

```python
# Sử dụng multiprocessing cho backtesting
from multiprocessing import Pool

def parallel_backtest(strategy_params):
    # Backtest logic
    pass

with Pool() as pool:
    results = pool.map(parallel_backtest, strategy_parameters)
```

## Integration với Trading Bot

### Signal Integration

```python
# Tích hợp với trading bot hiện tại
def enhanced_signal_generator(market_data):
    # Phân tích quantitative
    analysis = system.analyze_trading_opportunity(market_data)
    
    # Kết hợp với tín hiệu hiện tại
    base_signal = generate_base_signal(market_data)
    
    # Enhance với quantitative analysis
    enhanced_signal = combine_signals(base_signal, analysis['trading_recommendation'])
    
    return enhanced_signal
```

### Risk Integration

```python
# Tích hợp risk management
def risk_managed_order(order, portfolio):
    # Kiểm tra risk limits
    risk_check = system.risk_manager.calculate_risk_metrics(
        returns=portfolio.returns,
        signal_data=order,
        position_size=order.size
    )
    
    if risk_check['portfolio_risk'] > risk_check['risk_limit']:
        # Reduce position size
        order.size *= 0.5
    
    return order
```

Hệ thống Quantitative Trading này cung cấp một framework toàn diện cho việc phân tích và giao dịch định lượng, giúp nâng cao hiệu suất và giảm thiểu rủi ro cho trading bot của bạn. 