import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FactorModel:
    """
    Advanced factor model for quantitative trading.
    Implements PCA-based factor analysis, multi-factor models, and factor attribution.
    """
    
    def __init__(self, n_factors: int = 5, min_eigenvalue: float = 1.0):
        self.n_factors = n_factors
        self.min_eigenvalue = min_eigenvalue
        self.factor_history = []
        self.scaler = StandardScaler()
        
    def build_factor_model(self, returns_data: pd.DataFrame, 
                          factor_data: pd.DataFrame = None) -> Dict:
        """
        Build comprehensive factor model.
        
        Args:
            returns_data: DataFrame of asset returns
            factor_data: DataFrame of factor returns (optional)
            
        Returns:
            Dict: Factor model results
        """
        try:
            # Convert returns dict to DataFrame if needed
            if isinstance(returns_data, dict):
                returns_df = pd.DataFrame(returns_data)
            else:
                returns_df = returns_data
            
            results = {}
            
            # PCA-based factor analysis
            pca_results = self._perform_pca_analysis(returns_df)
            if 'error' in pca_results:
                return pca_results  # Return error immediately
            results['pca_factors'] = pca_results
            
            # Multi-factor model
            if factor_data is not None:
                multifactor_results = self._build_multifactor_model(returns_df, factor_data)
                results['multifactor_model'] = multifactor_results
            
            # Factor attribution
            attribution_results = self._perform_factor_attribution(returns_df, pca_results)
            results['factor_attribution'] = attribution_results
            
            # Risk decomposition
            risk_results = self._decompose_risk(returns_df, pca_results)
            results['risk_decomposition'] = risk_results
            
            # Factor timing analysis
            timing_results = self._analyze_factor_timing(returns_df, pca_results)
            results['factor_timing'] = timing_results
            
            self._store_factor_result(results)
            return results
            
        except Exception as e:
            logger.error(f"Error building factor model: {str(e)}")
            return {'error': str(e)}
    
    def _perform_pca_analysis(self, returns_data: pd.DataFrame) -> Dict:
        """Perform PCA-based factor analysis."""
        try:
            # Ensure returns_data is a DataFrame
            if not isinstance(returns_data, pd.DataFrame):
                if isinstance(returns_data, dict):
                    returns_data = pd.DataFrame(returns_data)
                elif isinstance(returns_data, list):
                    return {'error': 'Returns data is a list, expected DataFrame'}
                else:
                    return {'error': f'Invalid returns data type: {type(returns_data)}'}
            
            # Check if we have valid data
            if returns_data.empty or len(returns_data.columns) == 0:
                return {'error': 'No valid returns data for PCA analysis'}
            
            # Check if we have enough data points
            if len(returns_data) < 30:
                return {'error': 'Insufficient data points for PCA analysis (need at least 30)'}
            
            # Check if we have enough assets
            if len(returns_data.columns) < 2:
                return {'error': 'Insufficient assets for PCA analysis (need at least 2)'}
            
            # Check for NaN values
            if returns_data.isnull().any().any():
                # Fill NaN values with forward fill then backward fill
                returns_data = returns_data.fillna(method='ffill').fillna(method='bfill')
                if returns_data.isnull().any().any():
                    return {'error': 'Too many NaN values in returns data'}
            
            # Standardize returns
            returns_standardized = self.scaler.fit_transform(returns_data)
            
            # Perform PCA
            pca = PCA(n_components=min(self.n_factors, len(returns_data.columns)))
            pca_factors = pca.fit_transform(returns_standardized)
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Filter factors based on eigenvalue threshold
            significant_factors = []
            for i, eigenvalue in enumerate(pca.explained_variance_):
                if eigenvalue >= self.min_eigenvalue:
                    significant_factors.append({
                        'factor_id': i + 1,
                        'eigenvalue': eigenvalue,
                        'explained_variance': explained_variance[i],
                        'cumulative_variance': cumulative_variance[i],
                        'loadings': pca.components_[i]
                    })
            
            # Calculate factor returns
            factor_returns = pd.DataFrame(
                pca_factors[:, :len(significant_factors)],
                index=returns_data.index,
                columns=[f'Factor_{i+1}' for i in range(len(significant_factors))]
            )
            
            return {
                'n_factors': len(significant_factors),
                'significant_factors': significant_factors,
                'factor_returns': factor_returns,
                'explained_variance': explained_variance,
                'cumulative_variance': cumulative_variance,
                'total_explained_variance': cumulative_variance[-1] if len(cumulative_variance) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in PCA analysis: {str(e)}")
            return {'error': str(e)}
    
    def _build_multifactor_model(self, returns_data: pd.DataFrame, 
                                factor_data: pd.DataFrame) -> Dict:
        """Build multi-factor model with predefined factors."""
        try:
            results = {}
            
            # Run regression for each asset
            factor_loadings = {}
            factor_exposures = {}
            r_squared_values = {}
            
            for asset in returns_data.columns:
                # Prepare data
                asset_returns = returns_data[asset]
                factor_returns = factor_data
                
                # Run regression
                model = LinearRegression()
                model.fit(factor_returns, asset_returns)
                
                # Store results
                factor_loadings[asset] = model.coef_
                factor_exposures[asset] = dict(zip(factor_returns.columns, model.coef_))
                r_squared_values[asset] = model.score(factor_returns, asset_returns)
            
            # Calculate factor statistics
            factor_stats = {}
            for factor in factor_data.columns:
                loadings = [factor_exposures[asset][factor] for asset in returns_data.columns]
                factor_stats[factor] = {
                    'mean_loading': np.mean(loadings),
                    'std_loading': np.std(loadings),
                    'min_loading': np.min(loadings),
                    'max_loading': np.max(loadings)
                }
            
            return {
                'factor_loadings': factor_loadings,
                'factor_exposures': factor_exposures,
                'r_squared_values': r_squared_values,
                'factor_statistics': factor_stats,
                'average_r_squared': np.mean(list(r_squared_values.values()))
            }
            
        except Exception as e:
            logger.error(f"Error building multi-factor model: {str(e)}")
            return {'error': str(e)}
    
    def _perform_factor_attribution(self, returns_data: pd.DataFrame, 
                                   pca_results: Dict) -> Dict:
        """Perform factor attribution analysis."""
        try:
            if 'error' in pca_results:
                return {'error': pca_results['error']}
            
            factor_returns = pca_results['factor_returns']
            significant_factors = pca_results['significant_factors']
            
            attribution_results = {}
            
            for asset in returns_data.columns:
                asset_returns = returns_data[asset]
                
                # Calculate factor contributions
                factor_contributions = {}
                total_contribution = 0
                
                for factor_info in significant_factors:
                    factor_id = factor_info['factor_id']
                    factor_name = f'Factor_{factor_id}'
                    factor_return = factor_returns[factor_name]
                    loading = factor_info['loadings'][returns_data.columns.get_loc(asset)]
                    
                    contribution = loading * factor_return
                    factor_contributions[factor_name] = contribution
                    total_contribution += contribution
                
                # Calculate residual
                residual = asset_returns - total_contribution
                
                attribution_results[asset] = {
                    'factor_contributions': factor_contributions,
                    'total_factor_contribution': total_contribution,
                    'residual': residual,
                    'explained_variance': 1 - (residual.var() / asset_returns.var()) if asset_returns.var() > 0 else 0
                }
            
            return attribution_results
            
        except Exception as e:
            logger.error(f"Error in factor attribution: {str(e)}")
            return {'error': str(e)}
    
    def _decompose_risk(self, returns_data: pd.DataFrame, 
                        pca_results: Dict) -> Dict:
        """Decompose portfolio risk into factor contributions."""
        try:
            if 'error' in pca_results:
                return {'error': pca_results['error']}
            
            factor_returns = pca_results['factor_returns']
            significant_factors = pca_results['significant_factors']
            
            # Calculate factor covariance matrix
            factor_cov = factor_returns.cov()
            
            # Calculate portfolio weights (equal weight for simplicity)
            n_assets = len(returns_data.columns)
            portfolio_weights = np.array([1/n_assets] * n_assets)
            
            # Calculate factor exposures for portfolio
            portfolio_factor_exposures = {}
            for factor_info in significant_factors:
                factor_id = factor_info['factor_id']
                factor_name = f'Factor_{factor_id}'
                loadings = factor_info['loadings']
                
                # Portfolio exposure to this factor
                exposure = np.dot(portfolio_weights, loadings)
                portfolio_factor_exposures[factor_name] = exposure
            
            # Calculate factor risk contributions
            factor_risk_contributions = {}
            total_factor_risk = 0
            
            for factor_name, exposure in portfolio_factor_exposures.items():
                # Factor variance contribution
                factor_variance = factor_cov.loc[factor_name, factor_name]
                risk_contribution = (exposure ** 2) * factor_variance
                factor_risk_contributions[factor_name] = {
                    'exposure': exposure,
                    'variance': factor_variance,
                    'risk_contribution': risk_contribution
                }
                total_factor_risk += risk_contribution
            
            # Calculate specific risk (residual)
            portfolio_returns = returns_data.dot(portfolio_weights)
            total_variance = portfolio_returns.var()
            specific_risk = total_variance - total_factor_risk
            
            return {
                'portfolio_factor_exposures': portfolio_factor_exposures,
                'factor_risk_contributions': factor_risk_contributions,
                'total_factor_risk': total_factor_risk,
                'specific_risk': specific_risk,
                'total_risk': total_variance,
                'factor_risk_ratio': total_factor_risk / total_variance if total_variance > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in risk decomposition: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_factor_timing(self, returns_data: pd.DataFrame, 
                              pca_results: Dict) -> Dict:
        """Analyze factor timing opportunities."""
        try:
            if 'error' in pca_results:
                return {'error': pca_results['error']}
            
            factor_returns = pca_results['factor_returns']
            
            timing_results = {}
            
            for factor_name in factor_returns.columns:
                factor_series = factor_returns[factor_name]
                
                # Calculate factor momentum
                momentum_1m = factor_series.rolling(window=21).mean()
                momentum_3m = factor_series.rolling(window=63).mean()
                momentum_6m = factor_series.rolling(window=126).mean()
                
                # Calculate factor volatility
                volatility = factor_series.rolling(window=21).std()
                
                # Calculate factor regime indicators
                regime_indicator = np.where(factor_series > factor_series.rolling(window=252).mean(), 1, -1)
                
                # Calculate factor autocorrelation
                autocorr_1d = factor_series.autocorr(lag=1)
                autocorr_5d = factor_series.autocorr(lag=5)
                
                timing_results[factor_name] = {
                    'momentum_1m': momentum_1m.iloc[-1] if len(momentum_1m) > 0 else 0,
                    'momentum_3m': momentum_3m.iloc[-1] if len(momentum_3m) > 0 else 0,
                    'momentum_6m': momentum_6m.iloc[-1] if len(momentum_6m) > 0 else 0,
                    'volatility': volatility.iloc[-1] if len(volatility) > 0 else 0,
                    'regime_indicator': regime_indicator[-1] if len(regime_indicator) > 0 else 0,
                    'autocorr_1d': autocorr_1d,
                    'autocorr_5d': autocorr_5d,
                    'current_value': factor_series.iloc[-1] if len(factor_series) > 0 else 0
                }
            
            return timing_results
            
        except Exception as e:
            logger.error(f"Error in factor timing analysis: {str(e)}")
            return {'error': str(e)}
    
    def generate_factor_signals(self, factor_timing_results: Dict) -> Dict:
        """Generate trading signals based on factor timing analysis."""
        try:
            signals = {}
            
            for factor_name, timing_data in factor_timing_results.items():
                if 'error' in timing_data:
                    continue
                
                signal_strength = 0
                signal_reasons = []
                
                # Momentum-based signals
                if timing_data['momentum_1m'] > 0:
                    signal_strength += 0.3
                    signal_reasons.append('positive_1m_momentum')
                elif timing_data['momentum_1m'] < 0:
                    signal_strength -= 0.3
                    signal_reasons.append('negative_1m_momentum')
                
                if timing_data['momentum_3m'] > 0:
                    signal_strength += 0.2
                    signal_reasons.append('positive_3m_momentum')
                elif timing_data['momentum_3m'] < 0:
                    signal_strength -= 0.2
                    signal_reasons.append('negative_3m_momentum')
                
                # Volatility-based signals
                if timing_data['volatility'] < timing_data['volatility'] * 0.8:  # Low volatility
                    signal_strength += 0.1
                    signal_reasons.append('low_volatility')
                elif timing_data['volatility'] > timing_data['volatility'] * 1.2:  # High volatility
                    signal_strength -= 0.1
                    signal_reasons.append('high_volatility')
                
                # Regime-based signals
                if timing_data['regime_indicator'] > 0:
                    signal_strength += 0.2
                    signal_reasons.append('positive_regime')
                else:
                    signal_strength -= 0.2
                    signal_reasons.append('negative_regime')
                
                # Autocorrelation-based signals
                if timing_data['autocorr_1d'] > 0.1:
                    signal_strength += 0.1
                    signal_reasons.append('positive_autocorr')
                elif timing_data['autocorr_1d'] < -0.1:
                    signal_strength -= 0.1
                    signal_reasons.append('negative_autocorr')
                
                signals[factor_name] = {
                    'signal_strength': signal_strength,
                    'signal_direction': 'long' if signal_strength > 0.1 else 'short' if signal_strength < -0.1 else 'neutral',
                    'signal_reasons': signal_reasons,
                    'confidence': min(abs(signal_strength), 1.0)
                }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating factor signals: {str(e)}")
            return {'error': str(e)}
    
    def _store_factor_result(self, result: Dict):
        """Store factor analysis result in history."""
        self.factor_history.append({
            'timestamp': pd.Timestamp.now(),
            'result': result
        })
    
    def get_factor_summary(self) -> Dict:
        """Get summary of all factor analysis results."""
        if not self.factor_history:
            return {'message': 'No factor analysis history available'}
        
        summary = {
            'total_analyses': len(self.factor_history),
            'average_factors': [],
            'average_explained_variance': [],
            'recent_results': []
        }
        
        for record in self.factor_history:
            result = record['result']
            
            if 'pca_factors' in result and 'n_factors' in result['pca_factors']:
                summary['average_factors'].append(result['pca_factors']['n_factors'])
            
            if 'pca_factors' in result and 'total_explained_variance' in result['pca_factors']:
                summary['average_explained_variance'].append(result['pca_factors']['total_explained_variance'])
        
        if summary['average_factors']:
            summary['avg_n_factors'] = np.mean(summary['average_factors'])
        
        if summary['average_explained_variance']:
            summary['avg_explained_variance'] = np.mean(summary['average_explained_variance'])
        
        # Get recent results
        summary['recent_results'] = self.factor_history[-5:]
        
        return summary 