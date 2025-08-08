#!/usr/bin/env python3
"""
WorldQuant Standards Validation System
Implements rigorous validation with 85% confidence, 15% risk max requirements.
Multi-layer validation: statistical, market regime, factor model.
"""

import asyncio
import logging
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import traceback

logger = logging.getLogger(__name__)

class ValidationLayer(Enum):
    """Validation layers for WorldQuant standards."""
    STATISTICAL = "statistical"
    MARKET_REGIME = "market_regime"
    FACTOR_MODEL = "factor_model"
    RISK_MANAGEMENT = "risk_management"
    MACHINE_LEARNING = "machine_learning"

class MarketRegime(Enum):
    """Market regime types."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    NORMAL = "normal"
    CRISIS = "crisis"

@dataclass
class ValidationResult:
    """Structured validation result."""
    is_valid: bool
    confidence_score: float
    risk_score: float
    worldquant_compliance: bool
    validation_details: Dict[str, Any]
    layer_results: Dict[str, Dict]
    warnings: List[str]
    timestamp: datetime

class WorldQuantValidationSystem:
    """
    WorldQuant Standards Validation System
    
    Requirements:
    - Minimum 85% confidence
    - Maximum 15% risk
    - Multi-layer validation
    - Statistical significance
    - Market regime compatibility
    - Factor model validation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.validation_history = deque(maxlen=10000)
        self.performance_metrics = {
            'validation_success_rate': 0.0,
            'avg_confidence_score': 0.0,
            'avg_risk_score': 0.0,
            'worldquant_compliance_rate': 0.0
        }
        
        # WorldQuant standards thresholds
        self.worldquant_thresholds = {
            'min_confidence': 0.85,      # 85% minimum confidence
            'max_risk': 0.15,            # 15% maximum risk
            'min_statistical_significance': 0.05,  # p < 0.05
            'min_sample_size': 30,       # Minimum sample size
            'max_factor_exposure': 0.3,  # Maximum 30% factor exposure
            'min_sharpe_ratio': 0.5,     # Minimum Sharpe ratio
            'max_drawdown': 0.15         # Maximum 15% drawdown
        }
        
        # Layer-specific validation criteria
        self.layer_criteria = {
            ValidationLayer.STATISTICAL: {
                'min_p_value': 0.05,
                'min_t_stat': 2.0,
                'min_sample_size': 30,
                'min_effect_size': 0.2
            },
            ValidationLayer.MARKET_REGIME: {
                'regime_compatibility': True,
                'volatility_adjustment': True,
                'correlation_threshold': 0.7
            },
            ValidationLayer.FACTOR_MODEL: {
                'max_factor_exposure': 0.3,
                'factor_neutrality': True,
                'risk_attribution': True
            },
            ValidationLayer.RISK_MANAGEMENT: {
                'max_var_95': 0.02,      # 2% VaR at 95% confidence
                'max_expected_shortfall': 0.03,  # 3% Expected Shortfall
                'max_leverage': 2.0,     # Maximum 2x leverage
                'position_size_limit': 0.1  # Maximum 10% position size
            },
            ValidationLayer.MACHINE_LEARNING: {
                'min_model_agreement': 0.7,  # 70% model agreement
                'min_prediction_confidence': 0.8,  # 80% prediction confidence
                'max_model_uncertainty': 0.2  # 20% maximum uncertainty
            }
        }
        
        logger.info("WorldQuant Validation System initialized with strict standards")
    
    async def validate_signal_worldquant(self, signal: Dict, market_data: Dict) -> ValidationResult:
        """
        Comprehensive WorldQuant standards validation.
        
        Args:
            signal: Trading signal to validate
            market_data: Market data for validation
            
        Returns:
            ValidationResult with detailed validation results
        """
        try:
            validation_start = datetime.now()
            
            # Initialize validation result
            validation_result = ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                risk_score=1.0,
                worldquant_compliance=False,
                validation_details={},
                layer_results={},
                warnings=[],
                timestamp=validation_start
            )
            
            # Multi-layer validation
            layer_results = {}
            
            # 1. Statistical Validation
            statistical_result = await self._validate_statistical_layer(signal, market_data)
            layer_results[ValidationLayer.STATISTICAL.value] = statistical_result
            
            # 2. Market Regime Validation
            regime_result = await self._validate_market_regime_layer(signal, market_data)
            layer_results[ValidationLayer.MARKET_REGIME.value] = regime_result
            
            # 3. Factor Model Validation
            factor_result = await self._validate_factor_model_layer(signal, market_data)
            layer_results[ValidationLayer.FACTOR_MODEL.value] = factor_result
            
            # 4. Risk Management Validation
            risk_result = await self._validate_risk_management_layer(signal, market_data)
            layer_results[ValidationLayer.RISK_MANAGEMENT.value] = risk_result
            
            # 5. Machine Learning Validation
            ml_result = await self._validate_machine_learning_layer(signal, market_data)
            layer_results[ValidationLayer.MACHINE_LEARNING.value] = ml_result
            
            # Calculate composite scores
            confidence_score = self._calculate_composite_confidence(layer_results)
            risk_score = self._calculate_composite_risk(layer_results)
            
            # Check WorldQuant compliance
            worldquant_compliance = self._check_worldquant_compliance(
                confidence_score, risk_score, layer_results
            )
            
            # Generate warnings
            warnings = self._generate_validation_warnings(layer_results)
            
            # Update validation result
            validation_result.confidence_score = confidence_score
            validation_result.risk_score = risk_score
            validation_result.worldquant_compliance = worldquant_compliance
            validation_result.is_valid = worldquant_compliance
            validation_result.layer_results = layer_results
            validation_result.warnings = warnings
            
            # Store validation history
            self._store_validation_history(validation_result)
            
            # Update performance metrics
            self._update_performance_metrics(validation_result)
            
            validation_time = (datetime.now() - validation_start).total_seconds()
            logger.info(f"WorldQuant validation completed in {validation_time:.3f}s - "
                       f"Confidence: {confidence_score:.3f}, Risk: {risk_score:.3f}, "
                       f"Compliance: {'✅' if worldquant_compliance else '❌'}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in WorldQuant validation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return failed validation result
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                risk_score=1.0,
                worldquant_compliance=False,
                validation_details={'error': str(e)},
                layer_results={},
                warnings=[f'Validation error: {str(e)}'],
                timestamp=datetime.now()
            )
    
    async def _validate_statistical_layer(self, signal: Dict, market_data: Dict) -> Dict:
        """Validate statistical significance and quality."""
        try:
            result = {
                'is_valid': False,
                'confidence_score': 0.0,
                'risk_score': 1.0,
                'details': {}
            }
            
            # Extract statistical measures
            p_value = signal.get('p_value', 1.0)
            t_stat = signal.get('t_statistic', 0.0)
            sample_size = signal.get('sample_size', 0)
            effect_size = signal.get('effect_size', 0.0)
            
            # Check statistical significance
            is_significant = (
                p_value < self.layer_criteria[ValidationLayer.STATISTICAL]['min_p_value'] and
                abs(t_stat) > self.layer_criteria[ValidationLayer.STATISTICAL]['min_t_stat'] and
                sample_size >= self.layer_criteria[ValidationLayer.STATISTICAL]['min_sample_size'] and
                effect_size >= self.layer_criteria[ValidationLayer.STATISTICAL]['min_effect_size']
            )
            
            # Calculate confidence score based on statistical measures
            p_value_score = max(0, 1 - (p_value / 0.05))  # Higher score for lower p-value
            t_stat_score = min(1, abs(t_stat) / 3.0)  # Normalize t-statistic
            sample_size_score = min(1, sample_size / 100)  # Normalize sample size
            effect_size_score = min(1, effect_size / 0.5)  # Normalize effect size
            
            # Weighted statistical confidence
            statistical_confidence = (
                p_value_score * 0.4 +
                t_stat_score * 0.3 +
                sample_size_score * 0.2 +
                effect_size_score * 0.1
            )
            
            result.update({
                'is_valid': is_significant,
                'confidence_score': statistical_confidence,
                'risk_score': 1.0 - statistical_confidence,
                'details': {
                    'p_value': p_value,
                    't_statistic': t_stat,
                    'sample_size': sample_size,
                    'effect_size': effect_size,
                    'is_significant': is_significant
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in statistical validation: {str(e)}")
            return {'is_valid': False, 'confidence_score': 0.0, 'risk_score': 1.0, 'details': {'error': str(e)}}
    
    async def _validate_market_regime_layer(self, signal: Dict, market_data: Dict) -> Dict:
        """Validate market regime compatibility."""
        try:
            result = {
                'is_valid': False,
                'confidence_score': 0.0,
                'risk_score': 1.0,
                'details': {}
            }
            
            # Detect market regime
            market_regime = self._detect_market_regime(market_data)
            signal_type = signal.get('signal_type', 'unknown')
            
            # Regime compatibility rules
            compatibility_rules = {
                MarketRegime.TRENDING: ['trend_following', 'momentum', 'breakout'],
                MarketRegime.MEAN_REVERTING: ['mean_reversion', 'contrarian', 'reversal'],
                MarketRegime.VOLATILE: ['volatility', 'breakout', 'adaptive'],
                MarketRegime.NORMAL: ['all'],
                MarketRegime.CRISIS: ['defensive', 'hedge', 'risk_off']
            }
            
            # Check compatibility
            allowed_types = compatibility_rules.get(market_regime, ['all'])
            is_compatible = signal_type in allowed_types or 'all' in allowed_types
            
            # Calculate regime-adjusted confidence
            volatility = market_data.get('volatility', 0.02)
            correlation = market_data.get('correlation', 0.5)
            
            # Volatility adjustment
            volatility_factor = max(0.5, 1.0 - (volatility - 0.02) * 10)
            
            # Correlation adjustment
            correlation_factor = 1.0 - abs(correlation - 0.5) * 0.5
            
            # Regime-specific confidence boost
            regime_confidence_boost = {
                MarketRegime.TRENDING: 0.1,
                MarketRegime.MEAN_REVERTING: 0.05,
                MarketRegime.VOLATILE: -0.1,
                MarketRegime.NORMAL: 0.0,
                MarketRegime.CRISIS: -0.2
            }.get(market_regime, 0.0)
            
            regime_confidence = (
                (1.0 if is_compatible else 0.5) * 0.6 +
                volatility_factor * 0.2 +
                correlation_factor * 0.1 +
                regime_confidence_boost * 0.1
            )
            
            result.update({
                'is_valid': is_compatible,
                'confidence_score': regime_confidence,
                'risk_score': 1.0 - regime_confidence,
                'details': {
                    'market_regime': market_regime.value,
                    'signal_type': signal_type,
                    'is_compatible': is_compatible,
                    'volatility': volatility,
                    'correlation': correlation
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in market regime validation: {str(e)}")
            return {'is_valid': False, 'confidence_score': 0.0, 'risk_score': 1.0, 'details': {'error': str(e)}}
    
    async def _validate_factor_model_layer(self, signal: Dict, market_data: Dict) -> Dict:
        """Validate factor model exposures and neutrality."""
        try:
            result = {
                'is_valid': False,
                'confidence_score': 0.0,
                'risk_score': 1.0,
                'details': {}
            }
            
            # Get factor exposures
            factor_exposures = signal.get('factor_exposures', {})
            
            # Check factor exposure limits
            max_exposure = self.layer_criteria[ValidationLayer.FACTOR_MODEL]['max_factor_exposure']
            exposure_violations = []
            
            for factor, exposure in factor_exposures.items():
                if abs(exposure) > max_exposure:
                    exposure_violations.append(f"{factor}: {exposure:.3f}")
            
            # Calculate factor neutrality score
            total_exposure = sum(abs(exposure) for exposure in factor_exposures.values())
            neutrality_score = max(0, 1.0 - (total_exposure / 1.0))  # Perfect neutrality = 1.0
            
            # Calculate factor-adjusted confidence
            factor_confidence = (
                (1.0 if not exposure_violations else 0.5) * 0.7 +
                neutrality_score * 0.3
            )
            
            result.update({
                'is_valid': len(exposure_violations) == 0,
                'confidence_score': factor_confidence,
                'risk_score': 1.0 - factor_confidence,
                'details': {
                    'factor_exposures': factor_exposures,
                    'exposure_violations': exposure_violations,
                    'neutrality_score': neutrality_score,
                    'total_exposure': total_exposure
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in factor model validation: {str(e)}")
            return {'is_valid': False, 'confidence_score': 0.0, 'risk_score': 1.0, 'details': {'error': str(e)}}
    
    async def _validate_risk_management_layer(self, signal: Dict, market_data: Dict) -> Dict:
        """Validate risk management parameters."""
        try:
            result = {
                'is_valid': False,
                'confidence_score': 0.0,
                'risk_score': 1.0,
                'details': {}
            }
            
            # Extract risk metrics
            var_95 = signal.get('var_95', 0.05)
            expected_shortfall = signal.get('expected_shortfall', 0.08)
            leverage = signal.get('leverage', 1.0)
            position_size = signal.get('position_size', 0.01)
            
            # Check risk limits
            risk_criteria = self.layer_criteria[ValidationLayer.RISK_MANAGEMENT]
            
            var_ok = var_95 <= risk_criteria['max_var_95']
            es_ok = expected_shortfall <= risk_criteria['max_expected_shortfall']
            leverage_ok = leverage <= risk_criteria['max_leverage']
            position_ok = position_size <= risk_criteria['position_size_limit']
            
            # Calculate risk score
            risk_score = (
                (var_95 / risk_criteria['max_var_95']) * 0.3 +
                (expected_shortfall / risk_criteria['max_expected_shortfall']) * 0.3 +
                (leverage / risk_criteria['max_leverage']) * 0.2 +
                (position_size / risk_criteria['position_size_limit']) * 0.2
            )
            
            risk_confidence = max(0, 1.0 - risk_score)
            
            result.update({
                'is_valid': var_ok and es_ok and leverage_ok and position_ok,
                'confidence_score': risk_confidence,
                'risk_score': risk_score,
                'details': {
                    'var_95': var_95,
                    'expected_shortfall': expected_shortfall,
                    'leverage': leverage,
                    'position_size': position_size,
                    'risk_limits_ok': {
                        'var_ok': var_ok,
                        'es_ok': es_ok,
                        'leverage_ok': leverage_ok,
                        'position_ok': position_ok
                    }
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in risk management validation: {str(e)}")
            return {'is_valid': False, 'confidence_score': 0.0, 'risk_score': 1.0, 'details': {'error': str(e)}}
    
    async def _validate_machine_learning_layer(self, signal: Dict, market_data: Dict) -> Dict:
        """Validate machine learning predictions and model agreement."""
        try:
            result = {
                'is_valid': False,
                'confidence_score': 0.0,
                'risk_score': 1.0,
                'details': {}
            }
            
            # Extract ML metrics
            model_agreement = signal.get('model_agreement', 0.5)
            prediction_confidence = signal.get('prediction_confidence', 0.5)
            model_uncertainty = signal.get('model_uncertainty', 0.5)
            
            # Check ML criteria
            ml_criteria = self.layer_criteria[ValidationLayer.MACHINE_LEARNING]
            
            agreement_ok = model_agreement >= ml_criteria['min_model_agreement']
            confidence_ok = prediction_confidence >= ml_criteria['min_prediction_confidence']
            uncertainty_ok = model_uncertainty <= ml_criteria['max_model_uncertainty']
            
            # Calculate ML confidence
            ml_confidence = (
                (model_agreement / ml_criteria['min_model_agreement']) * 0.4 +
                (prediction_confidence / ml_criteria['min_prediction_confidence']) * 0.4 +
                (1.0 - model_uncertainty / ml_criteria['max_model_uncertainty']) * 0.2
            )
            
            # Ensure confidence is between 0 and 1
            ml_confidence = max(0.0, min(1.0, ml_confidence))
            
            result.update({
                'is_valid': agreement_ok and confidence_ok and uncertainty_ok,
                'confidence_score': ml_confidence,
                'risk_score': 1.0 - ml_confidence,
                'details': {
                    'model_agreement': model_agreement,
                    'prediction_confidence': prediction_confidence,
                    'model_uncertainty': model_uncertainty,
                    'ml_criteria_ok': {
                        'agreement_ok': agreement_ok,
                        'confidence_ok': confidence_ok,
                        'uncertainty_ok': uncertainty_ok
                    }
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in machine learning validation: {str(e)}")
            return {'is_valid': False, 'confidence_score': 0.0, 'risk_score': 1.0, 'details': {'error': str(e)}}
    
    def _calculate_composite_confidence(self, layer_results: Dict) -> float:
        """Calculate composite confidence score from all layers."""
        try:
            confidence_scores = []
            weights = {
                ValidationLayer.STATISTICAL.value: 0.3,
                ValidationLayer.MARKET_REGIME.value: 0.25,
                ValidationLayer.FACTOR_MODEL.value: 0.2,
                ValidationLayer.RISK_MANAGEMENT.value: 0.15,
                ValidationLayer.MACHINE_LEARNING.value: 0.1
            }
            
            for layer, weight in weights.items():
                if layer in layer_results:
                    layer_confidence = layer_results[layer].get('confidence_score', 0.0)
                    confidence_scores.append(layer_confidence * weight)
            
            composite_confidence = sum(confidence_scores) if confidence_scores else 0.0
            return min(1.0, max(0.0, composite_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating composite confidence: {str(e)}")
            return 0.0
    
    def _calculate_composite_risk(self, layer_results: Dict) -> float:
        """Calculate composite risk score from all layers."""
        try:
            risk_scores = []
            weights = {
                ValidationLayer.STATISTICAL.value: 0.3,
                ValidationLayer.MARKET_REGIME.value: 0.25,
                ValidationLayer.FACTOR_MODEL.value: 0.2,
                ValidationLayer.RISK_MANAGEMENT.value: 0.15,
                ValidationLayer.MACHINE_LEARNING.value: 0.1
            }
            
            for layer, weight in weights.items():
                if layer in layer_results:
                    layer_risk = layer_results[layer].get('risk_score', 1.0)
                    # Ensure risk score is between 0 and 1
                    layer_risk = max(0.0, min(1.0, layer_risk))
                    risk_scores.append(layer_risk * weight)
            
            composite_risk = sum(risk_scores) if risk_scores else 1.0
            return min(1.0, max(0.0, composite_risk))
            
        except Exception as e:
            logger.error(f"Error calculating composite risk: {str(e)}")
            return 1.0
    
    def _check_worldquant_compliance(self, confidence_score: float, risk_score: float, 
                                   layer_results: Dict) -> bool:
        """Check if signal meets WorldQuant standards."""
        try:
            # Check confidence threshold
            confidence_ok = confidence_score >= self.worldquant_thresholds['min_confidence']
            
            # Check risk threshold
            risk_ok = risk_score <= self.worldquant_thresholds['max_risk']
            
            # Check all layers are valid
            layers_ok = all(
                layer_result.get('is_valid', False)
                for layer_result in layer_results.values()
            )
            
            # WorldQuant compliance requires all conditions
            worldquant_compliance = confidence_ok and risk_ok and layers_ok
            
            return worldquant_compliance
            
        except Exception as e:
            logger.error(f"Error checking WorldQuant compliance: {str(e)}")
            return False
    
    def _generate_validation_warnings(self, layer_results: Dict) -> List[str]:
        """Generate warnings for validation issues."""
        warnings = []
        
        try:
            for layer, result in layer_results.items():
                if not result.get('is_valid', False):
                    warnings.append(f"{layer} validation failed")
                
                details = result.get('details', {})
                
                # Statistical warnings
                if layer == ValidationLayer.STATISTICAL.value:
                    p_value = details.get('p_value', 1.0)
                    if p_value >= 0.05:
                        warnings.append(f"Statistical significance insufficient (p={p_value:.3f})")
                
                # Market regime warnings
                elif layer == ValidationLayer.MARKET_REGIME.value:
                    if not details.get('is_compatible', False):
                        warnings.append(f"Signal incompatible with market regime")
                
                # Factor model warnings
                elif layer == ValidationLayer.FACTOR_MODEL.value:
                    violations = details.get('exposure_violations', [])
                    if violations:
                        warnings.append(f"Factor exposure violations: {', '.join(violations)}")
                
                # Risk management warnings
                elif layer == ValidationLayer.RISK_MANAGEMENT.value:
                    risk_limits = details.get('risk_limits_ok', {})
                    for limit, ok in risk_limits.items():
                        if not ok:
                            warnings.append(f"Risk limit exceeded: {limit}")
                
                # ML warnings
                elif layer == ValidationLayer.MACHINE_LEARNING.value:
                    ml_criteria = details.get('ml_criteria_ok', {})
                    for criterion, ok in ml_criteria.items():
                        if not ok:
                            warnings.append(f"ML criterion failed: {criterion}")
            
        except Exception as e:
            logger.error(f"Error generating validation warnings: {str(e)}")
            warnings.append(f"Warning generation error: {str(e)}")
        
        return warnings
    
    def _detect_market_regime(self, market_data: Dict) -> MarketRegime:
        """Detect current market regime."""
        try:
            volatility = market_data.get('volatility', 0.02)
            correlation = market_data.get('correlation', 0.5)
            trend_strength = market_data.get('trend_strength', 0.5)
            
            # Regime detection logic
            if volatility > 0.05:  # High volatility
                return MarketRegime.VOLATILE
            elif trend_strength > 0.7:  # Strong trend
                return MarketRegime.TRENDING
            elif correlation > 0.8:  # High correlation (crisis)
                return MarketRegime.CRISIS
            elif trend_strength < 0.3:  # Weak trend (mean reverting)
                return MarketRegime.MEAN_REVERTING
            else:
                return MarketRegime.NORMAL
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return MarketRegime.NORMAL
    
    def _store_validation_history(self, validation_result: ValidationResult) -> None:
        """Store validation result in history."""
        try:
            self.validation_history.append({
                'timestamp': validation_result.timestamp,
                'confidence_score': validation_result.confidence_score,
                'risk_score': validation_result.risk_score,
                'worldquant_compliance': validation_result.worldquant_compliance,
                'is_valid': validation_result.is_valid,
                'warnings': validation_result.warnings
            })
        except Exception as e:
            logger.error(f"Error storing validation history: {str(e)}")
    
    def _update_performance_metrics(self, validation_result: ValidationResult) -> None:
        """Update performance metrics."""
        try:
            if len(self.validation_history) < 10:
                return
            
            recent_validations = list(self.validation_history)[-100:]
            
            # Calculate metrics
            success_count = sum(1 for v in recent_validations if v['is_valid'])
            compliance_count = sum(1 for v in recent_validations if v['worldquant_compliance'])
            
            self.performance_metrics['validation_success_rate'] = success_count / len(recent_validations)
            self.performance_metrics['worldquant_compliance_rate'] = compliance_count / len(recent_validations)
            self.performance_metrics['avg_confidence_score'] = sum(v['confidence_score'] for v in recent_validations) / len(recent_validations)
            self.performance_metrics['avg_risk_score'] = sum(v['risk_score'] for v in recent_validations) / len(recent_validations)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation system summary."""
        try:
            return {
                'worldquant_thresholds': self.worldquant_thresholds,
                'performance_metrics': self.performance_metrics,
                'validation_history_size': len(self.validation_history),
                'layer_criteria': {layer.value: criteria for layer, criteria in self.layer_criteria.items()}
            }
        except Exception as e:
            logger.error(f"Error getting validation summary: {str(e)}")
            return {}
    
    async def close(self) -> None:
        """Close validation system and cleanup resources."""
        try:
            logger.info("Closing WorldQuant Validation System...")
            self.validation_history.clear()
            logger.info("WorldQuant Validation System closed successfully")
        except Exception as e:
            logger.error(f"Error closing WorldQuant Validation System: {str(e)}")
