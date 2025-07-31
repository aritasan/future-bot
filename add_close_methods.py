#!/usr/bin/env python3
"""
Add close methods to all quantitative components
"""

import os
import re

def add_close_method_to_file(file_path: str, class_name: str):
    """Add close method to a specific class in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if close method already exists
        if f'async def close(self)' in content:
            print(f"✅ {file_path} already has close method")
            return
        
        # Find the last method in the class
        lines = content.split('\n')
        class_start = -1
        class_end = -1
        
        for i, line in enumerate(lines):
            if f'class {class_name}:' in line:
                class_start = i
            elif class_start != -1 and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                # Found end of class
                class_end = i
                break
        
        if class_start == -1:
            print(f"❌ Could not find class {class_name} in {file_path}")
            return
        
        if class_end == -1:
            class_end = len(lines)
        
        # Add close method before the end of class
        close_method = f"""    async def close(self) -> None:
        \"\"\"Close the {class_name.lower()} and cleanup resources.\"\"\"
        try:
            logger.info("Closing {class_name}...")
            
            # Clear any stored data
            if hasattr(self, 'analysis_cache'):
                self.analysis_cache.clear()
            if hasattr(self, 'history'):
                self.history.clear()
            if hasattr(self, 'metrics_history'):
                self.metrics_history.clear()
            
            logger.info("{class_name} closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing {class_name}: {{str(e)}}")
            raise
"""
        
        # Insert close method before the end of class
        lines.insert(class_end, close_method)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Added close method to {class_name} in {file_path}")
        
    except Exception as e:
        print(f"❌ Error adding close method to {file_path}: {str(e)}")

def main():
    """Add close methods to all quantitative components."""
    components = [
        ('src/quantitative/market_microstructure.py', 'MarketMicrostructureAnalyzer'),
        ('src/quantitative/backtesting_engine.py', 'AdvancedBacktestingEngine'),
        ('src/quantitative/factor_model.py', 'WorldQuantFactorModel'),
        ('src/quantitative/statistical_validator.py', 'StatisticalValidator'),
        ('src/quantitative/ml_ensemble.py', 'WorldQuantMLEnsemble'),
        ('src/quantitative/performance_tracker.py', 'WorldQuantPerformanceTracker'),
    ]
    
    print("🔧 Adding close methods to quantitative components...")
    
    for file_path, class_name in components:
        if os.path.exists(file_path):
            add_close_method_to_file(file_path, class_name)
        else:
            print(f"⚠️ File not found: {file_path}")
    
    print("✅ Completed adding close methods")

if __name__ == "__main__":
    main() 