#!/usr/bin/env python3
"""
Update main_with_quantitative.py with proper logging configuration
"""

import re
import os

def update_main_with_logging():
    """Update main_with_quantitative.py to use proper logging configuration."""
    
    # Read the current main_with_quantitative.py
    try:
        with open('main_with_quantitative.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ main_with_quantitative.py not found")
        return False
    
    # Check if logging config is already imported
    if 'from src.utils.logging_config import setup_logging' in content:
        print("✅ Logging configuration already imported")
        return True
    
    # Add logging configuration import at the top
    logging_import = '''import logging
import sys
from pathlib import Path

# Setup logging configuration
from src.utils.logging_config import setup_logging
setup_logging()

# Disable werkzeug logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('dash').setLevel(logging.ERROR)
logging.getLogger('dash.dash').setLevel(logging.ERROR)
logging.getLogger('flask').setLevel(logging.ERROR)
logging.getLogger('aiohttp.access').setLevel(logging.ERROR)
logging.getLogger('websockets.server').setLevel(logging.WARNING)

'''
    
    # Find the first import statement
    import_pattern = r'^(import .*?)$'
    imports = re.findall(import_pattern, content, re.MULTILINE)
    
    if imports:
        # Insert logging config after the first import
        first_import_end = content.find(imports[0]) + len(imports[0])
        new_content = content[:first_import_end] + '\n' + logging_import + content[first_import_end:]
    else:
        # If no imports found, add at the beginning
        new_content = logging_import + content
    
    # Write the updated content
    try:
        with open('main_with_quantitative.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Updated main_with_quantitative.py with logging configuration")
        return True
    except Exception as e:
        print(f"❌ Error updating main_with_quantitative.py: {e}")
        return False

def update_run_complete_system():
    """Update run_complete_system.py to use proper logging configuration."""
    
    try:
        with open('run_complete_system.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ run_complete_system.py not found")
        return False
    
    # Check if logging config is already imported
    if 'from src.utils.logging_config import setup_logging' in content:
        print("✅ Logging configuration already imported in run_complete_system.py")
        return True
    
    # Add logging configuration import
    logging_import = '''import logging
import sys
from pathlib import Path

# Setup logging configuration
from src.utils.logging_config import setup_logging
setup_logging()

# Disable werkzeug logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('dash').setLevel(logging.ERROR)
logging.getLogger('dash.dash').setLevel(logging.ERROR)
logging.getLogger('flask').setLevel(logging.ERROR)
logging.getLogger('aiohttp.access').setLevel(logging.ERROR)
logging.getLogger('websockets.server').setLevel(logging.WARNING)

'''
    
    # Find the first import statement
    import_pattern = r'^(import .*?)$'
    imports = re.findall(import_pattern, content, re.MULTILINE)
    
    if imports:
        # Insert logging config after the first import
        first_import_end = content.find(imports[0]) + len(imports[0])
        new_content = content[:first_import_end] + '\n' + logging_import + content[first_import_end:]
    else:
        # If no imports found, add at the beginning
        new_content = logging_import + content
    
    # Write the updated content
    try:
        with open('run_complete_system.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Updated run_complete_system.py with logging configuration")
        return True
    except Exception as e:
        print(f"❌ Error updating run_complete_system.py: {e}")
        return False

def create_enhanced_logging_config():
    """Create an enhanced logging configuration with better filtering."""
    
    config_content = '''import logging
import sys
from pathlib import Path

def setup_logging():
    """Setup enhanced logging configuration."""
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading_bot_quantitative_20250731.log', mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Disable noisy logs completely
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('dash').setLevel(logging.ERROR)
    logging.getLogger('dash.dash').setLevel(logging.ERROR)
    logging.getLogger('flask').setLevel(logging.ERROR)
    logging.getLogger('aiohttp.access').setLevel(logging.ERROR)
    logging.getLogger('websockets.server').setLevel(logging.WARNING)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
    logging.getLogger('requests.packages.urllib3.connectionpool').setLevel(logging.ERROR)
    
    # Keep important logs at INFO level
    logging.getLogger('src.strategies.enhanced_trading_strategy_with_quantitative').setLevel(logging.INFO)
    logging.getLogger('main_with_quantitative').setLevel(logging.INFO)
    logging.getLogger('src.quantitative').setLevel(logging.INFO)
    logging.getLogger('src.services').setLevel(logging.INFO)
    logging.getLogger('src.core').setLevel(logging.INFO)
    logging.getLogger('src.utils').setLevel(logging.INFO)
    
    # Set specific loggers to WARNING to reduce noise
    logging.getLogger('src.quantitative.performance_tracker').setLevel(logging.WARNING)
    logging.getLogger('src.quantitative.real_time_performance_monitor').setLevel(logging.WARNING)
    
    print("✅ Enhanced logging configured successfully")

if __name__ == "__main__":
    setup_logging()
'''
    
    try:
        # Create utils directory if it doesn't exist
        os.makedirs('src/utils', exist_ok=True)
        
        with open('src/utils/logging_config.py', 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print("✅ Created enhanced logging configuration file: src/utils/logging_config.py")
        return True
        
    except Exception as e:
        print(f"❌ Error creating enhanced logging config: {e}")
        return False

def main():
    """Main function to update logging configuration."""
    print("🔧 Updating Logging Configuration")
    print("=" * 50)
    
    # 1. Create enhanced logging config
    if create_enhanced_logging_config():
        print("✅ Enhanced logging config created")
    else:
        print("❌ Failed to create enhanced logging config")
        return
    
    # 2. Update main_with_quantitative.py
    if update_main_with_logging():
        print("✅ main_with_quantitative.py updated")
    else:
        print("❌ Failed to update main_with_quantitative.py")
    
    # 3. Update run_complete_system.py
    if update_run_complete_system():
        print("✅ run_complete_system.py updated")
    else:
        print("❌ Failed to update run_complete_system.py")
    
    print("\n" + "=" * 50)
    print("✅ Logging configuration updated!")
    print("\n📋 Changes made:")
    print("   • Created enhanced logging configuration")
    print("   • Updated main_with_quantitative.py")
    print("   • Updated run_complete_system.py")
    print("   • Disabled werkzeug, dash, and flask logs")
    print("   • Configured proper log levels")
    
    print("\n🚀 Next steps:")
    print("   1. Restart the bot: python run_complete_system.py")
    print("   2. Monitor logs for reduced noise")
    print("   3. Check if enhanced_trading_strategy_with_quantitative logs continue")
    print("   4. Verify symbol processing continues")

if __name__ == "__main__":
    main() 