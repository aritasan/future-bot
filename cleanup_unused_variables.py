#!/usr/bin/env python3
"""
Script to clean up unused variables, imports, and parameters in the strategy file.
"""

import re
import os
from typing import List, Set, Dict

def analyze_unused_imports(file_path: str) -> Dict:
    """Analyze unused imports in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all imports
    import_pattern = r'^import\s+(\w+)|^from\s+(\w+(?:\.\w+)*)\s+import\s+(.+)$'
    imports = re.findall(import_pattern, content, re.MULTILINE)
    
    # Find all variable usages
    variable_pattern = r'\b(\w+)\b'
    variables = re.findall(variable_pattern, content)
    
    # Analyze which imports are actually used
    used_imports = set()
    unused_imports = []
    
    for import_match in imports:
        if import_match[0]:  # direct import
            module = import_match[0]
            if module in variables:
                used_imports.add(module)
            else:
                unused_imports.append(f"import {module}")
        else:  # from import
            module = import_match[1]
            imported_items = import_match[2].split(',')
            for item in imported_items:
                item = item.strip()
                if item in variables:
                    used_imports.add(f"{module}.{item}")
                else:
                    unused_imports.append(f"from {module} import {item}")
    
    return {
        'used_imports': list(used_imports),
        'unused_imports': unused_imports
    }

def analyze_unused_variables(file_path: str) -> Dict:
    """Analyze unused instance variables in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find instance variable assignments
    instance_var_pattern = r'self\.(\w+)\s*='
    instance_vars = re.findall(instance_var_pattern, content)
    
    # Find instance variable usages
    usage_pattern = r'self\.(\w+)'
    usages = re.findall(usage_pattern, content)
    
    # Find unused instance variables
    unused_vars = []
    for var in instance_vars:
        if usages.count(var) <= 1:  # Only assigned, never used
            unused_vars.append(var)
    
    return {
        'instance_variables': instance_vars,
        'unused_instance_variables': unused_vars
    }

def analyze_unused_methods(file_path: str) -> Dict:
    """Analyze unused methods in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find method definitions
    method_pattern = r'async def (\w+)|def (\w+)'
    methods = re.findall(method_pattern, content)
    
    # Find method calls
    call_pattern = r'(\w+)\('
    calls = re.findall(call_pattern, content)
    
    # Find unused methods
    unused_methods = []
    for method in methods:
        method_name = method[0] if method[0] else method[1]
        if calls.count(method_name) <= 1:  # Only defined, never called
            unused_methods.append(method_name)
    
    return {
        'methods': [m[0] if m[0] else m[1] for m in methods],
        'unused_methods': unused_methods
    }

def analyze_unused_parameters(file_path: str) -> Dict:
    """Analyze unused parameters in method definitions."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find method definitions with parameters
    method_pattern = r'(?:async )?def (\w+)\([^)]*\):'
    methods = re.findall(method_pattern, content)
    
    # Find parameter usage in method bodies
    unused_params = {}
    
    # Split content into methods
    method_blocks = re.split(r'(?:async )?def \w+\([^)]*\):', content)
    
    for i, block in enumerate(method_blocks[1:], 1):  # Skip first empty block
        # Find parameters in method signature
        param_pattern = r'(\w+):\s*\w+'
        params = re.findall(param_pattern, block)
        
        # Check which parameters are used in method body
        unused_in_method = []
        for param in params:
            if param not in block:
                unused_in_method.append(param)
        
        if unused_in_method:
            unused_params[methods[i-1]] = unused_in_method
    
    return unused_params

def generate_cleanup_report(file_path: str) -> str:
    """Generate a comprehensive cleanup report."""
    report = []
    report.append("# WorldQuant Strategy Cleanup Report")
    report.append("=" * 50)
    
    # Analyze imports
    import_analysis = analyze_unused_imports(file_path)
    report.append("\n## Unused Imports:")
    for imp in import_analysis['unused_imports']:
        report.append(f"❌ {imp}")
    
    # Analyze instance variables
    var_analysis = analyze_unused_variables(file_path)
    report.append("\n## Unused Instance Variables:")
    for var in var_analysis['unused_instance_variables']:
        report.append(f"❌ self.{var}")
    
    # Analyze methods
    method_analysis = analyze_unused_methods(file_path)
    report.append("\n## Unused Methods:")
    for method in method_analysis['unused_methods']:
        report.append(f"❌ {method}")
    
    # Analyze parameters
    param_analysis = analyze_unused_parameters(file_path)
    report.append("\n## Unused Parameters:")
    for method, params in param_analysis.items():
        for param in params:
            report.append(f"❌ {method}() -> {param}")
    
    # Summary
    total_issues = (
        len(import_analysis['unused_imports']) +
        len(var_analysis['unused_instance_variables']) +
        len(method_analysis['unused_methods']) +
        sum(len(params) for params in param_analysis.values())
    )
    
    report.append(f"\n## Summary:")
    report.append(f"Total issues found: {total_issues}")
    report.append(f"Unused imports: {len(import_analysis['unused_imports'])}")
    report.append(f"Unused variables: {len(var_analysis['unused_instance_variables'])}")
    report.append(f"Unused methods: {len(method_analysis['unused_methods'])}")
    report.append(f"Unused parameters: {sum(len(params) for params in param_analysis.values())}")
    
    return "\n".join(report)

def create_cleanup_script(file_path: str) -> str:
    """Create a script to automatically clean up the issues."""
    script = []
    script.append("#!/usr/bin/env python3")
    script.append('"""')
    script.append("Auto-generated cleanup script for WorldQuant strategy.")
    script.append('"""')
    script.append("")
    script.append("import re")
    script.append("")
    
    # Analyze issues
    import_analysis = analyze_unused_imports(file_path)
    var_analysis = analyze_unused_variables(file_path)
    method_analysis = analyze_unused_methods(file_path)
    param_analysis = analyze_unused_parameters(file_path)
    
    script.append("def cleanup_strategy_file(file_path: str):")
    script.append('    """Clean up unused imports, variables, and methods."""')
    script.append("    with open(file_path, 'r', encoding='utf-8') as f:")
    script.append("        content = f.read()")
    script.append("")
    
    # Remove unused imports
    if import_analysis['unused_imports']:
        script.append("    # Remove unused imports")
        for imp in import_analysis['unused_imports']:
            if imp.startswith('import '):
                module = imp.replace('import ', '')
                script.append(f'    content = re.sub(r"^import {module}\\s*$", "", content, flags=re.MULTILINE)')
            else:
                # Handle from imports
                parts = imp.replace('from ', '').replace(' import ', '.').split('.')
                if len(parts) >= 3:
                    module = '.'.join(parts[:-1])
                    item = parts[-1]
                    script.append(f'    content = re.sub(r"from {module} import [^,]*{item}[^,]*", "", content)')
        script.append("")
    
    # Remove unused instance variables
    if var_analysis['unused_instance_variables']:
        script.append("    # Remove unused instance variables")
        for var in var_analysis['unused_instance_variables']:
            script.append(f'    content = re.sub(r"self\\.{var}\\s*=\\s*[^\\n]+\\n", "", content)')
        script.append("")
    
    # Remove unused methods
    if method_analysis['unused_methods']:
        script.append("    # Remove unused methods")
        for method in method_analysis['unused_methods']:
            script.append(f'    # Remove {method} method')
            script.append(f'    content = re.sub(r"(?:async )?def {method}\\([^)]*\\):[^\\n]*\\n(?:\\s+[^\\n]*\\n)*", "", content)')
        script.append("")
    
    script.append("    # Write cleaned content")
    script.append("    with open(file_path, 'w', encoding='utf-8') as f:")
    script.append("        f.write(content)")
    script.append("")
    script.append("    print(f'Cleaned up {file_path}')")
    script.append("")
    script.append("if __name__ == '__main__':")
    script.append("    cleanup_strategy_file('src/strategies/enhanced_trading_strategy_with_quantitative.py')")
    
    return "\n".join(script)

def main():
    """Main function to generate cleanup report and script."""
    file_path = "src/strategies/enhanced_trading_strategy_with_quantitative.py"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Generate report
    report = generate_cleanup_report(file_path)
    
    # Write report to file
    with open("cleanup_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    # Generate cleanup script
    script = create_cleanup_script(file_path)
    
    # Write script to file
    with open("cleanup_strategy.py", "w", encoding="utf-8") as f:
        f.write(script)
    
    print("✅ Cleanup analysis completed!")
    print("📄 Report saved to: cleanup_report.md")
    print("🔧 Cleanup script saved to: cleanup_strategy.py")
    print("\n📊 Summary:")
    print(report.split("## Summary:")[-1])

if __name__ == "__main__":
    main() 