"""
Script to fix the pandas_ta import error
"""

import os

# Path to the problematic file
file_path = r"C:\Users\spars\Desktop\kise pata\.venv\Lib\site-packages\pandas_ta\momentum\squeeze_pro.py"

# Read the file content
with open(file_path, 'r') as file:
    content = file.read()

# Replace the problematic import
fixed_content = content.replace(
    "from numpy import NaN as npNaN",
    "from numpy import nan as npNaN"  # 'nan' is lowercase in NumPy
)

# Write the fixed content back
with open(file_path, 'w') as file:
    file.write(fixed_content)

print(f"Fixed pandas_ta squeeze_pro.py file at: {file_path}")