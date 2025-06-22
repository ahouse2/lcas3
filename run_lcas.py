#!/usr/bin/env python3
"""
LCAS Startup Script
Handles path setup and launches the application
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variable for the project root
os.environ['LCAS_PROJECT_ROOT'] = str(project_root)

# Now import and run LCAS
if __name__ == "__main__":
    try:
        from lcas2.main import main
        main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed and paths are correct.")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting LCAS: {e}")
        sys.exit(1)
