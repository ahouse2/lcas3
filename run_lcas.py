
#!/usr/bin/env python3
"""
LCAS Run Script
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from lcas2.main import main
    main()
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying direct GUI import...")
    try:
        from lcas2.gui.main_gui import main as gui_main
        gui_main()
    except Exception as e2:
        print(f"GUI import failed: {e2}")
        sys.exit(1)
except Exception as e:
    print(f"Error running LCAS: {e}")
    sys.exit(1)
