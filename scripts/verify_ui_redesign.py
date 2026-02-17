
import sys
import os
import json
from datetime import datetime
import random

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from PyQt6 import QtWidgets, QtCore, QtGui
from ui.adaptive_learning_tab import AdaptiveLearningTab

# Mock Data Generation
def generate_mock_kelly_state():
    return {
        "created_at": datetime.now().timestamp(),
        "kelly_adapter": {
            "kelly_fraction": 0.85,
            "recent_performance": [1.0 + random.uniform(-0.5, 0.5) for _ in range(20)]
        }
    }

def generate_mock_bayesian_state():
    return {
        "state": {
            "total_signals_received": 120,
            "total_signals_accepted": 78,
        }
    }

def generate_mock_tpsl_state():
    return {"state": {}}

# Mock Class
class MockAdaptiveLearningTab(AdaptiveLearningTab):
    def _load_state_file(self, key):
        print(f"Loading mock state for: {key}")
        if key == 'kelly':
            return generate_mock_kelly_state()
        elif key == 'bayesian':
            return generate_mock_bayesian_state()
        elif key == 'tpsl':
            # Verification script handles specific override logic
            return generate_mock_tpsl_state()
        return None

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Set global font to something standard if Orbitron is missing
    font = QtGui.QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("R3000 HUD Design Verification")
    window.resize(1200, 800)
    window.setStyleSheet("background-color: #0B0E11;")
    
    tab = MockAdaptiveLearningTab()
    window.setCentralWidget(tab)
    
    # Simulate log stream
    tab.append_adaptive_journal("Mock verification started.")
    tab.append_adaptive_journal("Connecting to neural interface...")
    tab.append_adaptive_journal("Data stream established.")
    
    window.show()
    
    print("UI Verification launched. Close the window to exit.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
