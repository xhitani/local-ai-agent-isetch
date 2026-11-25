"""Point d'entrée Streamlit : streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "agent"))

import ui  # noqa: E402

ui.render()
