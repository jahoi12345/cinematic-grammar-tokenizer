"""Entry point of the interactive demo run locally with Streamlit: `streamlit run streamlit_app.py` (the live demo is the static site in app/static/).

The app is public by construction: before the app is imported, public mode is switched on and the data root is pinned to the
bundled `demo_data/` folder, which holds only the public-domain film Night of the Living Dead (1968). Any environment value set on the host
is overridden here. The app's own guard (app/demo/core.py, `public_mode()`) also fails closed.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["CGT_DEMO_PUBLIC"] = "1"
os.environ["CGT_DEMO_ROOT"] = str(ROOT / "demo_data")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

runpy.run_path(str(ROOT / "app" / "demo" / "app.py"), run_name="__main__")
