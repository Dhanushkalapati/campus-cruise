"""Crop the full app screenshots (from shots.js / shots2.js) to the regions used on slides 6 and 7.
Run from presentation/build: python crop.py ../screenshots
Boxes assume a 1600x1000 viewport at 1.5x scale, as set in shots.js."""
import sys
from pathlib import Path
from PIL import Image

out = Path(sys.argv[1] if len(sys.argv) > 1 else "../screenshots")
for src, box, dst in [
    ("app-routes.png", (564, 468, 2286, 1262), "crop-routes.png"),
    ("app-overview.png", (564, 288, 2286, 1050), "crop-overview.png"),
    ("app-assistant.png", (1704, 348, 2364, 1398), "crop-assistant.png"),
]:
    Image.open(out / src).crop(box).save(out / dst)
    print("wrote", out / dst)
