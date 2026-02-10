"""
run training when executing the package.

  python -m boostrap_bot [--algo PPO --steps 100000 ...]
  python -m boostrap_bot.export_onnx model.zip out.onnx
"""

import runpy
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    runpy.run_path(str(script), run_name="__main__")
