"""Frozen command-line entry point for identity search.

This module deliberately imports no computer-vision dependency. In particular,
``--help`` remains available before NumPy/OpenCV/InsightFace/ONNX Runtime are
installed.
"""

from identity_search.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
