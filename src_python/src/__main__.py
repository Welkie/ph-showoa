try:
    from .main import main
except ImportError:
    # Allow running as a script: python src/__main__.py ...
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.main import main


if __name__ == "__main__":
    main()
