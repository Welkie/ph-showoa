try:
    from .main import main
except ImportError:
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sim_test.main import main


if __name__ == "__main__":
    main()
