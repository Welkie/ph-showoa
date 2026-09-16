import argparse


class ArgumentParser:
    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser()
        self._args = None
        self._dest_map = {}

    def add_argument(self, name: str, nargs=0, optional: bool = True) -> None:
        dest = self._strip(name)
        if nargs == 0:
            self._parser.add_argument(name, dest=dest, action="store_true", required=False)
        elif nargs in ("+", "*"):
            self._parser.add_argument(name, dest=dest, nargs=nargs, required=not optional)
        elif nargs == 1:
            self._parser.add_argument(name, dest=dest, required=not optional)
        else:
            self._parser.add_argument(name, dest=dest, nargs=nargs, required=not optional)
        self._dest_map[name] = dest
        self._dest_map[dest] = dest

    def parse(self, argv) -> None:
        self._args = self._parser.parse_args(argv[1:])

    def exists(self, name: str) -> bool:
        if self._args is None:
            raise RuntimeError("Arguments not parsed")
        dest = self._normalize(name)
        value = getattr(self._args, dest, None)
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, list):
            return len(value) > 0
        return True

    def retrieve(self, name: str):
        if self._args is None:
            raise RuntimeError("Arguments not parsed")
        dest = self._normalize(name)
        value = getattr(self._args, dest)
        if isinstance(value, list):
            if len(value) == 1:
                return value[0]
            return value
        return value

    def _normalize(self, name: str) -> str:
        return self._dest_map.get(name, self._strip(name))

    @staticmethod
    def _strip(name: str) -> str:
        return name.lstrip("-").replace("-", "_")
