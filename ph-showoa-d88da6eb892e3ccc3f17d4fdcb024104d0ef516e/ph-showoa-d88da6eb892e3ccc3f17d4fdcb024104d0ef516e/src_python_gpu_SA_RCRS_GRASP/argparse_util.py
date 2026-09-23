import sys

class ArgumentParser:
    def __init__(self):
        self.options = {}

    def add_argument(self, name, nargs=0, default=None):
        clean_name = name.lstrip('-')
        self.options[clean_name] = default

    def parse(self, argv=None):
        if argv is None:
            argv = sys.argv
        i = 1
        while i < len(argv):
            arg = argv[i]
            if arg.startswith('--'):
                key = arg[2:]
                if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                    self.options[key] = argv[i + 1]
                    i += 2
                else:
                    self.options[key] = True
                    i += 1
            else:
                if 'problem' in self.options and (self.options['problem'] is None or self.options['problem'] is False):
                    self.options['problem'] = arg
                i += 1

    def exists(self, key):
        return key in self.options and self.options[key] is not None and self.options[key] is not False

    def retrieve(self, key):
        return self.options.get(key, None)
