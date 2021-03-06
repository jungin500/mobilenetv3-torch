from datetime import datetime
import re

ansi_escape_regex = re.compile(r'''
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by a control sequence
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
''', re.VERBOSE)


def now():
    return str(datetime.now())


def escape(data):
    return ansi_escape_regex.sub('', data)


class BC:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class OutputLogger(object):
    def __init__(self, run_uuid):
        self.writer = open('latest.log', 'a')
        self.print = print
        self.sep = '\n'
        self.run_uuid = run_uuid

        self.__write__('** Training %s Started **' % self.run_uuid)

    def __del__(self):
        self.__write__('** Training %s End **' % self.run_uuid)
        self.writer.flush()
        self.writer.close()

    def __write__(self, *args):
        real_data = ''.join([str(x) for x in args]) + self.sep

        self.writer.write(escape(real_data))
        self.writer.flush()  # I/O-slow but accurate
        self.print('[%s]\t%s' % (now(), real_data), end='')

    def info(self, *args):
        self.__write__(*args)

    def debug(self, *args):
        self.__write__(BC.HEADER, *args, BC.ENDC)

    def warning(self, *args):
        self.__write__(BC.WARNING, *args, BC.ENDC)

    def error(self, *args):
        self.__write__(BC.FAIL, *args, BC.ENDC)