class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_warning(string):
    print(bcolors.WARNING+string+bcolors.ENDC)


def print_blue(string):
    print(bcolors.OKBLUE+string+bcolors.ENDC)


def print_green(string):
    print(bcolors.OKGREEN+string+bcolors.ENDC)


def print_header(string):
    print(bcolors.HEADER+string+bcolors.ENDC)


def print_fail(string):
    print(bcolors.FAIL+string+bcolors.ENDC)


def print_bold(string):
    print(bcolors.BOLD+string+bcolors.ENDC)


def print_underline(string):
    print(bcolors.UNDERLINE+string+bcolors.ENDC)
