from os.path import dirname, basename, isfile
from importlib import import_module
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

#is this sane or deeply wrong? it works ...
from modules import *

solvers = {}
for name in __all__:
    solvers[name] = eval(f"{name}.Solver()")