from Cope import *
import EasyRegex as er
import re, math, os
from os.path import dirname, join
import clipboard as clip
from clipboard import copy, paste
import scinot
from sympy import *
from sympy import abc
from sympy.abc import *
from sympy.calculus.util import continuous_domain
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication,
                                        implicit_multiplication_application,
                                        lambda_notation, parse_expr,
                                        standard_transformations)
from sympy.physics.units import *
import sympy.physics.units as _units
from sympy.physics.units.prefixes import Prefix
import sys


def importpath(path, name, moduleName):
    spec = importutil.spec_from_file_location(name, path)
    module = importutil.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # This is kinda ugly and nasty, but it works. For now.
    globals()[moduleName] = importlib.import_module(name, moduleName).__getattribute__(moduleName)



importpath(HOME + "/hello/python/MathTranspiler/src/Vector.py", "Vector", "Vector2D")
importpath(HOME + "/hello/python/MathTranspiler/src/Particle.py", "Particle", "Particle2D")
# importpath("~/hello/python/MathTranspiler/src/MainWindow/_customFuncs", "*")


def parallel(*resistors):
    bottom = 0
    for r in resistors:
        bottom += 1/r
    return 1/bottom

def series(*resistors):
    return sum(resistors)

def voltDivider(r1, r2) -> 'voltage':
    pass

s = series
ll = parallel

# ≈θ𝜙°Ω±𝛼𝚫𝜔