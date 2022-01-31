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
from collections import *

# This SHOULD be in Cope.py, but it wont work there. Not sure why.
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

# returns true if all of the parameters are not None
def have(*obj):
    yes = True
    for i in obj:
        if i is None:
            yes = False
    return yes

# returns true if all of the parameters are None
def need(*obj):
    yes = True
    for i in obj:
        if i is not None:
            yes = False
    return yes

# returns true if there's no more than 1 None parameter
def involves(*obj):
    unknowns = 0
    for i in obj:
        if i is None:
            unknowns += 1
    return unknowns <= 1

# returns the only parameter equal to None (or None if there are more or less than 1)
def unknown(d:dict, *obj:str):
    if Counter(d.values())[None] != 1:
        return None
    else:
        return invertDict(d)[None]


def parallel(*resistors):
    bottom = 0
    for r in resistors:
        bottom += 1/r
    return 1/bottom

def series(*resistors):
    return sum(resistors)

def voltDivider(inVoltage, r1, r2) -> 'voltage in the middle':
    return (inVoltage * r2) / (r1 + r2)

# grandpa satan
# barage corbin with genuine complements
# soda + salad packet + warmslough
# berlin wall
# chaperone on shopping trips

# masterSolve Functions
def ohmsLaw(v=None, i=None, r=None, p=None) -> 'dict(v, i, r, p)':
    """ This is really just a nice implementation of the ohm's law wheel """
    if need(i):
        if have(v, r):
            i = v/r
        elif have(p, v):
            i = p/v
        elif have(p, r):
            i = sqrt(p/r)

    if need(v):
        if have(i, r):
            v = i*r
        elif have(p, i):
            v = p / i
        elif have(p, r):
            v = sqrt(p*r)

    if need(r):
        if have(v, i):
            r = v/i
        elif have(v, p):
            r = (v**2)/p
        elif have(p, i):
            r = p / (i**2)

    if need(p):
        if have(v, i):
            p = v * i
        elif have(v, r):
            p = (v**2)/r
        elif have(i, r):
            p = (i**2)*r

    return {
        'v': v,
        'i': i,
        'r': r,
        'p':p
    }



# These each solve for a single axis at a time
motionEquations = [
    "Eq(velocity_x, initalVelocity_x - acceleration_x * time)",
    "Eq(velocity_y, initalVelocity_y - acceleration_y * time)",
    "Eq(position_x, initialPosition_x + initialVelocity_x * time - (1/2)*acceleration_x*(time**2))",
    "Eq(position_y, initialPosition_y + initialVelocity_y * time - (1/2)*acceleration_y*(time**2))",
    "Eq(velocity_x, sqrt((initialVelocity_x**2) - 2 * acceleration_x * (position_x-initialPosition_x)))",
    "Eq(velocity_y, sqrt((initialVelocity_y**2) - 2 * acceleration_y * (position_y-initialPosition_y)))",
]

miscEquations = [
    'Eq(angularSpeed, deltaAngle/deltaTime)',
    'Eq(angularAcceleration, deltaAngSpeed/deltaTime)'
]

# Solves for a single axis at a time
def solveMotion(**args):
    """
        Allowed Parameters:
            acceleration
            velocity:number
            initialVelocity:number
            time
            position:number
            initialPosition:number
    """
    for equation in motionEquations:
        eq = parse_expr(equation)
        u = unknown(args, *[a.name for a in eq.atoms(Symbol)])
        if u:
            args[u] = solve(eq, args[u])

    return {
        'acceleration': acceleration,
        'velocity': velocity,
        'initialVelocity': initialVelocity,
        'time': time,
        'position': position,
        'initalPosition': initialPosition
        # 'gravity': gravity
    }

masterSolveParams = [
    "voltage",
    "current",
    "resistance",
    "power",
    "equivalentResistance",
    "theta",
    "degTheta",
    "vectorMagnitude",
    "position",
    "initialPosition",
    "displacement",
    "distance",
    "speed",
    "initialSpeed",
    "velocity",
    "initialVelocity",
    "acceleration",
    "rpm",
    "angularSpeed",
    "angularAcceleration",
    "time",
    "deltaAngle",
    "deltaTime",
    "deltaAngSpeed",
    "deltaTime",
    "R1",
    'R2',
    'isVoltageDivider'
]

# Don't input these
masterSolveOutputParams = [
    "initalVelocity_x",
    "initalVelocity_y",
    "position_x",
    "position_y",
    "initialPosition_x",
    "initialPosition_y",
    "acceleration_x",
    "acceleration_y",
    "velocity_x",
    "velocity_y",
]

masterSolvePsuedonyms = {
    'accel': 'acceleration',
    'vel': 'velocity',
    'angSpeed': 'angularSpeed',
    'dist': 'distance',
    'Req': 'equivalentResistance',
    'angAccel': 'angularAcceleration',
    'disp': 'displacement',
    'v': 'voltage',
    'volt': 'voltage',
    'volts': 'voltage',
    'i': 'current',
    'p': 'power',
    'r': 'vectorMagnitude',
    'R': 'resistance',
    't': 'time',
    'initVel': 'initialVelocity',
    'initX': 'initialX',
    'initY': 'initialY',
    'initZ': 'initialZ',
    'θ': 'theta',
    '𝜙': 'phi',
    '°': 'degTheta',
    'Ω': 'resistance',
    '𝛼': 'angularAcceleration',
    '𝜔': 'angularSpeed',
    'θ°': 'degTheta',
    '𝚫ang': 'deltaAngle',
    '𝚫t': 'deltaTime',
    '𝚫angSpeed': 'deltaAngSpeed',
    'pos': 'position',
    'initPos': 'initialPosition',
    'isVoltDivider': 'isVoltageDivider',
}

def masterSolve(maxIterations = 1, __iterations=0, **v) -> "dict(solved parameters)":
    # This just makes it so if I try to get a key that doesn't exist, it gives None instead of an error
    v = ZerosDict(v)

    #* To make this as idiot-proof as possible
    # Make sure the parameters are valid, but only when the user is inputting them
    if __iterations == 0:
        for key in v.keys():
            if key not in masterSolveParams + list(masterSolvePsuedonyms.keys()):
                raise TypeError(f"Invalid parameter {key}")

    # Sync psuedonyms
    for psuedo, key in masterSolvePsuedonyms.items():
        if have(key) and need(psuedo):
            v[psuedo] = v[key]
        elif have(psuedo) and need(key):
            v[key] = v[psuedo]


    # Make sure we have the right types
    NoneType = type(None)
    vector = (NoneType, Vector2D)
    number = (NoneType, float, int, SupportsInt, Symbol, Expr)
    point  = (NoneType, Point2D)
    boolean= (NoneType, bool)
    for i in masterSolveParams:
        def throw():
            raise TypeError(f"Parameter {i} has invalid value of {v[i]}. (Expected {point}, got {type(v[i])}")

        if i in ('position', 'initialPosition', 'displacement', 'distance'):
            if type(v[i]) not in point:
                throw()
        elif i in ('velocity', 'initialVelocity', 'acceleration'):
            if type(v[i]) not in vector:
                throw()
        elif i in ('isVoltageDivider',):
            if type(v[i]) not in boolean:
                throw()
        elif type(v[i]) not in number:
            throw()

    # Now add the axis specific parameters
    # NOTE -- These get reset every iteration
    # Use getattr so it doesn't throw an error
    v['initalVelocity_x'] = getattr(v['initalVelocity'], 'x', None)
    v['initalVelocity_y'] = getattr(v['initalVelocity'], 'y', None)
    v['position_x'] = getattr(v['position'], 'x', None)
    v['position_y'] = getattr(v['position'], 'y', None)
    v['initialPosition_x'] = getattr(v['initialPosition'], 'x', None)
    v['initialPosition_y'] = getattr(v['initialPosition'], 'y', None)
    v['acceleration_x'] = getattr(v['acceleration'], 'x', None)
    v['acceleration_y'] = getattr(v['acceleration'], 'y', None)
    v['velocity_x'] = getattr(v['velocity'], 'x', None)
    v['velocity_y'] = getattr(v['velocity'], 'y', None)


    #* Done idiot proofing, now actually solve stuff
    # First, electrical equations
    # v.update(ohmsLaw(v=v['voltage'], i=v['current'], r=v['resistance'], p=v['power']))
    if v['isVoltageDivider']:
        v['middleVoltage'] = (v['voltage'] * v['R2']) / (v['R1'] + v['R2'])

    ohmsLawAnswers = ohmsLaw(v=v['voltage'], i=v['current'], r=v['resistance'], p=v['power'])
    # v['voltage'] = ohmsLawAnswers['v']
    # v['current'] = ohmsLawAnswers['i']
    # v['resistance'] = ohmsLawAnswers['r']
    # v['power'] = ohmsLawAnswers['p']

    # Take a stab at vector stuff
    todo('add vectors to masterSolve', enabled=False)

    # Solve the equations for the missing value, if we can
    for equation in miscEquations + motionEquations:
        try:
            eq = parse_expr(equation)
        except:
            raise SyntaxError(f"Error parsing equation '{equation}'")
        u = unknown(v, *[a.name for a in eq.atoms(Symbol)])
        if u:
            v[u] = solve(eq, v[u])

    if __iterations <= maxIterations:
        masterSolve(maxIterations=maxIterations, __iterations=__iterations + 1, **v)

    return v

masterSolve.__doc__ = 'Valid Parameters are: ' + '\n'.join(masterSolveParams)



s = series
ll = parallel

scinot.start(4, 3)

specialSymbols = '≈θ𝜙°Ω±𝛼𝚫𝜔'

debug(masterSolve(acceleration=Vector2D(0, 9.8), initialVelocity=Vector2D(90, 30, False), initialPosition=Point2D(0, 10), time=3))