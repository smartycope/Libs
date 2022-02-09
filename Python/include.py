from Cope import *
import EasyRegex as er
import re
import math
import os
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
    globals()[moduleName] = importlib.import_module(
        name, moduleName).__getattribute__(moduleName)


importpath(HOME + "/hello/python/MathTranspiler/src/Vector.py",
           "Vector", "Vector2D")
importpath(HOME + "/hello/python/MathTranspiler/src/Particle.py",
           "Particle", "Particle2D")
# importpath("~/hello/python/MathTranspiler/src/MainWindow/_customFuncs", "*")


class MasterSolveDict(ZerosMultiAccessDict):
    """ A combonation of a ZerosDict and a MultiAccessDict """

    def __getitem__(self, keys):
        return MasterSolveDict(zip(keys, ensureIterable(super().__getitem__(keys))))


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


def unknown(d: dict, *obj: str):
    # if Counter(d.values())[None] != 1:
    count = 0
    for key in obj:
        if d[key] is None:
            count += 1
            thing = key
    if count != 1:
        return None
    else:
        return thing
        # return debug(invertDict(d)[None], clr=2)
        # return


def parallel(*resistors):
    bottom = 0
    for r in resistors:
        bottom += 1/r
    return 1/bottom


def series(*resistors):
    return sum(resistors)


def voltDivider(inVoltage, r1, r2) -> 'voltage in the middle':
    return (inVoltage * r2) / (r1 + r2)


def currentThroughParallelResistors(current, *resistors):
    totalR = parallel(*resistors)
    return [((totalR / r) * current) for r in resistors]

'''
def currentThroughParallelResistors(current, r1, r2):
    totalR = parallel(r1, r2)
    i1 = (totalR /r1) * current
    i2 = (totalR /r2) * current
    return (i1, i2)
'''

# masterSolve Functions
def solveOhmsLaw(v=None, i=None, r=None, p=None) -> 'dict(v, i, r, p)':
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
        'p': p
    }


def ohmsLaw(v=None, i=None, r=None) -> 'The one not specified':
    if have(v, r) and need(i):
        return v/r
    elif have(i, r) and need(v):
        return i*r
    elif have(v, i) and need(r):
        return v/i
    else:
        raise TypeError(f"Wrong number of parameters, bub")



def newtonsLaw(f=None, m=None, a=None) -> 'The one not specified':
    # F = ma
    if have(f, m) and need(a):
        return f/m
    elif have(m, a) and need(f):
        return m*a
    elif have(f, a) and need(m):
        return f/a
    else:
        raise TypeError(f"Wrong number of parameters, bub")

# Solves for a single axis at a time


@depricated
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


todo('add vectors to masterSolve', enabled=False)
allEquations = [
    # angular motion
    'Eq(angularSpeed, deltaAngle/deltaTime)',
    'Eq(angularAcceleration, deltaAngSpeed/deltaTime)',
    # newtons laws
    'Eq(netForce_x, mass * acceleration_x',
    'Eq(netForce_y, mass * acceleration_y',
    'Eq(weight, mass * acceleration_y)',
    'Eq(drag_x, -b * velocity_x)',
    'Eq(drag_y, -b * velocity_y)',

    # friction
    # What the heck is f_smax???
    # 'Eq(f_smax, staticFriction * netForce) ',
    # Motion equations
    "Eq(velocity_x, initialVelocity_x - acceleration_x * time)",
    "Eq(velocity_y, initialVelocity_y - acceleration_y * time)",
    "Eq(position_x, initialPosition_x + initialVelocity_x * time - (1/2)*acceleration_x*(time**2))",
    "Eq(position_y, initialPosition_y + initialVelocity_y * time - (1/2)*acceleration_y*(time**2))",
    "Eq(velocity_x, sqrt((initialVelocity_x**2) - 2 * acceleration_x * (position_x-initialPosition_x)))",
    "Eq(velocity_y, sqrt((initialVelocity_y**2) - 2 * acceleration_y * (position_y-initialPosition_y)))",
    # 'Eq(mass, 2*time)',
    # 'Eq(theta, mass*time)',
]

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
    'isVoltageDivider',
    # 'force',
    'netForce',
    'mass',
    'weight',
    'friction',
    'keneticFriction',
    'drag',
]
# Don't input these
masterSolveOutputParams = [
    "initialVelocity_x",
    "initialVelocity_y",
    "position_x",
    "position_y",
    "initialPosition_x",
    "initialPosition_y",
    "acceleration_x",
    "acceleration_y",
    "velocity_x",
    "velocity_y",
    'netForce_x',
    'netForce_y',
    'drag_x',
    'drag_y',
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
    'F': 'netForce',
    'µ': 'staticFriction',
    'mu': 'staticFriction',
    'friction': 'staticFriction'
}


vectorParams = ('velocity', 'initialVelocity', 'acceleration', 'netForce', 'drag')
pointParams = ('position', 'initialPosition', 'displacement', 'distance')
boolParams = ('isVoltageDivider',)

def masterSolve(maxIterations=2, __iteration=1, **v) -> "dict(solved parameters)":
    debug(__iteration, clr=3)
    # This just makes it so if I try to get a key that doesn't exist, it gives None instead of an error
    # And multiAccess becaues I want to be able to get multiple values from it
    v = ZerosMultiAccessDict(v)

    # * To make this as idiot-proof as possible
    # Make sure the parameters are valid, but only when the user is inputting them
    if __iteration == 0:
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
    number = (NoneType, float, int, SupportsInt,
              Symbol, Expr, Integer, Float, Rational)
    point = (NoneType, Point2D)
    boolean = (NoneType, bool)

    # Make sure everything is a sympy type, unless its a Vector2D or a bool
    newv = v
    for key, val in v.items():
        if type(val) in number:
            newv[key] = sympify(val)
    v = newv

    for i in masterSolveParams:
        def throw(param, value, expectType, type):
            raise TypeError(
                f"Parameter {param} has invalid value of {value}. (Expected {expectType}, got {type}")

        if i in pointParams:
            if type(v[i]) not in point:
                throw(i, v[i], point, type(v[i]))
        elif i in vectorParams:
            if type(v[i]) not in vector:
                throw(i, v[i], vector, type(v[i]))
        elif i in boolParams:
            if type(v[i]) not in boolean:
                throw(i, v[i], boolean, type(v[i]))
        elif type(v[i]) not in number:
            throw(i, v[i], number, type(v[i]))

    # Split up the axis parameters
    # NOTE -- These get reset every iteration
    # Use getattr so it doesn't throw an error
    if have(v['initialVelocity']):
        v['initialVelocity_x'] = getattr(v['initialVelocity'], 'x', None)
        v['initialVelocity_y'] = getattr(v['initialVelocity'], 'y', None)
    if have(v['position']):
        v['position_x'] = getattr(v['position'], 'x', None)
        v['position_y'] = getattr(v['position'], 'y', None)
    if have(v['initialPosition']):
        v['initialPosition_x'] = getattr(v['initialPosition'], 'x', None)
        v['initialPosition_y'] = getattr(v['initialPosition'], 'y', None)
    if have(v['acceleration']):
        v['acceleration_x'] = getattr(v['acceleration'], 'x', None)
        v['acceleration_y'] = getattr(v['acceleration'], 'y', None)
    if have(v['velocity']):
        v['velocity_x'] = getattr(v['velocity'], 'x', None)
        v['velocity_y'] = getattr(v['velocity'], 'y', None)
    if have(v['netForce']):
        v['netForce_x'] = getattr(v['netForce'], 'x', None)
        v['netForce_y'] = getattr(v['netForce'], 'y', None)

    # * Done idiot proofing, now actually solve stuff
    # First, electrical equations
    # v.update(solveOhmsLaw(v=v['voltage'], i=
    # v['current'], r=v['resistance'], p=v['power']))
    if v['isVoltageDivider']:
        v['middleVoltage'] = (v['voltage'] * v['R2']) / (v['R1'] + v['R2'])

    ohmsLawAnswers = solveOhmsLaw(v=v['voltage'], i=v['current'], r=v['resistance'], p=v['power'])
    if have(v['voltage']):
        v['voltage'] = ohmsLawAnswers['v']
    if have(v['current']):
        v['current'] = ohmsLawAnswers['i']
    if have(v['resistance']):
        v['resistance'] = ohmsLawAnswers['r']
    if have(v['power']):
        v['power'] = ohmsLawAnswers['p']

    # Solve the equations for the missing value, if we can
    for equation in allEquations:
        # try:
        # This is a hack, but I just want it to work.
        # Instead use eq.subs
        # for key, value in v.items():
        #     if value is not None:
        #         equation = equation.replace(key, str(value))
        eq = parse_expr(equation)
        # Make a new dict exactly the same as v, but instead of the keys being strings they're Symbols of those strings
        subv = {}
        for key, val in v.items():
            if type(val) not in (Point2D, Vector2D):
                subv[Symbol(key)] = val
        eq = eq.subs(subv)

        # except:
        # raise SyntaxError(f"Error parsing equation '{equation}'")
        atoms = [a.name for a in eq.atoms(Symbol)]
        u = unknown(v, *atoms)
        # debug(u)
        if u:
            v[u] = ensureNotIterable(solve(eq, Symbol(u)), None)
            if v[u] is None:
                debug(f'Unable to solve {eq} for {u}', clr=Colors.ALERT)

    # Recombine the axis parameters
    # NOTE -- These get reset every iteration
    if have(v['position_x'], v['position_y']):
        v['position'] = Point2D(v['position_x'], v['position_y'])
    if have(v['initialPosition_x'], v['initialPosition_y']):
        v['initialPosition'] = Point2D(
            v['initialPosition_x'], v['initialPosition_y'])
    if have(v['initialVelocity_x'], v['initialVelocity_y']):
        v['initialVelocity'] = Vector2D.fromxy(
            v['initialVelocity_x'], v['initialVelocity_y'])
    if have(v['velocity_x'], v['velocity_y']):
        v['velocity'] = Vector2D.fromxy(v['velocity_x'], v['velocity_y'])
    if have(v['acceleration_x'], v['acceleration_y']):
        v['acceleration'] = Vector2D.fromxy(
            v['acceleration_x'], v['acceleration_y'])
    if have(v['netForce_x'], v['netForce_y']):
        v['netForce'] = Vector2D.fromxy(v['netForce_x'], v['netForce_y'])

    # Sync psuedonyms
    for psuedo, key in masterSolvePsuedonyms.items():
        if have(key) and need(psuedo):
            v[psuedo] = v[key]
        elif have(psuedo) and need(key):
            v[key] = v[psuedo]

    if __iteration < maxIterations:
        masterSolve(maxIterations=maxIterations,
                    __iteration=__iteration + 1, **v)

    return v

masterSolve.__doc__ = 'Valid Parameters are: ' + '\n'.join(masterSolveParams)

s = series
ll = parallel

scinot.start(4, 3)

specialSymbols = '≈θ𝜙°Ω±𝛼𝚫𝜔'
# gravity = Vector2D(9.8, 270, False)
gravity = Vector2D.fromxy(0, 9.8)
# debug(masterSolve(acceleration=gravity, initialVelocity=Vector2D(90, 30, False), initialPosition=Point2D(0, 10), time=3))

#! Actually test this with practice problems!

# debug(masterSolve(time=.2, acceleration=gravity, initialPosition=Point2D(0, 0))['position', 'a', 'time', 'velocity'])
# debug(masterSolve(time=.2, acceleration=gravity, initialPosition=Point2D(0, 0)))
# 'Eq(mass, 2*time)',
# 'Eq(theta, mass*time)',
# debug(masterSolve(mass=4))
# debug(masterSolve(mass=4)['mass'])
# debug(masterSolve(mass=4)['mass', 'time'])

# def node(input)

# def findI()

# def something(inVCs, outVCs):


# Wire = namedtuple('Wire', 'v, i, r')
# Node = namedtuple('Node', 'in, out')

# Node((Wire(v=15, r=0),), (Wire()))


# def volts(inVs, outVs, inR, outR):

todo('If a class inherits from 2 parent classes which both implement a function, and that function is called, which function gets called?', enabled=False)


# def current(inVCs, outVCs):

class Resistor:
    def __init__(self, resistance):
        self.r = resistance


class Node:
    def __init__(self, inVoltages=None, outVoltages=None, inCurrents=None, outCurrents=None, inResistors=None, outResistors=None):
        self.inVoltages = inVoltages
        self.outVoltages = outVoltages
        self.inCurrents = inCurrents
        self.outCurrents = outCurrents
        self.inResistors = inResistors
        self.outResistors = outResistors

    # def current(self):


# ElectricalPart


# PowerSupply
# Resistor
# Node
# connections =


# def currentAtNode(*voltages):
#     return ohmsLaw(v=Eq(sum(voltages), 0), )


# vs1=24; vs2=12; R1=110; R2=270; R3=560

# i1Equ = Eq(-ohmsLaw(r=R1, i=i1) - ohmsLaw(r=R2, i=i1 - i2), -vs1)
# i2Equ = Eq-(vs2, -ohmsLaw(r=R2, i=i2 - i1) - ohmsLaw(r=R3, i=i2))
# i1Solved = ensureNotIterable(solve(i1Equ, i1))
# i2Solved = ensureNotIterable(solve(i2Equ, i2))
# i2Solved.subs(i1, i1Solved)
# i1Solved.subs(i2, i2Solved)
# solve(i1Solved.subs(i2, i2Solved), i1)
# solve(i2Solved.subs(i1, i1Solved), i2)
# solve(i1Solved.subs(i2, i2Solved), i1)[0].evalf()
# solve(i2Solved.subs(i1, i1Solved), i2)[0].evalf()
