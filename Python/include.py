from Cope import *
import EasyRegex as er
import re
import math
import os
from os.path import dirname, join
import clipboard as clip
from clipboard import copy, paste
import scinot
from sympy.physics.units import *
from sympy.abc import *
from sympy import *
from sympy import abc
from sympy.calculus.util import continuous_domain
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication,
                                        implicit_multiplication_application,
                                        lambda_notation, parse_expr,
                                        standard_transformations)
import sympy.physics.units as _units
from sympy.physics.units.prefixes import Prefix
from sympy.physics.optics import TWave
import sys
from warnings import warn
from collections import *
# from sympy.core.evalf import N as evalf
# from sympy import N as evalf
from sympy import N
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from importlib import reload
from sympy.physics.vector import *
# This is intentionally imported last
from constants import *
# if not debug(ZerosDict(globals())['NO_EQUATIONS']):
# from Equations import *



# SUM SYNTAX
# Sum(val, (i, start, end))
# Sum(val, (i, low, high))

# INTEGRAL SYNTAX
# Integral(val, var)
# Integral(val, (var, low, high))


# This SHOULD be in Cope.py, but it wont work there. Not sure why.
def importpath(path, name, moduleName):
    spec = importutil.spec_from_file_location(name, path)
    module = importutil.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # This is kinda ugly and nasty, but it works. For now.
    globals()[moduleName] = importlib.import_module(
        name, moduleName).__getattribute__(moduleName)

# Ugh...
# home = os.path.expanduser("~")
# font = ImageFont.truetype(home+"/pyfreebody.ttf", 20)
# fontTag = ImageFont.truetype(home+"/pyfreebody.ttf", 12)

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

# returns the dict d without any of the stuff equal to None in it
def known(d: dict, *obj: str):
    newd = {}
    for key, val in d.items():
        if val is not None:
            newd[key] = val
    return newd




def parallel(*resistors):
    bottom = 0
    for r in resistors:
        bottom += 1/r
    return 1/bottom


def series(*resistors):
    return sum(resistors)



_i = I
_e = E


def voltDivider(inVoltage, r1, r2) -> 'voltage in the middle':
    return (inVoltage * r2) / (r1 + r2)

def splitEvenly(amount, *avenues):
    net = parallel(*avenues)
    return [((net / r) * amount) for r in avenues]

currentThroughParallelResistors = splitEvenly

"""
def currentThroughParallelResistors(current, *resistors):
    totalR = parallel(*resistors)
    return [((totalR / r) * current) for r in resistors]
"""

def voltageBetweenCapacitors(Vin, C1, C2):
    vn2 = symbols('vn2')
    C1Charge = capacitor(v=Vin-vn2, C=C1)
    C2Charge = capacitor(v=vn2,     C=C2)
    return ensureNotIterable(solve(Eq(C1Charge, C2Charge), vn2))


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

"""
def ohmsLaw(v=None, i=None, r=None) -> 'The one not specified':
    if have(v, r) and need(i):
        return v/r
    elif have(i, r) and need(v):
        return i*r
    elif have(v, i) and need(r):
        return v/i
    else:
        raise TypeError(f"Wrong number of parameters, bub")
"""

def ohmsLaw2(v=None, i=None, r=None, p=None) -> 'dict(v, i, r, p)':
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


def maxPower(Vth, Rth) -> 'P_Rl':
    return (Vth ** 2) / ( 4 * Rth)


def norton2Thevinin(In, Rn):
    return (In * Rn, Rn)

def thevinin2Norton(Vth, Rth):
    return (Vth / Rth, Rth)

"""
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
"""
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

'''
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

    #* Done idiot proofing, now actually solve stuff
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
'''

DOWN = 3*pi/2
UP   = pi/2
LEFT = pi
RIGHT= 0

scinot.start(4, 3)

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

class Component:
    def __init__(self, havePolarity=False):
        pass


# Integral syntax:
# Integral(base, (var, low, high))


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

# Particle = Particle2D
# Vector = Vector2D

specialSymbols = '≈θ𝜙°Ω±𝛼𝚫𝜔'

dummyDict = {'a':1, 'b':2, 'c':3}
# dummyDict2 = {'a':1, 'b':2}
dummySet = {1, 2, 3}
dummySet2 = {2, 3, 4, 5, 6, 7}

NORTH=pi/2
EAST=0
WEST=pi
SOUTH=3*pi/2

# My own copy function
def cp(thing, rnd=4, show=False):
    if isinstance(thing, Basic) and not isinstance(thing, Float):
        copy(latex(thing))
        if show:
            print('latexing')
    else:
        try:
            if rnd:
                copy(str(round(thing, rnd)))
                if show:
                    print('rounding')
            else:
                raise Exception()
        except:
            copy(str(thing))
            if show:
                print('stringing')
    return thing



class ReadOnlyError(Exception):
    pass

# Amplitude * sin(period * (xOffset - AngularFrequency) * time + phase) + yOffset
# frequency == ohmega
# ohmega == angular frequency
# ohmega == 1/period



# PHASOR QUESTIONS
# what is 120 in (170 V)sin(120πt))? is it period, or angular frequency or something else?
# y_t == A * cos(k * x - ohmega * t + phi) + y

# How do you get
# What is the sin(a) + jcos(b) thing for?
# How do you convert from polar to exponential coordinants and back?
# there's phase angle, phase, theta, phi, angular frequency, and frequency. Which ones are the same thing, and how do they relate to each other?


# period / pi -> Amp * sin(period/pi)
# theta == period


# period = 1/frequency = wavelength / velocity
# 2*pi *f == angular frequency == 2*pi/period

@reprise
class Phasor(AtomicExpr):
    def __init__(self, real, img):
        self.real = self.re = real
        self.img  = self.im = img
        self.peak2peak = self.amplitude = self.A
        self.phase = self.phi
        self.r = self.mag

        # self.theta = self.period

    def getQuadrant(self):
        pass

    def angularFrequency(self):
        return 2 * pi * frequency()

    def frequency(self):
        return self._wave.time_period

    def period(self):
        return self.theta

    def wavelength(self):
        NotImplemented


    @staticmethod
    @confidence(65)
    def fromExp(A, phi):
        return Phasor.fromPolar(A, phi)

    @staticmethod
    def fromWave(wave:TWave):
        # Amplitude * sin(period * (xOffset - AngularFrequency) * time + phase) + yOffset
        # sineParser = er.group(er.optional(er.number())) + er.group(er.either('sin', 'cos')) + '(' + er.group(er.optional(er.number())) +
        # return Phasor.fromExp(wave.amaplitude, wave.)
        NotImplemented

    @staticmethod
    def fromSine(amplitude, phase=0, sin=True):
        # phi == phase == xOffset


        # print(f'period = {period}')
        # print(f'frequency = {frequency}')
        # print(f'phase = {phase}')

        return Phasor.fromPolar(amplitude, (phase - pi) if sin else phase)
        # x, y = symbols('x y')
        # _x = ensureNotIterable(solve(amplitude * cos(period) / x + amplitude * sin(period) / y, x))
        # _y = ensureNotIterable(solve(amplitude * cos(period) / x + amplitude * sin(period) / y, y))
        # return Phasor(_x, _y)

        # wave = TWave(amplitude, period=period)
        # return Phasor.fromWave(wave)

    @staticmethod
    def fromRect(real, img):
        return Phasor(real, img)

    @staticmethod
    def fromPolar(mag, theta):
        return Phasor(mag * cos(theta), mag * sin(theta))

    @staticmethod
    def fromPolarDeg(mag, theta):
        return Phasor(mag * cos(rad(theta)), mag * sin(rad(theta)))


    def asExp(self, eval=True) -> "A*e^(j*phi)":
        _N = N if eval else lambda x: x
        return _N(self.A, n=5) * _e**(_i * _N(self.phi, n=5))

    def asSine(self, angFeq, sin=False, eval=True) -> "A*cos(ang*t+theta)":
        var('t')
        if sin:
            return im(self.A * _e ** (_i * (angFeq*t + self.phi)))
        else:
            return (self * _e ** (_i*angFeq*t)).im.simplify()

    def asRect(self, eval=True) -> "real + j*img":
        _N = N if eval else lambda x: x
        return _N(self.re, n=5) + _i * _N(self.im, n=5)

    def asPolar(self, eval=True) -> "(mag, theta)":
        # return (self.mag, self.theta)
        _N = N if eval else lambda x: x
        return f'{_N(self.mag, n=5)} ∠ {_N(self.theta, n=5)}'

    def asPolarDeg(self, eval=True) -> "(mag, theta)":
        # return (self.mag, deg(self.theta))
        _N = N if eval else lambda x: x
        return f'{_N(self.mag, n=5)} ∠ {_N(deg(self.theta), n=5)}°'


    @property
    def mag(self):
       return sqrt((self.re**2) + (self.im**2))
    @mag.setter
    def mag(self, to):
       raise ReadOnlyError()

    @property
    def theta(self):
       return atan2(self.im, self.re)
    @theta.setter
    def theta(self, to):
       raise ReadOnlyError()

    @property
    @confidence(55)
    def A(self):
       return self.mag
    @A.setter
    def A(self, to):
       raise ReadOnlyError()

    @property
    @confidence(55)
    def phi(self):
       return self.theta
    @phi.setter
    def phi(self, to):
       raise ReadOnlyError()


    def __add__(self, other):
        if type(other) is type(self):
            return Phasor(self.re + other.re, self.im + other.im)
        else:
            raise TypeError(f"Can't add Phasor by {type(other)}")

    def __sub__(self, other):
        if type(other) is type(self):
            return Phasor(self.re - other.re, self.im - other.im)
        else:
            raise TypeError(f"Can't subtract Phasor by {type(other)}")

    def __mul__(self, other):
        if type(other) is type(self):
            return Phasor.fromPolar(self.mag * other.mag, self.phi + other.phi)
        else:
            raise TypeError(f"Can't multiply Phasor by {type(other)}")

    def __div__(self, other):
        if type(other) is type(self):
            return Phasor.fromPolar(self.mag / other.mag, self.phi - other.phi)
        else:
            raise TypeError(f"Can't divide Phasor by {type(other)}")

    def __rdiv__(self, other):
        # Just handles the reciprical
        if other == 1:
            return Phasor.fromPolar(other / self.mag, self.phi * -other)
        else:
            raise TypeError(f"Can't divide {type(other)} by Phasor")

    def __pow__(self, exp):
        # Only gets the square root
        if exp == (1/2):
            Phasor.fromPolar(self.mag**exp, self.phi * exp)
        else:
            # raise TypeError()
            warn('Using untested Phasor exponents')
            Phasor.fromPolar(self.mag**exp, self.phi * exp)

    def __str__(self):
        return self.asPolarDeg(eval=True)

    def __abs__(self):
        return self.mag

    def _eval_power(self, exp):
        return self.__pow__(exp)



@reprise
class FastPhasor(complex):
    # def __init__(self, real, img):
        # super().__init__(real, img)
        # self.theta = self.period

    def getQuadrant(self):
        NotImplemented

    def angularFrequency(self):
        return 2 * math.pi * frequency()

    def frequency(self):
        NotImplemented

    def period(self):
        return self.theta

    def wavelength(self):
        NotImplemented


    @staticmethod
    @confidence(65)
    def fromExp(A, phi):
        return FastPhasor.fromPolar(A, phi)

    @staticmethod
    def fromWave(wave:TWave):
        # Amplitude * sin(period * (xOffset - AngularFrequency) * time + phase) + yOffset
        # sineParser = er.group(er.optional(er.number())) + er.group(er.either('sin', 'cos')) + '(' + er.group(er.optional(er.number())) +
        # return FastPhasor.fromExp(wave.amaplitude, wave.)
        NotImplemented

    @staticmethod
    def fromSine(amplitude, phase=0, sin=True):
        # phi == phase == xOffset


        # print(f'period = {period}')
        # print(f'frequency = {frequency}')
        # print(f'phase = {phase}')

        return FastPhasor.fromPolar(amplitude, (phase - math.pi) if sin else phase)
        # x, y = symbols('x y')
        # _x = ensureNotIterable(solve(amplitude * cos(period) / x + amplitude * sin(period) / y, x))
        # _y = ensureNotIterable(solve(amplitude * cos(period) / x + amplitude * sin(period) / y, y))
        # return FastPhasor(_x, _y)

        # wave = TWave(amplitude, period=period)
        # return FastPhasor.fromWave(wave)

    @staticmethod
    def fromRect(real, img):
        return FastPhasor(real, img)

    @staticmethod
    def fromPolar(mag, theta):
        return FastPhasor(mag * math.cos(theta), mag * math.sin(theta))

    @staticmethod
    def fromPolarDeg(mag, theta):
        return FastPhasor(mag * math.cos(math.radians(theta)), mag * math.sin(math.radians(theta)))

    @staticmethod
    def fromComplex(c):
        return FastPhasor(c.real, c.imag)

    @staticmethod
    def fromSymbolic(symbolic:Add):
        assert(isinstance(symbolic, Add))
        re, im = symbolic.evalf().args
        try:
            return FastPhasor(re, im)
        except TypeError:
            return FastPhasor(im, re)


    # def asExp(self) -> "A*e^(j*phi)":
        # return self.A * math.e**(_i * _N(self.phi, n=5))

    # def asSine(self, angFeq, sin=False, eval=True) -> "A*cos(ang*t+theta)":
        # if sin:
        #     return im(self.A * math.e ** (_i * (angFeq*t + self.phi)))
        # else:
        #     return (self * _e ** (_i*angFeq*t)).im.simplify()

    def asRect(self)->str:
        return f"{self.real} + {self.im}j"

    def asPolar(self) -> "(mag, theta)":
        return (self.mag, self.theta)

    def asPolarDeg(self) -> "(mag, theta)":
        return (self.mag, math.degrees(self.theta))


    @property
    def mag(self):
       return math.sqrt((self.re**2) + (self.im**2))

    @property
    def theta(self):
       return math.atan2(self.im, self.re)
       # return math.atan(self.im / self.re)

    @property
    @confidence(55)
    def A(self):
       return self.mag

    @property
    @confidence(55)
    def phi(self):
       return self.theta

    @property
    def re(self):
        return self.real

    @property
    def im(self):
        return self.imag

    @property
    def amplitude(self):
        return self.A

    @property
    def phase(self):
        return self.phi

    @property
    def r(self):
        return self.mag



    #! v(t) == r*cos(ohmega*t+phi) == (r*_e**(j*ohmega)... dang it!

    def __add__(self, other):
        if isinstance(other, Basic):
            return other + self
        else:
            return FastPhasor.fromComplex(super().__add__(other))

    def __sub__(self, other):
        if isinstance(other, Basic):
            return other + self
        else:
            return FastPhasor.fromComplex(super().__sub__(other))

    def __mul__(self, other):
        if isinstance(other, Basic):
            return other + self
        else:
            return FastPhasor.fromComplex(super().__mul__(other))

    def __div__(self, other):
        if isinstance(other, Basic):
            return other + self
        else:
            return FastPhasor.fromComplex(super().__div__(other))

    def __rdiv__(self, other):
        if isinstance(other, Basic):
            return other + self
        else:
            return FastPhasor.fromComplex(super().__rdiv__(other))

    def __pow__(self, exp):
        if isinstance(other, Basic):
            return other + self
        else:
            return FastPhasor.fromComplex(super().__pow__(other))

    def __str__(self):
        return f'{round(self.mag, 5)} ∠ {round(math.degrees(self.theta), 5)}°'



    # def __abs__(self):
    #     return self.mag

    # def _eval_power(self, exp):
    #     return self.__pow__(exp)



# derivative of a sinusoid == Phasor.fromSine(sinusoid) * _i * sinusoid.angFeq
# indefinite integral of a sinusoid == Phasor.fromSine(sinusoid) / (_i * sinusoid.angFeq)
# adding sinusoids of the same frequency is equal to adding their phasors

class Sinusoid(TWave):
    @property
    def phasor(self):
       return Phasor.fromSine(self.amplitude, self.phase)

    def diff(self, var=None):
        if var:
            NotImplemented
        else:
            return self.phasor * _i * self.angular_frequency

    def integrate(self):
        return self.phasor / (_i * self.angular_frequency)



def createPlot(xPoints, *yPoints, xlabel='x', ylabel='y', size=(15, 8), labels=None, grid=True, title='Copeland Carter', copy=False, show=False, dpi='figure', labelSize='xx-large'):
    fig = plt.figure(figsize=size)
    ax  = fig.add_subplot(111)
    ax.set_position([0, 0, 1, 1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if grid:
        ax.grid()

    # plot = ax.plot if type == 'rect' else ax.polar

    if labels:
        for y, label in zip(yPoints, labels):
            ax.plot(xPoints, y, label=label)
    else:
        for y in yPoints:
            ax.plot(xPoints, y)

    if labels or show:
        ax.legend(fontsize=labelSize)
    elif show:
        ax.show()

    if copy:
        fig.savefig("/tmp/imageCopyDeamon.png", bbox_inches='tight', dpi=dpi)

    return fig, ax


def createPolarPlot(thetaPoints, *rPoints, size=(15, 8), labels=None, grid=True, legendLoc='center', title='Copeland Carter', copy=False, show=False, dpi='figure'):
    fig = plt.figure(figsize=size)
    ax  = fig.add_subplot(111, projection='polar')
    # ax.set_position([0, 0, 1, 1])
    ax.set_rorigin(0)
    # ax.set_ylim(2, 6)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    plt.title(title)
    if grid:
        ax.grid(which='major', linestyle='-', linewidth='0.25', color='black')
        # ax.grid(which='minor', linestyle='--', linewidth='0.15', color='black')

    # plot = ax.plot if type == 'rect' else ax.polar
    # ax.minorticks_on()

    if labels:
        for y, label in zip(rPoints, labels):
            ax.plot(thetaPoints, y, label=label)
    else:
        for y in rPoints:
            ax.plot(thetaPoints, y)

    if labels or show:
        ax.legend(fontsize='x-large', loc=legendLoc, frameon=False)
    elif show:
        ax.show()

    if copy:
        fig.savefig("/tmp/imageCopyDeamon.png", bbox_inches='tight', dpi=dpi)

    return fig, ax




N = ReferenceFrame('N')
ml=MappingList

ans = Symbol('ans')

ll=parallel
s=series


# phasor mult is
# mult mags then add angles

# division is
# mult mags then sub angles


# resistor: (I think)
# v_R == i_R(t)*R
# v_R == V_m * cos(2*pi*f*t)
# i_R(t) == (V_m / R) * cos(ohmega * t) == i_m*cos(ohmega*t)

# inductor:
# v_L(t) == L*((Derivative(i_L(t), t)))
# V_L == j*ohmega*L*i_L -- i_L and V_L are bolded

# whenever you mult by j its a 90 degree phase shift

# capacitor:
# i_C(t) == C*Derivative(v_C(t), t) == j*ohmega*C*v_c -- CONFIRM

# Derivative(cos(2*pi*f*t), t) == (-2*pi*f)*sin(2*pi*f*t)

# its not the voltage/current, its the time rate of change, which is how they get out of phase in inductors/capacitors
