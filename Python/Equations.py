from sympy import *
from Cope import *
# from include import unknown
from clipboard import copy
import EasyRegex as er
import re
from warnings import warn


'''
 # Format is atomames: sympyEquation
_equationFunctions = {}

#* This works marvelously, don't discount it
# Make a new function that acts like ohmsLaw or newtonsLaw, but with an arbitrary equation
def equationFunction(sympyExpr: Eq, psuedonyms={}, help:str=None, name='', defaults={}):
    ALLOW_NONSENSE_PARAMS = False
    atoms = sympyExpr.atoms(Symbol)
    atomNames = [a.name for a in atoms]
    if len(psuedonyms):
        todo('psuedonyms', False)
    if str is None:
        help = pretty(sympyExpr)

    # Add to equation search
    _equationFunctions[tuple(atomNames)] = (name, sympyExpr)

    def func(*args, addSymbolically=False, solveSymbolically:Symbol=None, **kwargs):
        kwargs = ZerosDict(kwargs)
        if kwargs['help']:
            print(help)
            return

        if not len(kwargs) and not addSymbolically and not solveSymbolically:
            print(pretty(sympyExpr))
            return sympyExpr

        # Set the default params
        for key, val in defaults.items():
            if kwargs[key] is None:
                kwargs[key] = val

        # Just ignore everything else and return the symbolic equation for the wanted value
        if solveSymbolically is not None:
            # Assume we're solving for whatever is on the left
            if type(solveSymbolically) is bool and solveSymbolically == True:
                return sympyExpr.rhs
            else:
                # assert(type(solveSymbolically) is Symbol, f"Incorrect type '{type(solveSymbolically)}' for solveSymbolically parameter (accepts bool or Symbol)")
                return ensureNotIterable(solve(sympyExpr, solveSymbolically))

        # Add whatever we don't have as a symbol instead of solving for it or throwing an error
        if addSymbolically:
            return sympyExpr.subs(kwargs, simultaneous=True)
            # for atom in atoms:
                # if kwargs[atom.name] is None:
                    # kwargs[atom.name] = atom

        # Parameter checking
        if len(args):
            raise TypeError('Please specify all your parameters by name')
        if not ALLOW_NONSENSE_PARAMS:
            for input in kwargs.keys():
                if input not in atomNames:
                    raise TypeError(f'Parameter {input} not in equation:\n{pretty(sympyExpr)}')

        u = unknown(kwargs, *atomNames)
        if u:
            # MappingList, in case there's multiple solutions
            symbolicAns = MappingList(solve(sympyExpr, Symbol(u)))
            if not len(symbolicAns):
                debug(f'Unable to solve {sympyExpr} for {u}', clr=Colors.ALERT)
            return ensureNotIterable(symbolicAns.subs(known(kwargs)))
        else:
            raise TypeError(f'Incorrect number of parameters. Parameters are: {tuple(atomNames)}')

    return func

todo('add a loose search parameter to searchEquations', False)

# angular motion
centripitalMotion = equationFunction(parse_expr('Eq(a, (v**2)/r)'), {'velocity': 'v', 'radius': 'r', 'acceleration': 'a', 'accel': 'a'},
""" a = centripetal acceleration, v = tangential (regular) velocity, r = radius
    The centripetal acceleration is always pointing towards the center
""", name='centripitalMotion')

angularSpeed = equationFunction(parse_expr('Eq(angs, dang/dt)'), name='angularSpeed', help=
"""angs = angular speed, dang = delta angle, dt =kineticEnergy(m=.1, v=1000), kineticEnergy(m=1325, v=25).evalf(), kineticEnergy(m=.5*(1/1000), v=15000) delta time""")
angularAccel = equationFunction(parse_expr('Eq(angAccel, dangs/dt)'), name='angularAccel', help=
"""angAccel = angular acceleration, dangs = delta anglular speed, dt = delta time""")

# newtons laws
#  = equationFunction(parse_expr('Eq(netForce_x, mass * acceleration_x'))
newtonsLawWeight = equationFunction(parse_expr('Eq(w, m * a)'), name='newtonsLawWeight', help=
"""w = weight, m = mass, a = acceleration""")
dragEqu = equationFunction(parse_expr('Eq(R, -b * v)'), name='dragEqu', help=
"""R = drag force, v = velocity, and I have no CLUE what b is """)

# friction
# What the heck is f_smax???
# 'Eq(f_smax, staticFriction * netForce) ',
# Motion equations
motionHelp = """a = constant acceleration (usually gravity), t = time
v = velocity, initv = starting velocity (velocity at time 0)
p = position, initp = starting position (position at time 0)"""
motionEqu1 = equationFunction(parse_expr("Eq(v, initv - a * t)"), name='motionEqu1', help=motionHelp)
motionEqu2 = equationFunction(parse_expr("Eq(p, initp + initv * t - (1/2)*a*(t**2))"), name='motionEqu2', help=motionHelp)
motionEqu3 = equationFunction(parse_expr("Eq(v, sqrt((initv**2) - 2 * a * (p-initp)))"), name='motionEqu3', help=motionHelp)

ohmsLaw = equationFunction(parse_expr("Eq(v, i*r)"), name='ohmsLaw', help=
"""v = voltage, i = current, r = resistance """)
# ohmsLaw = equationFunction(parse_expr("Eq(p, i*v)"), name='ohmsLaw', help=
# """v = voltage, i = current, r = resistance """)
newtonsLaw = equationFunction(parse_expr("Eq(f, m*a)"), name='newtonsLaw', help=
"""f = force, m = mass, a = acceleration (often gravity) """)

conductanceSolver = equationFunction(parse_expr("Eq(iGn, (i*Gn)/Geq)"), name='conductanceSolver', help=
"""Geq = equivalent conductance
Gn = conductance of the chosen resistor
i = current into resistor
iGn = current through the chosen resistor""")

swapAngLinearSpeed = equationFunction(parse_expr("Eq(vlin, r*vang)"), name='swapAngLinearSpeed', help=
"""vlin = linear (tangential) velocity, r = radius, vang = angular velocity
linear velocity is in m/s, radius is in meters, and angular velocity is in radians/second
(pretty sure)""")

kineticEnergy = equationFunction(parse_expr("Eq(ke, (1/2) * m * v**2)"), name="kineticEnergy", help=
"""ke = kinetic energy, m = mass, v = velocity """)

gravityPotentialEnergy = equationFunction(parse_expr("Eq(peg, m * g * h)"), name="gravityPotentialEnergy", help=
"""peg = potential energy due to gravity, m = mass, g = gravity acceleration, h = height""",
defaults={"g": 9.805})

springPotentialEnergy = equationFunction(parse_expr("Eq(pes, (1/2) * k * (dx**2))"), name="springPotentialEnergy", help=
"""pes = potential energy in a spring, dx = delta x from equilibrium position, k = the spring's constant (pretty sure)
dx is in meters, k is in Newtons/meter""")

loopPotentialEnergy = equationFunction(parse_expr("Eq(pe, m * g * 2 * r)"), name="gravityPotentialEnergy", help=
"""pe = potential energy due to... the centripetal force?, m = mass, g = gravity acceleration, r = radius of the loop""",
defaults={"g": 9.805})

totalEnergy = equationFunction(parse_expr("Eq(te, ke + pe)"), name="totalEnergy", help=
"""te = total energy, ke = kenetic energy, pe = potential energy""")

invertingOpAmpHelp = """Av = Voltage Gain
Rf = the resistor connecting the op amp Vout and the op amp negative in terminals
Rin = the resistor between Vin and the op amp negative in
Vout = the output voltage of the op amp
Vin = voltage connected to Rin"""

noninvertingOpAmpHelp = """Av = Voltage Gain
Rf = the resistor connecting the op amp Vout and the op amp negative in terminals
R = the resistor between ground and the op amp negative in
Vout = the output voltage of the op amp
Vin = voltage connected to the op amp positive in"""

opAmpOutput = equationFunction(parse_expr("Eq(Vout, Av * Vin)"),  name='opAmpOutput', help=
"""Vout = the output voltage of the op amp, Av = Voltage Gain, Vin = voltage in """)

invertingOpAmpGain   = equationFunction(parse_expr("Eq(Av, -(Rf / Rin))"), name='invertingOpAmpGain',   help=invertingOpAmpHelp)
noninvertingOpAmpGain   = equationFunction(parse_expr('Eq(Av, 1 + (Rf / R))'), name="noninvertingOpAmpGain",   help=noninvertingOpAmpHelp)
# noninvertingOpAmpOutput = equationFunction(parse_expr('Eq(, )'), name="noninvertingOpAmpOutput", help=noninvertingOpAmpHelp)

capacitor = equationFunction(parse_expr('Eq(q, C * v)'), name='capacitor', help=
"""C = capacitance (farads), v = applied voltage, q = charge """)

plateCapaitor = equationFunction(parse_expr('Eq(C, (E*A)/d'), help=
"""C = Capacitance, A = surface area of each plate, d = distance between the plates, E = permittivity of the material between the plates""")

def rps2angV(rps) -> 'radians/second':
    return 2*pi * rps
# 58.5v max batteries
# 52v 19.2A -- 1850W --
# 48v 2000W

# This only kinda works
def searchEquations(*whatWeHave):
    usableEqus = []
    for atoms, eq in _equationFunctions.items():
        # This tells us there's only 1 unknown
        # This will not work
        missingCnt = 0
        for i in whatWeHave:
            if i not in atoms:
                missingCnt += 1
        if missingCnt == 1:
            usableEqus.append(eq)
    # return usableEqus
    for i in usableEqus:
        print(f'{i[0]}:\n{pretty(i[1])}')

'''


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

# returns the dict d without any of the stuff equal to None in it
def known(d: dict, *obj: str):
    newd = {}
    for key, val in d.items():
        if val is not None:
            newd[key] = val
    return newd



electronicsNamespace = {
    'v': 'voltage (volts)',
    'i': 'current (amps)',
    'r': 'resistance (ohms)',
    'Geq': 'equivalent conductance',
    'Gn': 'conductance of a given resistor',
    'iGn': 'current through a given resistor',
    'vL': 'induced voltage (volts)',
    'Av': 'Voltage Gain (volts)',
    'Vin': 'voltage in (volts)',
    'Vout': 'voltage out (volts)',
    # 'Rf': 'the resistor connecting the op amp Vout and the op amp negative in terminals',
    # 'Rin': 'the resistor between Vin and the op amp negative in',
    'C': 'capacitance (farads)',
    'q': 'charge',
    'A': 'surface area of each capacitor plate',
    'd': 'distance between the capacitor plates',
    # 'e': 'permittivity of the material between the plates'
    'εr': 'relative permittivity for the medium',
    'ε0': 'permittivity of space (constant, equal to 8.854x10-12 F/m)',
    'w': 'energy (joules)',
    'phi': 'magnetic flux (teslas (Webers / meter))',
    't': 'time (seconds)',
    'L': 'inductance (Henrys)',
    'iL': "induced current",
    'N': 'number of turns of the wire',
    'vs': 'secondary voltage',
    'vp': 'primary voltage',
    'is': 'secondary current',
    'ip': 'primary current',
    'Ns': 'secondary turns',
    'Np': 'primary turns',
}

physicsNamespace = {
    'a': 'acceleration (meters/second^2)',
    'v': 'velocity (meters/second)',
    'r': 'radius (meters)',
    'angA': 'angular acceleration (radians/second^2)',
    'vang': 'angular velocity (radians/second)',
    'angs': 'angular speed (radians/second)',
    'dangs': 'delta anglular speed (radians/second)',
    'dang': 'delta angle (radians)',
    'dt': 'delta time (seconds)',
    'w': 'weight (Newtons)',
    'W': 'work (Joules)',
    'm': 'mass (kilograms)',
    'b': 'NO idea',
    'f': 'force (Newtons)',
    'R': 'drag force (Newtons)',
    'vlin': 'linear (tangential) velocity',
    'ke': 'kinetic energy (Joules)',
    'peg': 'potential energy due to gravity (Joules)',
    'pes': 'potential energy in a spring (Joules)',
    'pec': 'potential energy due to... the centripetal force?',
    'g': 'gravity acceleration (meters/second^2)',
    'h': 'height (meters)',
    'dx': 'delta x (meters)',
    'k': 'the spring\'s constant (Newtons/meter)',
    'te': 'total energy (Joules)',
    'ke': 'kenetic energy (Joules)',
    'pe': 'potential energy (Joules)',
    't': 'time (seconds)',
    'initv': 'starting velocity (meters/second)',
    'vi': 'starting velocity (meters/second)',
    'vf': 'final ending velocity (meters/second)',
    'pos': 'position (meters?)',
    'initp': 'starting position (meters/second)',
    'mag': '"magnitude of the displacement vector"',
    'me': "Mechanical Energy (Joules)",
    'mei': 'Initial mechanical energy (Joules)',
    'mef': 'Final mechanical energy (joules)',
    'wdaf': '"Work Done Against Friction"',
    'p': 'Power (watts)',
    'pef': 'Final potential energy',
    'pei': 'Initial potential energy',
    'kef': 'Final kinetic energy',
    'kei': 'Initial kinetic energy',
    'd': 'displacement (meters)',
    'D': 'distance (meters)',
}

_latex = latex

@reprise
class Equation:
    ALLOW_NONSENSE_PARAMS = False

    @staticmethod
    def parseExpr(eq:str) -> Expr:
        doubleEqRegex = (er.group(er.anything() + er.matchMax()) + er.match('==') + er.group(er.anything() + er.matchMax())).compile()
        eqRegex       = (er.group(er.anything() + er.matchMax()) + er.match('=')  + er.group(er.anything() + er.matchMax())).compile()
        eq = re.subn(doubleEqRegex, r'Eq(\g<1>, \g<2>)',     eq, 1)[0]
        eq = re.subn(eqRegex,       r'\g<2> - \g<1>',        eq, 1)[0]
        return parse_expr(eq)

    def __init__(self, equation:str, parameters={}, help='', defaults={}, psuedonyms={}):
        self._paramsHelp = parameters
        self._help = help
        self.defaults = defaults
        self.psuedonyms = psuedonyms

        self.expr = self.parseExpr(equation)
        self.atoms = self.expr.atoms(Symbol)
        self.atomNames = [a.name for a in self.atoms]


        if len(psuedonyms):
            todo('psuedonyms', False)

        # if help is None:
            # self._help = pretty(self.expr)

        # Add to equation search
        todo()
        # _equationFunctions[tuple(atomNames)] = (name, sympyExpr)

    def help(self):
        print(pretty(self.expr), ', where:')

        # for param, help in self._paramsHelp.items():
        #     if param in self.atomNames: # or Symbol(param) in self.atoms:
        #         print(f'{param} = {help}')

        for param in self.atomNames:
            if param in self._paramsHelp.keys():
                print(f'    {param} = {self._paramsHelp[param]}')
            else:
                print(f'    {param}')


        print(self._help)

    def solve(self, for_:Symbol=None):
        # Just ignore everything else and return the symbolic equation for the wanted value
        # Assume we're solving for whatever is on the left
        if for_ is None:
            return sympyExpr.rhs
        else:
            # assert(type(solveSymbolically) is Symbol, f"Incorrect type '{type(solveSymbolically)}' for solveSymbolically parameter (accepts bool or Symbol)")
            return ensureNotIterable(solve(self.expr, for_))

    def __call__(self, *args, addSymbolically=False, **kwargs):
        kwargs = ZerosDict(kwargs)

        # Parameter checking
        if len(args):
            raise TypeError('Please specify all your parameters by name')
        if not self.ALLOW_NONSENSE_PARAMS:
            for input in kwargs.keys():
                if input not in self.atomNames:
                    raise TypeError(f'Parameter {input} not in equation:\n{pretty(self.expr)}')

        # If we're calling with no parameters
        if not len(kwargs) and not addSymbolically:
            # print(pretty(self.expr))
            return self.expr

        # Set the default params
        for key, val in self.defaults.items():
            if kwargs[key] is None:
                kwargs[key] = val

        # Add whatever we don't have as a symbol instead of solving for it or throwing an error
        if addSymbolically:
            return self.expr.subs(kwargs, simultaneous=True)

        # The important stuff
        u = unknown(kwargs, *self.atomNames)
        if u:
            # MappingList, in case there's multiple solutions
            symbolicAns = MappingList(solve(self.expr, Symbol(u)))
            if not len(symbolicAns):
                debug(f'Unable to solve {self.expr} for {u}', clr=Colors.ALERT)
            return ensureNotIterable(symbolicAns.subs(known(kwargs)))
        else:
            raise TypeError(f'Incorrect number of parameters. Parameters are: {tuple(self.atomNames)}')

    def __str__(self):
        return pretty(self.expr)

    def copy(self, latex=True):
        if latex:
            copy(_latex(self.expr))
        else:
            # f"{_debugGetMetaData().lineno}\n\t->
            copy(pretty(self.expr))

    def applicable(self, *atoms:str, loose=False) -> bool:
        if loose:
            return bool(len(set(atoms).intersection(self.atomNames)))
        else:
            return bool(len(set(atoms).issuperset(self.atomNames)))

# angular motion
centripitalMotion = Equation('a == (v**2)/r', {
        'a': 'centripetal acceleration',
        'v': 'tangential (regular) velocity',
        'r': 'radius'
    },
    'The centripetal acceleration is always pointing towards the center'
)

angularSpeed = Equation('angs == dang/dt', physicsNamespace)
angularAccel = Equation('angA == dangs/dt', physicsNamespace)
newtonsLawWeight = Equation('w == m * a', physicsNamespace)
dragEqu = Equation('R == -b * v', physicsNamespace)

# 'Eq(f_smax, staticFriction * netForce) ',

# Motion equations
newtonsLaw = Equation("f == m*a", physicsNamespace)
motionEqu1 = Equation("v == initv - a * t", physicsNamespace)
motionEqu2 = Equation("p == initp + initv * t - (1/2)*a*(t**2)", physicsNamespace)
motionEqu3 = Equation("v == sqrt((initv**2) - 2 * a * (p-initp))", physicsNamespace)

# Electronics
ohmsLaw = Equation("v == i*r", electronicsNamespace)
conductanceSolver = Equation("iGn == (i*Gn)/Geq",{
    'Geq': 'equivalent conductance',
    'Gn': 'conductance of the chosen resistor',
    'i': 'current into resistor',
    'iGn': 'current through the chosen resistor',
})

swapAngLinearSpeed = Equation("vlin == r*vang", physicsNamespace,
    help='linear velocity is in m/s, radius is in meters, and angular velocity is in radians/second (pretty sure)'
)

kineticEnergy = Equation("ke == (1/2) * m * v**2", physicsNamespace)
gravityPotentialEnergy = Equation("peg == m * g * h", physicsNamespace, defaults={"g": 9.805})
springPotentialEnergy = Equation("pes == (1/2) * k * (dx**2)", physicsNamespace, help='dx is in meters, k is in Newtons/meter')
loopPotentialEnergy = Equation("pec == m * g * 2 * r", physicsNamespace, defaults={"g": 9.805})
totalEnergy = Equation("te == ke + pe", physicsNamespace)
mechanicalEnergy = Equation("me == pe + ke")
workAgainstNonConservativeForce = Equation('mei - wdaf == mef', physicsNamespace)

# work == delta KE
workEqu = Equation("W == f * mag * cos(theta)", physicsNamespace)
workSimple = Equation("W == f * d", physicsNamespace)
workEnergyTheorem = Equation("W == ((1/2) * m * vf**2) - ((1/2) * m * vi ** 2)", physicsNamespace, help='Work is the net work')
workDoneByGravity = Equation("W == m * g * h", physicsNamespace, defaults={"g": 9.805})
conservativeForcePotentialEnergy = Equation('pef - pei == -Integral(Fx, x)', physicsNamespace)

springForce = Equation("f == -k*x", physicsNamespace)
springWork = Equation("W == (1/2) * k * x**2", physicsNamespace)

physicsPower = Equation('p = W / s', physicsNamespace)
physicsPower2 = Equation('p = f / v', physicsNamespace)


opAmpOutput = Equation("Vout == Av * Vin", {
    'Vout': 'the output voltage of the op amp',
    'Av': 'Voltage Gain, Vin = voltage in'
})

invertingOpAmpGain    = Equation("Av == -(Rf / Rin)", {
    'Av': 'Voltage Gain',
    'Rf': 'the resistor connecting the op amp Vout and the op amp negative in terminals',
    'Rin': 'the resistor between Vin and the op amp negative in',
    'Vout': 'the output voltage of the op amp',
    'Vin': 'voltage connected to Rin',
})

noninvertingOpAmpGain = Equation('Av == 1 + (Rf / R)', {
    'Av': 'Voltage Gain',
    'Rf': 'the resistor connecting the op amp Vout and the op amp negative in terminals',
    'R': 'the resistor between ground and the op amp negative in',
    'Vout': 'the output voltage of the op amp',
    'Vin': 'voltage connected to the op amp positive in',
})

capacitor = Equation('q == C * v', electronicsNamespace)
plateCapaitor = Equation('C == er*e0 * A/d', electronicsNamespace,  defaults={'e0': 8.854*10**(-12)})
capacitorEnergy = Equation('w == (1/2)*C*v**2', electronicsNamespace)

# the emf induced in an electric circuit is proportional to the rate of change of the mangetic flux linking the circuit
faradayslaw = Equation('vL(t) == Derivative(phi, t)', electronicsNamespace)
inductance = Equation('vL(t) == L * Derivative(iL(t), t)', electronicsNamespace)
# Changing currents in coils induce coil voltages opposite current changes
LenzsLaw = Equation('L = vL(t) / Derivative(i, t)', electronicsNamespace)

# inductorCurrent = iL(t) == (1/L) * Integral(high=t, low=0, vL(dummy)ddummy) + i_0

#? Vp = -Np * Derivative(phi, t)

transformerVoltage = Equation('Vs / Vp == Ns / Np', electronicsNamespace)
transformerCurrent = Equation('Is / Ip == Np / Ns', electronicsNamespace)


# inductorEnergy = w=integral(high=t, low=t_0, p(t))

#? capacitorVoltage ( vc(t) = 1/C * integrral(t, 0, ???))
#? w = L*integral(i(t), low=0, ???)


# watts == joules / second


def searchEquations(*whatWeHave, loose=True, names=True):
    rtn = []
    for i in whatWeHave:
        if i not in physicsNamespace.keys():
            warn(f"{i} not a valid variable name")

    for name, i in globals().items():
        if type(i) is Equation and i.applicable(*whatWeHave, loose=loose):
            rtn.append(name if names else i)

    return rtn



# Units
# Milli
m=1/1000
# Mega
M=100000
# Kilo
k=1000
# Micro
mu=1/100000
# Nano
n=1/100000000
# Default is kilograms
gram=1/1000
# Default is seconds
h=1/3600
# Defualt is meters
cm=1/100
mm=1/1000
km=1000
g=9.805
