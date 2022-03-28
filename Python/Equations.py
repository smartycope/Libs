from sympy import *
from Cope import *
# from include import unknown
from clipboard import copy
import EasyRegex as er
import re
from warnings import warn
from copy import copy as objcopy, deepcopy


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

    def func(*args, symbolic=False, solveSymbolically:Symbol=None, **kwargs):
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

UnitModifier = namedtuple('UnitModifier', 'description, psuedonyms, abbrev')
initial = UnitModifier(
    description=lambda d: 'initial ' + d,
    psuedonyms= lambda abbrev: ('init ' + abbrev, 'initial ' + abbrev, abbrev + 'i'),
    abbrev=     lambda abbrev: abbrev + '_i'
)
final = UnitModifier(
    lambda d: 'final ' + d,
    lambda abbrev: ('final ' + abbrev, 'fin ' + abbrev, abbrev + 'f'),
    lambda abbrev: abbrev + '_f'
)
delta = UnitModifier(
    lambda d: 'delta ' + d,
    lambda abbrev: (),
    lambda abbrev: 'd' + abbrev
)
time = UnitModifier(
    lambda d: d + ' at time t',
    lambda abbrev: (abbrev + 't', abbrev + '(t)'),
    lambda abbrev: abbrev + '_t'
)
primary = UnitModifier(
    lambda d: 'primary ' + d,
    lambda abbrev: ('1 ' + abbrev, 'primary ' + abbrev, abbrev + '1', abbrev + '_1', abbrev + 'p', 'p' + abbrev, 'a' + abbrev, abbrev + 'a', abbrev + '_a'),
    lambda abbrev: abbrev + '_p'
)
secondary = UnitModifier(
    lambda d: 'secondary ' + d,
    lambda abbrev: ('2 ' + abbrev, 'secondary ' + abbrev, abbrev + '2', abbrev + '_2', abbrev + 's', 's' + abbrev, 'b' + abbrev, abbrev + 'b', abbrev + '_b'),
    lambda abbrev: abbrev + '_p'
)
equivalent = UnitModifier(
    lambda d: 'equivalent ' + d,
    lambda abbrev: (),
    lambda abbrev: abbrev.capitalize() + 'eq'
)
throughCapacitor = UnitModifier(
    lambda d: d + ' through capacitor C',
    lambda abbrev: (abbrev + 'C',),
    lambda abbrev: abbrev + '_C'
)
throughInductor = UnitModifier(
    lambda d: d + ' through inductor L',
    lambda abbrev: (abbrev + 'L',),
    lambda abbrev: abbrev + '_L'
)
throughResistor = UnitModifier(
    lambda d: d + ' through resistor R',
    lambda abbrev: (abbrev + 'R',),
    lambda abbrev: abbrev + '_R'
)
inMod = UnitModifier(
    lambda d: d + ' in',
    lambda abbrev: (abbrev + 'in', abbrev.capitalize() + 'in', abbrev.capitalize() + '_in'),
    lambda abbrev: abbrev + '_in'
)
outMod = UnitModifier(
    lambda d: d + ' out',
    lambda abbrev: (abbrev + 'out', abbrev.capitalize() + 'out', abbrev.capitalize() + '_out'),
    lambda abbrev: abbrev + '_out'
)
phasor = UnitModifier(
    lambda d: d + ' phasor',
    lambda abbrev: ('ph_' + abbrev),
    lambda abbrev: 'ph' + abbrev
)

@reprise
class Unit:
    def __init__(self, abbrev:str, description:str='', units:str=None, psuedonyms=set(), mods=()):
        description = description.lower()
        psuedonyms=set(objcopy(psuedonyms))

        for mod in mods:
            description = mod.description(description)
            psuedonyms.union(mod.psuedonyms(abbrev))
            abbrev = mod.abbrev(abbrev)

        psuedonyms.add(description)

        self.abbrev = abbrev
        self.description = description.capitalize()
        self.units = units
        self.psuedonyms = psuedonyms
        self.mods = mods
        self.symbol = Symbol(abbrev)

    def help(self):
        unitStr = f' ({self.units})' if self.units else ''
        deStr = f' = {self.description}' if len(self.description) else ''
        return self.abbrev + deStr + unitStr

    def __str__(self):
        return self.abbrev

    def __eq__(self, other):
        if type(other) == type(self):
            if self.abbrev == other.abbrev:
                # assert(self.units == other.units)
                # assert(self.delta == other.delta)
                # assert(self.final == other.final)
                # assert(self.initial == other.initial)
                return True
            else:
                return False
        elif type(other) is str:
            return other == self.abbrev or other in self.psuedonyms
        elif isinstance(other, Symbol):
            return other == self.symbol or str(other) == self.abbrev or str(other) in self.psuedonyms
        else:
            raise TypeError()

    def __hash__(self):
        return hash((self.abbrev))

    def generateVariant(self, *mods, keepSelf=True):
        # A default that still captures all positional arguements
        if not len(mods):
            mods=(delta, initial, final)
        rtn = {self} if keepSelf else set()
        for mod in mods:
            rtn.add(Unit(self.abbrev, self.description, self.units, set(), self.mods + (mod,)))
        return rtn


class Namespace(list):
    def __init__(self, name, *args):
       self.name = name
       super().__init__(flatten(args))

    def __getitem__(self, i):
        # Searches in reversed because replace adds to the end
        if type(i) is int:
            return super().__getitem__(i)
        else:
            # apparently len(listData)-listData[::-1].index(x)-1 gets the last occurance in a list
            try:
                return super().__getitem__(len(self)-list(reversed(self)).index(i)-1)
            except ValueError:
                # TODO this isn't a very good solution.
                return Unit(i)

    def __setitem__(self, i, to):
        if type(i) is int:
            super().__setitem__(i, to)
        else:
            try:
                super().__setitem__(self.index(i), to)
            except ValueError:
                if type(to) == type(self):
                    self.append(to)
                elif type(to) is str:
                    self.append(Unit(to))
                else:
                    raise TypeError()

    def __eq__(self, other):
        return self.name == other.name

    def replace(self, *units):
        new = objcopy(self)
        new.extend(units)
        return new


fundamentalMath = Namespace('math',
    Unit('c', 'circumerence of a circle'),
    Unit('r', 'radius'),
    Unit('A', 'area'),
)

_latex = latex

#todo Make the help parameters say if they have a default or not

@reprise
class Equation:
    @staticmethod
    def parseExpr(eq:str) -> Expr:
        # Because it's a static method, it won't necisarrily be used in this global scope
        import sympy
        doubleEqRegex = (er.group(er.anything() + er.matchMax()) + er.match('==') + er.group(er.anything() + er.matchMax())).compile()
        eqRegex       = (er.group(er.anything() + er.matchMax()) + er.match('=')  + er.group(er.anything() + er.matchMax())).compile()
        eq = re.subn(doubleEqRegex, r'Eq(\g<1>, \g<2>)',     eq, 1)[0]
        eq = re.subn(eqRegex,       r'\g<2> - \g<1>',        eq, 1)[0]
        # print(N)=
        g = objcopy(sympy.__dict__)
        g['_i'] = I
        g['_e'] = E
        del g['N']
        del g['I']
        del g['E']
        return parse_expr(eq, global_dict=g)

    def __init__(self, equation:str, namespace, help='', defaults={}, psuedonyms={}, tags=set()):
        self.namespace = namespace
        self._help = help
        self.defaults = ZerosDict(defaults)
        self.psuedonyms = psuedonyms
        self.raw = equation
        self.tags = tags

        self.expr = self.parseExpr(equation)
        self.atoms = self.expr.atoms(Symbol)
        self.atomNames = [a.name for a in self.atoms]
        self.units = [(namespace[i] if i in namespace else Unit(i.name)) for i in self.atoms]


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

        for param in self.units:
            print(f'    {param.help()}' + (f' (default: {self.defaults[param]})' if self.defaults[param] is not None else ''))
            # if param in self.namespace:
                # print(f'    {param} = {self.namespace[param]}')
            # else:


        print(self._help)

    def solve(self, for_:Symbol=None):
        # Just ignore everything else and return the symbolic equation for the wanted value
        # Assume we're solving for whatever is on the left
        if for_ is None:
            return self.expr.rhs
        elif for_ in self.atoms:
            # assert(type(solveSymbolically) is Symbol, f"Incorrect type '{type(solveSymbolically)}' for solveSymbolically parameter (accepts bool or Symbol)")
            return ensureNotIterable(solve(self.expr, for_))
        elif for_ in self.atomNames:
            return ensureNotIterable(solve(self.expr, Symbol(for_)))
        else:
            raise TypeError(f"{for_} not in {self.expr}")

    def __call__(self, *args, show=False, symbolic=False, allowNonsenseParams=False, raiseErrors=False, **kwargs):
        kwargs = ZerosDict(kwargs)

        # Parameter checking
        if len(args):
            raise TypeError('Please specify all your parameters by name')
        if not allowNonsenseParams:
            for input in kwargs.keys():
                if input not in self.atomNames:
                    raise TypeError(f'Parameter {input} not in equation:\n{pretty(self.expr)}')

        # If we're calling with no parameters
        if not len(kwargs) and not symbolic:
            # print(pretty(self.expr))
            self.help()
            return
            # return self.expr

        # Set the default params
        for key, val in self.defaults.items():
            if kwargs[key] is None:
                kwargs[key] = val

        # Add whatever we don't have as a symbol instead of solving for it or throwing an error
        if symbolic:
            return self.expr.subs(kwargs, simultaneous=True)

        # If we are given all the variables, and exactly all the variables, just solve, don't try to solve *for* anything
        if set(kwargs.keys()) == set(self.atomNames):
            symbolicAns = MappingList(solve(self.expr))
            if not len(symbolicAns):
                err = Exception(f'Unable to solve {self.expr} for {u}')
                if raiseErrors:
                    raise err
                else:
                    debug(err, clr=Colors.ALERT)
            if symbolic:
                return symbolicAns

            try:
                return ensureNotIterable(symbolicAns.subs(known(kwargs)))
            except AttributeError:
                try:
                    # Yes, yes, I know, this line of code is disgusting. Let me explain.
                    # No no, it is too long. Let me sum up.
                    # Often sympy gives you this: [{<symbol>: <solution>}].
                    # This turns that into just <solution>, but ONLY if it is exactly in that format, and then prints what <symbol> was
                    dict = ensureNotIterable(symbolicAns)
                    print(f'solving for {ensureNotIterable(ensureNotIterable(flatten(dict)).subs(known(kwargs)))}...')
                    return ensureNotIterable(ensureNotIterable(flatten(invertDict(dict))).subs(known(kwargs)))
                except AttributeError:
                    return symbolicAns

        # The important stuff
        u = unknown(kwargs, *self.atomNames)
        if u:
            # MappingList, in case there's multiple solutions
            symbolicAns = MappingList(solve(self.expr, Symbol(u)))

            if not len(symbolicAns):
                err = Exception(f'Unable to solve {self.expr} for {u}')
                if raiseErrors:
                    raise err
                else:
                    debug(err, clr=Colors.ALERT)

            ans=ensureNotIterable(symbolicAns.subs(known(kwargs)))
            if show:
                unit = self.namespace[u].units
                if unit:
                    print(f'{self.namespace[u]} is in {unit}')
            return ans
        else:
            raise TypeError(f'Incorrect number of parameters. Parameters are: {tuple(self.atomNames)}')

    def __str__(self):
        # self.help()
        return pretty(self.expr)

    def copy(self, latex=True):
        if latex:
            copy(_latex(self.expr))
        else:
            # f"{_debugGetMetaData().lineno}\n\t->
            copy(pretty(self.expr))

    def applicable(self, *atoms:str, loose=False, tags=True) -> bool:
        """ Returns True if we can use the variables given to derive a new variable using this equation.
            If loose is set to True, then return True if any of the variables given relate to this equation.
            If tags is set to True, then search the tags as well
        """
        # We want it to be applicable if we have a default for it
        unknownAtomNames = set(self.atomNames).difference(self.defaults.keys())
        if loose:
            l = bool(len(set(atoms).intersection(unknownAtomNames)))
        else:
            l = len(set(atoms).intersection(unknownAtomNames)) == len(unknownAtomNames) - 1

        if tags:
            return l or bool(len(set(atoms).intersection(self.tags)))
        else:
            return l

            # return set(atoms).issuperset(self.atomNames)
            # return set(self.atomNames).issubset(atoms)


###################################### Physics ################################


physics = Namespace('physics',
    Unit('a',    'acceleration', 'meters/second^2').generateVariant(),
    Unit('v',    'velocity', 'meters/second').generateVariant(),
    Unit('r',    'radius', 'meters'),
    Unit('angA', 'angular acceleration', 'radians/second^2').generateVariant(),
    Unit('angV', 'angular velocity', 'radians/second', {'wThing', 'ohmega'}).generateVariant(),
    Unit('angs', 'angular speed', 'radians/second', {'aThing', 'alpha'}).generateVariant(),
    Unit('dangs' 'delta anglular speed', 'radians/second').generateVariant(),
    Unit('dang', 'delta angle', 'radians').generateVariant(),
    Unit('dt',   'delta time', 'seconds'),
    Unit('w',    'weight', 'Newtons'),
    Unit('W',    'work', 'Joules').generateVariant(),
    Unit('m',    'mass', 'kilograms'),
    Unit('b',    'NO idea'),
    Unit('f',    'force', 'Newtons').generateVariant(),
    Unit('R',    'drag force', 'Newtons').generateVariant(),
    Unit('vlin', 'linear (tangential) velocity', 'meters/second').generateVariant(),
    Unit('ke',   'kinetic energy', 'Joules').generateVariant(),
    Unit('peg',  'potential energy due to gravity', 'Joules').generateVariant(),
    Unit('pes',  'potential energy in a spring', 'Joules').generateVariant(),
    Unit('pec',  'potential energy due to... the centripetal force?', 'Joules').generateVariant(),
    Unit('g',    'gravity acceleration', 'meters/second^2'),
    Unit('h',    'height', 'meters').generateVariant(),
    Unit('x',    'x', 'meters').generateVariant(),
    Unit('k',    'the spring\'s constant', 'Newtons/meter'),
    Unit('te',   'total energy', 'Joules').generateVariant(),
    Unit('ke',   'kenetic energy', 'Joules').generateVariant(),
    Unit('pe',   'potential energy', 'Joules').generateVariant(),
    Unit('T',    'Period', 'revolutions/second'),
    Unit('t',    'time', 'seconds'),
    Unit('pos',  'position', 'meters?').generateVariant(),
    Unit('mag',  'vector magnitude').generateVariant(),
    Unit('me',   "Mechanical Energy", 'Joules').generateVariant(),
    Unit('wdaf', '"Work Done Against Friction"').generateVariant(),
    Unit('P',    'Power', 'watts').generateVariant(),
    Unit('p',    'Momentum', 'kilogram meters/second').generateVariant(),
    Unit('ker',  'rotational kinetic energy', 'Joules').generateVariant(),
    Unit('d',    'displacement', 'meters').generateVariant(),
    Unit('D',    'distance', 'meters').generateVariant(),
    Unit('vec',  'generic position vector'),
    Unit('mu_k', 'coefficent of kinetic friction'),
    Unit('mu_s', 'coefficent of static friction'),
    Unit('N',    'Magnitude of the normal force', 'Newtons').generateVariant(),
    Unit('imp',  'Impulse', 'Newton seconds').generateVariant(),
    Unit('I',    'moment of inertia', 'kilogram meters^2').generateVariant(),
    Unit('f_avg','average force over time'),
    Unit('tau',  'torque', 'Newton meters').generateVariant(),
    Unit('kew',  'kinetic energy in a flywheel', 'Joules'),
    Unit('L',    'Angular Momentum', 'kilogram meters^2/second').generateVariant(),
    # Unit(# 'f', 'frequency', 'Hertz'),
)


newtonsLaw = Equation("f == m*a", physics)
newtonsLawGravity = Equation('f == m * g', physics, defaults={'g': 9.805})
newtonsLawFriction = Equation('f == mu * f_N', physics)
# newtonsLawWeight = Equation('w == m * a', physics)
dragEqu = Equation('R == -b * v', physics, tags={'drag'})

thrust = Equation('f == m*v', physics.replace(
    Unit('m', 'Mass Flow Rate', 'kg/s'),
    Unit('v', 'Exhaust velocity', 'm/s')
))

# Frantic notes from the notes page
avgVelocity           = Equation('v == d/t',                                                 physics)
# directionalVeloctiy   = Equation('v == Derivative(vec, t)',                                  physicsNamespace)
# directionalAccel      = Equation('a == Derivative(v, t)',                                    physicsNamespace)
# # directionalAccel2     = Equation('a == Derivative(vec**2, t**2)',                            physicsNamespace) # This throws an error??
# finalVec              = Equation('vecf == veci + vi * dt + (1/2)*a*dt**2',                   physicsNamespace)
finalVelocity         = Equation('vf == vi + a*dt',                                          physics)
finalVelocity2        = Equation('vf**2 == vi**2 + 2*a*dvec',                                physics)
vectorMagnitude       = Equation('mag == sqrt(rx**2 + ry**2)',                               physics)
vectorTheta           = Equation('theta == atan(ry/rx)',                                     physics)
# angularAccelIThink    = Equation('a == (v**2) / r',                                          physicsNamespace)
# angularAccelIThink2   = Equation('a == wThing * r',                                          physicsNamespace)
# angularAccelIThink3   = Equation('a == a_t + a_r',                                           physicsNamespace)
# wThingFinal           = Equation('wThingf == wThingi + aThing * dt',                         physicsNamespace)
# wThingFinal2          = Equation('wThingf**2 == wThingi**2 + 2 * aThing * dtheta',           physicsNamespace)
# springForce           = Equation('fs == -k*dx',                                              physicsNamespace)
# anotherWorkEqu        = Equation('Wnet == deltaKE',                                          physicsNamespace)
# anotherWorkEqu2       = Equation('W == F * dvec',                                            physicsNamespace)
# anotherWorkEqu3       = Equation('W == F * vec * cos(theta)',                                physicsNamespace)
# anotherWorkEqu4       = Equation('W_nc == dke + dpe',                                        physicsNamespace)
# powerSomething        = Equation('P == Derivative(W, t)',                                    physicsNamespace)
# finalTheta            = Equation('thetaf == thetai + wThingi * dt + (1/2) * aThing * dt**2', physicsNamespace)
# powerSomething2       = Equation('P == f * v',                                               physicsNamespace)
# someForceThing        = Equation('f == -Derivative(U, r)',                                   physicsNamespace)
# noClueWhatThisIs      = Equation('s == mag * theta',                                         physicsNamespace)
# whatIsThis            = Equation('T == (2*pi*r) / v',                                        physicsNamespace)
# wThing                = Equation('wThing == Derivative(theta, t)',                           physicsNamespace)
# whatIsThis2           = Equation('T == (2*pi) / wThing',                                     physicsNamespace)
# explainMe             = Equation('fNet == m * ((v**2) / r)',                                 physicsNamespace)
# someLeftoverLine      = Equation('f_smax == staticFriction * netForce',                      physicsNamespace)


#* Obvious ones for masterSolve
deltaV = Equation('dv == initv - finalv', physics, tags={'motion'})
deltaV = Equation('v == dv', physics, tags={'motion'})
circumference = Equation('C == 2*pi*r', fundamentalMath, tags={'geometry', 'circles'})
circleArea = Equation('A == pi*r**2', fundamentalMath, tags={'geometry', 'area', 'circles'})

#* Angular Kinematics
centripitalMotion = Equation('a == (v**2)/r', physics.replace(
    Unit('a', 'centripetal acceleration'),
    Unit('v', 'tangential (regular) velocity'),
    Unit('r', 'radius')
), 'The centripetal acceleration is always pointing towards the center', tags={'motion', 'angular motion'})
# angularSpeed = Equation('angs == dang/dt', physicsNamespace)
angularAccel = Equation('angA == dangs/dt', physics, tags={'motion', 'angular motion'})
angularVelocity = Equation('v == angV * r', physics, tags={'motion', 'angular motion'})
angularAcceleration = Equation('angA == Derivative(angV, t)', physics, tags={'motion', 'angular motion'})
kineticEnergyRolling = Equation('ker == (1/2) * I * angV**2 + (1/2) * m * v**2', physics.replace(Unit('m', 'the mass of the center of mass')),
    defaults={'v': Symbol('angV')*Symbol('r')},
    help='Remember that v is linear velocity, while angV is the angular velocity!',
    tags={'energy', 'kinetic energy', 'rolling', 'motion'}
)
rps2angV = Equation('angV == 2*pi*T', physics)
rps2linV = Equation('linV == 2*pi*r*T', physics)

#* Kinematics
motionEqu1 = Equation("v == initv - a * t", physics, tags={'motion'})
motionEqu2 = Equation("pos == initPos + initv * t - (1/2)*a*(t**2)", physics, tags={'motion'})
motionEqu3 = Equation("v == sqrt((initv**2) - 2 * a * (p-initp))", physics, tags={'motion'})
angV2linV = linV2angV = Equation("v == r*angV", physics, tags={'motion', 'angular motion'})
angA2linA = linA2angA = Equation('a == r*angA', physics)
ang2dist = dist2ang = Equation('theta == d/r', physics)

#* Energy
kineticEnergy = Equation("ke == (1/2) * m * v**2", physics, tags={'energy', 'kinetic energy'})
gravityPotentialEnergy = Equation("peg == m * g * h", physics, defaults={"g": 9.805}, tags={'energy', 'potential energy'})
# springPotentialEnergy = Equation('pes == (1/2) * k * (dx**2)', physicsNamespace)
loopPotentialEnergy = Equation("pec == m * g * 2 * r", physics, defaults={"g": 9.805}, tags={'energy', 'potential energy'})
totalEnergy = Equation("te == ke + pe", physics, tags={'energy'})
mechanicalEnergy = Equation("me == pe + ke", physics, tags={'energy'})

#* Work
# work == delta KE
work = Equation('W == dke', physics)
workEqu = Equation("W == f * mag * cos(theta)", physics, tags={'work'})
workSimple = Equation("W == f * d", physics, tags={'work'})
workEnergyTheorem = Equation("W == ((1/2) * m * vf**2) - ((1/2) * m * vi ** 2)", physics, help='Work is the net work', tags={'work', 'energy'})
workDoneByGravity = Equation("W == m * g * h", physics, defaults={"g": 9.805}, tags={'work'})
conservativeForcePotentialEnergy = Equation('pef - pei == -Integral(Fx, x)', physics, tags={'work'})
workAgainstNonConservativeForce = Equation('mei - wdaf == mef', physics, tags={'work', 'friction'})
workDoneByNonConservativeForce = Equation('W == f * d * cos(theta)', physics.replace(
    Unit('W', 'Work done by the force (i.e. friction)'),
    Unit('theta', 'The angle at which the force is pushing the object')
), tags={'work', 'friction'})
friction = Equation('f == mu_k * N', physics, tags={'friction'})

#* Springs
springForce = Equation("f == -k*x", physics, tags={'springs'})
springWork = Equation("W == (1/2) * k * x**2", physics, tags={'springs', 'work'})
springPotentialEnergy = Equation("pes == (1/2) * k * (dx**2)", physics, help='dx is in meters, k is in Newtons/meter', tags={'springs', 'energy', 'potential energy'})

#* Power
physicsPower = Equation('P == W / s', physics, tags={'power'})
physicsPower2 = Equation('P == f / v', physics, tags={'power'})

#* Momentum
momentum = Equation('p == m * v', physics, tags={'momentum'})
# newtonsLawD = Equation('f == m * Derivative(v, t)', physicsNamespace)
newtonsLawD = Equation('f == Derivative(m*v, t)', physics, tags={'momentum'})
newtonsLawMomentum = Equation('f == Derivative(p, t)', physics, tags={'momentum'})
conservationOfLinearMomentum = Equation('p1i + p2i == p1f + p2f', physics.replace(
    Unit('p1i', 'object 1\'s initial momentum'),
    Unit('p2i', 'object 2\'s initial momentum'),
    Unit('p1f', 'object 1\'s final momentum'),
    Unit('p2f', 'object 2\'s final momentum'),
), help='Only applies if theres only 2 objects, acting soley from forces caused by each other',
    tags={'momentum'}
)
momentum2energy = energy2momentum = Equation('ke == (p**2)/(2*m)', physics, tags={'momentum', 'energy'})

#* Impulse
# Impulse equals average force over time
avgForceOverTime = Equation('imp == f_avg', physics, tags={'impulse'})
impulse = Equation('imp == Integral(f, t)', physics, tags={'impulse'})
impulseMomentumTheorem = Equation('imp == p_f - p_i', physics, tags={'impulse', 'momentum'})

frequency = Equation('T == 1/f', physics, tags={'frequency', 'waves'})

#* Inertia
rotationalKineticEnergy = Equation('ker == (1/2) * I * wThing**2', physics, tags={'energy', 'kinetic', 'angular motion'})
momentOfInertia = Equation('I == Sum(m_i * r_i**2, (i, 1, n))', physics.replace(
    Unit('m_i', 'The mass of a concentric inscribed disk'),
    Unit('r_i', 'The radius of a concentric inscribed disk'),
), tags={'motion', 'inertia'})
momentOfInertiaHoop = Equation('I == m * r**2', physics, tags={'inertia'})
momentOfInertiaCylinder = momentOfInertiaDisk = Equation('I == (1/2) * m * r**2', physics, tags={'inertia'})
momentOfInertiaSphere = Equation('I == (2/5) * m * r**2', physics, tags={'inertia'})
momentOfInertiaPoint = Equation('I == m * r**2', physics, tags={'inertia'})
inertiaEnergyThing = Equation('m*g*h == (1/2) * m * v**2 + (1/2) * I * angV**2', physics, tags={'inertia', 'energy'})
# velocityInertiaEnergyThing = Equation('v == ((2*g*h) / (1 + (I / (m*r**2))))**(1/2)', physics, tags={'inertia', 'energy'}, defaults={"g": 9.805})
velocityInertiaEnergyThing = Equation('v == sqrt(2)*sqrt(h*m*r**2*g/(m*r**2 + I))', physics, tags={'inertia', 'energy'}, defaults={"g": 9.805})
kineticEnergyFlywheel = Equation('kew == (1/2) * I * angv', physics)


#* Torque
torque3 = Equation('tau == f*r*sin(theta)', physics, tags={'torque'})
torque = Equation('tau == cross(f, r)', physics, tags={'torque'})
torque2 = Equation('tau == I*angA', physics, tags={'torque'})
# Just a derivation of the above 2 equations:
inertia2Force = Equation('I*angA == f*r', physics)
# torque = forceAt90DegreeAngle

# ! net torque in a non-moving system is 0
# radius (and all positions) can be vectors


# Positive torque = counterclockwise
# Negative torque = clockwise

#* Angular Momentum
angularMomentum = Equation('L == cross(r, p)', physics)
angularImpulseMomentumTheorem = Equation('T == Derivative(L, t)', physics, tags={'impulse', 'momentum'})
angularMomentumInertia = Equation('L_xaxis == I * angS', physics)
conservationOfAngularMomentum = Equation('L_i == L_f', physics)



# SUM SYNTAX
# Sum(val, (i, start, end))
# Sum(val, (i, low, high))

# INTEGRAL SYNTAX
# Integral(val, var)
# Integral(val, (var, low, high))

############################## Electronics ##########################

electronics = Namespace('electronics',
    Unit('v',        'voltage', 'volts').generateVariant(initial, final, time, primary, secondary, throughResistor, throughCapacitor, throughInductor, inMod, outMod),
    Unit('i',        'current', 'amps').generateVariant(initial, final, time, primary, secondary, throughResistor, throughCapacitor, throughInductor, inMod, outMod),
    Unit('r',        'resistance', 'ohms').generateVariant(time, primary, secondary, equivalent, throughCapacitor, throughInductor),
    Unit('G',        'conductance', 'Seimanns').generateVariant(time, primary, secondary, equivalent),
    Unit('Gn',       'conductance of a given resistor'),
    Unit('iGn',      'current through a given resistor'),
    Unit('vL',       'induced voltage', 'volts'),
    Unit('Av',       'Voltage Gain', 'volts'),
    Unit('C',        'capacitance', 'farads').generateVariant(initial, final, time, primary, secondary, throughCapacitor),
    Unit('q',        'charge').generateVariant(initial, final, time, primary, secondary, throughCapacitor),
    Unit('d',        'distance between the capacitor plates'),
    Unit('mu',       'permeability'),
    Unit('εr',       'relative permittivity for the medium'),
    Unit('ε0',       'permittivity of space (constant, equal to 8.854x10-12 F/m)'),
    Unit('w',        'energy', 'joules'),
    Unit('phi',      'magnetic flux', 'teslas (Webers / meter)'),
    Unit('t',        'time', 'seconds'),
    Unit('L',        'inductance', 'Henrys').generateVariant(initial, final, time, primary, secondary, throughInductor),
    Unit('N',        'number of turns of the wire'),
    Unit('W',        'Work', 'Joules'),
    Unit('tc',       'time constant'),
    Unit('vL_t',     'induced voltage at time t -- vL(t)'),
    Unit('v_t',      'voltage at time t -- v(t)'),
    Unit('i_t',      'current at time t -- i(t)'),
    Unit('f',        'frequency', 'Hertz'),
    Unit('ph',       'phase shift'),
    Unit('ohmega',   'angular frequency', 'radians/second', {'wThing'}),
    Unit('T',        'Period', 'revolutions/second'),
    Unit('j',        'i (sqrt(-1))'),
    Unit('phaseAng', 'the phase angle', 'radians'),
    Unit('Z',        'impedance').generateVariant(time, initial, final, primary, secondary, throughCapacitor, throughInductor, phasor),
    Unit('X',        'reactance').generateVariant(time, initial, final, primary, secondary, throughCapacitor, throughInductor, phasor),
    Unit('Y',        'admittance', 'Seimenns').generateVariant(time, initial, final, primary, secondary, throughCapacitor, throughInductor, phasor),
    Unit('B',        'suceptance', '1/reactance').generateVariant(time, initial, final, primary, secondary, throughCapacitor, throughInductor, phasor),
    # Unit('G',        'conductance').generateVariant(time, initial, final, primary, secondary, throughCapacitor, throughInductor, phasor),


    # 'Rf': 'the resistor connecting the op amp Vout and the op amp negative in terminals',
    # 'Rin': 'the resistor between Vin and the op amp negative in',
    # 'A': 'surface area of each capacitor plate',
)

ohmsLaw = Equation("v == i*r", electronics)
conductanceSolver = Equation("iGn == (i*Gn)/Geq",{
    'Geq': 'equivalent conductance',
    'Gn': 'conductance of the chosen resistor',
    'i': 'current into resistor',
    'iGn': 'current through the chosen resistor',
})

#* Op Amps:
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

#* Capacitors:
capacitorEnergy = Equation('W == (1/2) * C * v**2', electronics)
capacitorCurrent = Equation('i == C * Derivative(v, t)', electronics)
capacitorVoltage = Equation('v == (1/C) * Integral(i(t2), (t2, t_0, 1)) + v(t_0)', electronics)
capacitorCharge = Equation('q == C * v', electronics)
plateCapaitor = Equation('C == er*e0 * A/d', electronics.replace(Unit('A', 'surface area of each capacitor plate')),  defaults={'e0': 8.854*10**(-12)})

#* Inductors:
# the emf induced in an electric circuit is proportional to the rate of change of the mangetic flux linking the circuit
inductorEnergy = Equation('W == (1/2) * L * i**2', electronics.replace(Unit('i', 'final current in the inductor')))
inductorCurrent = Equation('i == (1/L) * Integral(v(t2), (t2, t_0, 1)) + ii', electronics)
inductorVoltage = Equation('v == L * Derivative(i, t)', electronics)
faradayslaw = Equation('vL_t == Derivative(phi, t)', electronics)
# Changing currents in coils induce coil voltages opposite current changes
lenzsLaw = Equation('L == vL_t / Derivative(i, t)', electronics)

solenoid = Equation('L == (N**2 * mu * A) / l', electronics.replace(
    Unit('A', 'cross sectional area of the solenoid', 'meters^2'),
    Unit('mu', 'permeability of the core material', 'Henrys/meter'),
    Unit('l', 'length of the solenoid', 'meters')
))

transformerVoltage = Equation('Vs / Vp == Ns / Np', electronics)
transformerCurrent = Equation('Is / Ip == Np / Ns', electronics)

#* Source-Free circuits
# RC == resistor-capacitor circuit
# RL == resistor-inductor circuit
# t == tc * ln(v0/v(t))
# v_c(t) = V_o*E**(-1/tc) -- discharge equation for a source free rc circuit
RCDischargeVoltage = Equation('v_t == vi * _e ** (-t/tc)', electronics)
RCTimeConstant = Equation('tc == Req * C', electronics)
RCChargeVoltage = Equation('v_t == vs + (vi - vs) * _e**(-t/tc)', electronics)
RCChargeCurrent = Equation('i_t == (vs/r) * _e**(-t/tc)', electronics)
someImportantRCEquation = Equation('x_t == x_f+ (x_i - x_f)*_e**(-t/tc)', physics)

# i_L(t) == I_0*e*(-t/tc)
RLDischargeCurrent = Equation('i_t == ii * _e**(-t/tc)', electronics)
RLTimeConstant = Equation('tc == L/Req', electronics)
# RLChargeCurrent = Equation('i_t == (vs/r) + ((ii - (vs/r)) * _e**(-t/tc))', electronicsNamespace)
RLChargeCurrent = Equation('v_t == vs * _e**(-t/tc)', electronics)


# inductorCurrent = iL(t) == (1/L) * Integral(high=t, low=0, vL(dummy)ddummy) + i_0
#? Vp = -Np * Derivative(phi, t)
# inductorEnergy = w=integral(high=t, low=t_0, p(t))
#? capacitorVoltage ( vc(t) = 1/C * integrral(t, 0, ???))
#? w = L*integral(i(t), low=0, ???)
# watts == joules / second
# Coulomb = amps / second

#* AC Current
# ACPowerLoss = Equation*('P_loss == i**2 * R_loss', electronicsNamespace)
ACSine = Equation('v_t == V_m * sin(2*pi*f*t + ph)', electronics)
frequency = Equation('T == 1/f', electronics)

angFrequency = Equation('wThing = 2*pi*f', electronics)

rect2polarMag = Equation('mag == sqrt(a**2 + b**2)', electronics)
rect2polarAng = Equation('theta == atan(b/a)', electronics)

#* Phasors
EulersFormula = Equation('_e**(_i*theta) == cos(theta) + _i*sin(theta)', electronics)
# ohmega == wthing == angularsomething radians/second
EulersFormula = Equation('_e**(_i*ohmega*theta) == cos(ohmega*theta) + _i*sin(ohmega*theta)', electronics)

# somePhasorThing = Equation('v(t) == (r*_e**(_i * phaseAngle) * _e**(_i*ohmega*t))', electronics)

# phasor = Equation('phasor == r*_e**(j*phase)', electronics)
timeDerivativeOfPhasor = Equation('Derivative(cos(2*pi*f*t), t) == (-2*pi*f)*sin(2*pi*f*t)', electronics)
capacitorCurrent = Equation('i_c_t == C*Derivative(v_c_t, t)*i_c_t', electronics)
inductorVoltage  = Equation('v_L_t == L*Derivative(i_L_t, t)*v_L_t', electronics)


#* Impedance
# reactance is the resistance to voltage flow
# resistance is the restistance to current flow
# Or the other way around?

# For inductors
inductorImpedance = Equation('phZ_L == phV_L/phI_L', electronics)
inductorImpedance2 = Equation('phZ_L == _i*ohmega*L', electronics)
inductorImpedance3 = Equation('phZ_L == _i*X_L', electronics)
seriesRLImpedance = Equation('Z_eq_RL = R+_i*X_L', electronics)
inductorReactance = Equation('X_L == ohmega*L', electronics)

#? X_L=delta(ohmega*L)

# For capacitors
capacitorImpedance = Equation('phZ_C == phV_C/phi_C', electronics)
capacitorImpedance2 = Equation('phZ_C == 1/_i*ohmega*C', electronics)
capacitorImpedance3 = Equation('phZ_C == -_i*X_C', electronics)
seriesRCImpedance = Equation('Z_eq_RC = R-_i*X_C', electronics)
capacitorReactance = Equation('X_C == 1/(ohmega*C)', electronics)
capacitorReactance2 = Equation('X_C == V_m/I_m', electronics)
# Equation('X_C == 1/(2*pi*f*C)', electronics)(f=5*k, )

resistorImpedance = Equation('phZ == r', electronics)
inductorImpedance = Equation('phZ == _i*X', electronics)
inductorImpedance2 = Equation('phZ == _i*ohmega*L', electronics)
capacitorImpedance = Equation('phZ == -_i*X', electronics)
capacitorImpedance2 = Equation('phZ == 1/(_i*ohmega*C)', electronics)

ohmsLawImpedance = Equation('phV == phZ*phI', electronics)



#? X_C == delta(1/(ohmega*C))

impedance = Equation('phZ == phv/phi', electronics)
# impedance = Equation('phZ == r(ohmega) + _i*X(ohmega)', electronics)
admittance = Equation('phY == 1/phZ', electronics)
# impedance = Equation('phY == G(ohmega) + _i*B(ohmega)', electronics)
reactance = Equation('reactance == im(Z)', electronics)

# impedance of a purely resistive AC circuit is R<_0
# Z of an idea capacitor == -j*(1/2*pi*f*c)
# Z of an idea inductor == j*(2*pi*f*L)

# Capcitor current alwasy leasts capacitor voltage by 90 degress

#* AC Power
instantPower = Equation('p_t == v_t * i_t', electronics)
instantPowerSine = Equation('p_t == (1/2) * v_m*i_m * cos(theta_v - theta_i) + (1/2)*v_m*i_m * cos(2*ohmega*t + theta_v + theta_i)', electronics)
Equation('v_t == v_m * cos(ohmega*t + theta_v', electronics)
Equation('i_t == i_m * cos(ohmega*t + theta_i', electronics)

# The average power, in watts, is the average of the instantaneous power over one period.
averagePower = Equation('p == (1/T) * Integral(p_t, (t, 0, T))')
# Integral(val, (var, low, high))
averagePowerExpanded = Equation('p == (1/T) * Integral((1/2) * v_m*i_m * cos(theta_v - theta_i), (t, 0, T)) + (1/T) * Integral((1/2)*v_m*i_m * cos(2*ohmega*t + theta_v + theta_i), (t, 0, T))', electronics)
averagePower2 = Equation('p == (1/2) v_m * i_m * cos(theta_v - theta_i)', electronics)

pureResistivePower = Equation('p == (1/2) * i_m**2 * r', electronics)
pureReactivePower = Equation('p == 0', electronics)

################################### Waves ##################################

waves = Namespace(
    Unit('y_t', 'the amplitude (y value) of the wave at time t'),
    Unit('t', 'time'),
    Unit('A', 'amplitude'),
    Unit('k', 'the wavenumber (spatial frequency)'),
    Unit('x', 'the x offset'),
    Unit('y', 'the y offset'),
    Unit('phi', 'phase angle'),
    Unit('k', 'wavenumber (spatial frequency)'),
    Unit('angF', 'angular frequency', psuedonyms={'ohmega', 'angf'}),
    Unit('f', 'frequency'),
    Unit('T', 'period'),
)

complexWaveform = Equation('y_t == A * cos(k * x - ohmega * t + phi) + y', waves,
    help='This may be incorrect, I think I might have messed it up')

sinusoid = Equation('y_t == A * sin(ohmega * t + phi) + y', waves, defaults={'phi': 0, 'y': 0})
cosinusiod = Equation('y_t == A * sin(ohmega * t + phi) + y', waves, defaults={'phi': 0, 'y': 0})

angularFrequency2frequency = frequency2angularFrequency = Equation('angF == 2*pi*f', waves)
period2angFrequency = angularFrequency2period = Equation('T == (2*pi) / angF', waves)
frequency2period = period2frequency = Equation('f == 1/period', waves)

# angFrequency = 2*pi*frequency
# period = (2*pi) / angFrequency
# frequency = 1/period

# impedance of an inductor approches oo as frequency approaches oo
# impedance of a capacitor approches 0 as frequency approaches oo

# e**jwt == ohmega*t
# 'r*_e**(_i*ohmega*t) == ohmega*t'

# magnitude for cos(theta) + j*sin(tehta) == sqrt(cos(theta)**2 + sin(theta)**2)
# mag of cos(theta) + j*sin(theta) ALWAYS == 1
# angle of cos(theta) + j *sin(theta) ALWAYS == theta

# for v(t)

# differentiation in the time domain == multiplying by j*ohmega in the phasor domain
# integration in the time domain == dividing by j*ohmega in the phasor domain



def parallel(*resistors):
    bottom = 0
    for r in resistors:
        bottom += 1/r
    return 1/bottom


def series(*resistors):
    return sum(resistors)

# q = C*V_v
# dq/dt = C * dv_c/dt
# i_c(t) = C * d*v_c/dt
# q=Cv_

# series connected capacitors all have the same seperated charge
# 2 capacitors in series --
# The imaginary part of impedance is reactance

# # for constant dc current, a capaciter behaves like an open circuit -- and no current passes through

# powerDelivered to a capacitor = v*i = v * (C*dv/dt)

# power is the energy rate, p = dw/dt
# energy = w = integral(high=t, low=t_0, p(0) * something he changed the slide)

ll=parallel
s=series

parallelCapacitors = llCap = series
seriesCapacitors = sCap = parallel

parallelInductors = llInd = parallel
seriesInductors = sInd = series

seriesImpedances = series
parallelImpedances = series

seriesAdmittances = parallel
parallelAdmittances = series


#* Misc:
# coulombsLaw = Equation()
# magneticFlux =

listOfEquations = [i for i in globals().values() if type(i) is Equation]

# I don't understand how this error is possible. This should work.
# globalsItems = globals().items()
# listOfEquations = []
# dictOfEquations = {}
# for name, eq in globalsItems:
#     if type(eq) is Equation:
#         listOfEquations.append(eq)
#         dictOfEquations[eq] = name


def dictOfEquations():
    d = {}
    for name, eq in globals().items():
        if type(eq) is Equation:
            d[eq] = name
    return d


def searchEquations(*whatWeHave, loose=True, searchNames=False, names=True, tags=True, namespace=physics):
    rtn = []
    # Just check if what they've entered is valid
    for i in whatWeHave:
        if i not in namespace and not loose and not searchNames and not tags:
            warn(f"{i} not a valid variable name")

    for name, i in globals().items():
        if type(i) is Equation:
            # Check if the equation applies (by comparing variable names)
            if i.applicable(*whatWeHave, loose=loose, tags=tags) and i.namespace.name in ('na', namespace.name):
                rtn.append(name if names else i)
            elif searchNames:
                # Check if anything we have is in the name of the instance itself
                for n in whatWeHave:
                    if n in name:
                        rtn.append(name if names else i)

    return rtn




#* The NEW MasterSolve
# Not sure how this will react if given a default parameter
def masterSolve(*args, namespace=physics, recurse=True, **kwargs):
    global listOfEquations
    derived = {}
    # loop through all the equations we can actually use
    for eq in filter(lambda x: x.applicable(*(kwargs.keys())), listOfEquations):
        # Only use the equations relavant to the variables passed in
        if eq.namespace == namespace:
            solvingFor = ensureNotIterable(set(eq.atomNames).difference(kwargs.keys()))
            # We can assume the defaults as well
            # solvingFor = ensureNotIterable(solvingFor.difference(eq.defaults.keys()))

            # assert(not isiterable(solvingFor)), f'Equation.applicable isnt working properly, gave {solvingFor}'

            # params = set(eq.atomNames).difference((solvingFor,))
            params = set(eq.atomNames).difference(ensureIterable(solvingFor))

            try:
                derived[solvingFor] = eq(**kwargs, allowNonsenseParams=True, raiseErrors=True)
            except:
                with coloredOutput(Colors.ERROR):
                    print(f'Failed to use {dictOfEquations()[eq]+",":23} {eq.raw:47} to get {solvingFor} from {params}')
            else:
                # print(f'Using {dictOfEquations()[eq]+",":31} {eq.raw:47} to get {solvingFor:6} from {params}')
                paramStr = ''
                for var in params:
                    val = kwargs[var]
                    paramStr += f"{var}={val}, "
                call = f"{dictOfEquations()[eq]}({paramStr[:-2]})"
                print(f"Using {call:45} to get {solvingFor:6} = {str(derived[solvingFor]):20} from {eq.raw}")


    # No point in recursing if we haven't found anything new
    if len(derived) and recurse:
        derived = addDicts(derived, masterSolve(**addDicts(kwargs, derived)))

    return derived





#* Units
# Milli
m=1/1000
# Mega
M=1000000
# Kilo
k=1000
# Micro
mu=1/1000000
# Nano
n=1/1000000000
# Default is kilograms
gram=1/1000
# Default is seconds
h=1/3600
# Defualt is meters
cm=1/100
mm=1/1000
km=1000
# Defualt is seconds
ms=1/1000
g=9.80665
# Default is revolutions/second
rpm=60
minute=1/60
# todo('make Equation and the search functions accept any psuedonym of their units as valid parameters')