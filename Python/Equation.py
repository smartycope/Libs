from sympy import *
from Cope import *
from clipboard import copy
import EasyRegex as er
import re
from warnings import warn
from copy import deepcopy, copy as objcopy
from include import unknown, known
from equationModifiers import delta, initial, final


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


def parallel(*resistors):
    bottom = 0
    for r in resistors:
        bottom += 1/r
    return 1/bottom


def series(*resistors):
    return sum(resistors)


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


def listOfEquations(globals):
    return [i for i in globals.values() if type(i) is Equation]

# I don't understand how this error is possible. This should work.
# globalsItems = globals().items()
# listOfEquations() = []
# dictOfEquations = {}
# for name, eq in globalsItems:
#     if type(eq) is Equation:
#         listOfEquations().append(eq)
#         dictOfEquations[eq] = name


def dictOfEquations(globals):
    d = {}
    for name, eq in globals.items():
        if type(eq) is Equation:
            d[eq] = name
    return d


def searchEquations(globals, *whatWeHave, namespace, loose=True, searchNames=False, names=True, tags=True):
    rtn = []
    # Just check if what they've entered is valid
    for i in whatWeHave:
        if i not in namespace and not loose and not searchNames and not tags:
            warn(f"{i} not a valid variable name")

    for name, i in globals.items():
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
def masterSolve(globals, *args, namespace, recurse=True, **kwargs):
    derived = {}
    # loop through all the equations we can actually use
    for eq in filter(lambda x: x.applicable(*(kwargs.keys())), listOfEquations(globals)):
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
