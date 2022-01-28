#!/usr/bin/env python3
""" Cope.py
A bunch of generic functions and classes useful in multiple projects
"""
__version__ = '3.2.0'
__author__ = 'Copeland Carter'
__email__ = 'smartycope@gmail.com'
__license__ = 'GPL 3.0'
__copyright__ = '(c) 2021, Copeland Carter'

################################### Suggested Features ###################################
#! make confidence print the confidence level -- needs testing
#! make that self-modifying function, both one that removes itself and one that adds a line above it -- needs more testing
#! KeyShortcut could use more testing
#? I think @debug and @todo are still failing periodically -- needs debugging
#// Add a function that takes a parameter and a bunch of values and just checks to see if the parameter is one of those values, otherwise throws a type error with a preset message -- Done?
#// get the key class that I wrote on Rebecca
#todo change _debugBeingUsedAsDecorator() to use an enum of decorator types, and also make it generally useable
#todo also, couldn't it automatically get the function name from the metadata?
#todo write an alias decorator that sets the thing it's decorating to a provided alias and injects it into the global namespace
# add a filter parameter that lets you specify a function that must return true -- dont remember what this means
#todo if todo is called with no parameters, make it say "{functionName} needs implemented!" or something
#todo make log, and allow it to write to a file as well as command line
#todo Uhh... how do I not have a logging function??? Add a logging function, a global debug level, possibly a debug level enum, and incorperate verbose
#todo make verbose and debug level more globally implemented
#todo A title case function that doesn't ignore exisiting case
#todo make an intuituve case funciton that doens't capitalize little words like 'and' and 'of' and such
#todo Add string color names to parseColor
#? The manual debug regex to parse a line doesn't really work (is it even possible to do that with regex?)
#? ROOT doesn't work (I suspect DIR doesn't either)
#todo Add a bunch of Errors and warnings you can raise (like depricationWarning/error, and some others)
#todo maybe write a warn function, and probably just have it call log with parameters
#? Shift (JUST FREAKING SHIFT) doesn't work with KeyShortcut. No clue why. Debug eventually.
#? Add the + operator to keys so you can add keys to make a KeyShortcut (figure out prototyping)
#todo Make debug show generators (map, filter, range, etc) like lists and other iterables
#todo checkImport vs ensureImport
#todo add a priority parameter to todo (maybe change the color?)
#? My color function (whatever is coloring debug and todo) still colors errors after it. possibly in a different thread.
#? remove duplicates sucks (still returning a generator)
#? debug doesn't display lists properly when using repr
#todo Make debug print the value entirely on the next line if it wont all fit on this line
#? debug's shortened paths don't work anymore because ROOT doesn't work anymore
#? Make debug's nice iterable printer recursively call itself (so a list of lists still prints nice on all levels)
#todo Make ensureImport/checkImport pass whether it's imported or not to the @'ed fucntion as a named keyword arguement
#todo Add more parameters to depricated
#todo Make get metadata more globally useful
#todo change all "calls" parameters to "additionalCalls", cause that makes more sense
#todo add ifUsedAsDecorator to warn
#? in _debugGetListStr, somehow round any float to a given length, including those printed in iterables
#? Finish writing KeyChord and such
#? Todo as a decorator is called in the global scope instead of when the function is called
#todo add a log level parameter to debug
#todo make tested/untested their own functions (similar, but not the same as confidence)
#? importpath works, but it could use some fine tuning
#todo tested and untested don't give the right metadata (cause they're called somewhere else)


################################### Imports ###################################
from atexit import register as registerExit
from re import search as research
from re import match as rematch
from math import floor, ceil, tau, pi as PI
from ctypes import pointer, py_object
from inspect import stack
from os.path import basename, dirname, join
from random import randint
from time import process_time
from typing import Any, Callable, Iterable, Optional, Union
from enum import Enum, auto
from copy import deepcopy
from os import get_terminal_size
from collections import namedtuple
from pathlib import Path
import importlib
from importlib import util as importutil
import sys

################################### Constants ###################################
ENABLE_TESTING = True

# This is because I write a lot of C/C++ code
true, false = True, False

_debugCount = 0

# Override the debug parameters and display the file/function for each debug call
#   (useful for finding debug calls you left laying around and forgot about)
DISPLAY_FILE = False
DISPLAY_PATH = False
DISPLAY_FUNC = False
DISPLAY_LINK = False
HIDE_TODO    = False
# FORCE_TODO_LINK = False

#* Convenience commonly used paths. ROOT can be set by the setRoot() or markRoot() functions
DIR  = dirname(__file__)
ROOT = dirname(DIR) if basename(DIR) in ('src', 'source') else DIR
HOME = str(Path.home())

# Yes, this is not strictly accurate.
MAX_INT_SIZE = 2147483645

VERBOSE = True
DEBUG_LEVEL = LOG_LEVEL = 0

# A unique dummy class for parameters
class _None: pass


################################### Setters for Globals ###################################
def displayAllFiles(to=True):
    global DISPLAY_FILE
    DISPLAY_FILE = to

def displayAllPaths(to=True):
    global DISPLAY_PATH
    DISPLAY_PATH = to

def displayAllFuncs(to=True):
    global DISPLAY_FUNC
    DISPLAY_FUNC = to

def displayAllLinks(to=True):
    global DISPLAY_LINK
    DISPLAY_LINK = to

def hideAllTodos(to=True):
    global HIDE_TODO
    HIDE_TODO = to

def setRoot(path):
    global ROOT
    ROOT = path

def markRoot(relative='.'):
    global ROOT
    ROOT = join(dirname(stack()[1].filename), relative)
    print(ROOT)

def setVerbose(to=True):
    global VERBOSE
    VERBOSE = to

def verbose():
    global VERBOSE
    return VERBOSE

def setLogLevel(to):
    global LOG_LEVEL
    LOG_LEVEL = to


################################### Enums ###################################
class CommonResponses:
    """ A collection of default responses for inputs. Make sure to use .lower() when testing agaisnt these.
        Note: There is some overlap between them, so testing order matters.
    """
    YES   = ('y', 'yes', 'ya', 'yeah', 'si', 'true', 'definitely', 'accurate', 'totally')
    NO    = ('n', 'no', 'not', 'nien', 'false', 'nope', 'not really', 'nah')
    MAYBE = ('sure', 'kinda', 'i guess', 'kind of', 'maybe', 'ish', 'sorta')
    NA    = ('none', 'na', 'n/a', 'not applicable')
    HIGH_AMOUNT = ('very', 'much', 'very much', 'extremely', 'quite', 'quite a bit', 'lot', 'a lot', 'lots',
                   'super', 'high', 'ton', 'a ton', 'bunch', 'a bunch')
    MODERATE_AMOUNT = ('fairly', 'somewhat', 'enough')
    SOME_AMOUNT = ('a little bit', 'a bit', 'a little', 'ish', 'not a lot', 'not a ton', 'some', 'mostly')
    LOW_AMOUNT  = ("not at all", 'not very', 'not much', 'low', 'none', 'none at all', 'not terribly')

class LogLevel(Enum):
    NONE = 0
    LOGGING = 1
    WARNINGS = 2
    ERRORS = 3

class Colors:
    # Default color constants
    DEFAULT = (204, 204, 204)
    ALERT   = (220, 0, 0)
    WARN    = (150, 30, 30)
    ERROR   = ALERT

    # A set of distinct characters for debugging
    _colors = [(43, 142, 213), (19, 178, 118), (163, 61, 148), (255, 170, 0), (255, 170, 255), (170, 0, 255)]

    # Default colors for debugging -- None for using the previously set color
    NOTE_CALL          = (211, 130, 0)
    EMPTY              = NOTE_CALL
    CONTEXT            = None
    COUNT              = (34, 111, 157)
    DEFAULT_DEBUG      = (34, 179, 99)
    TODO               = (128, 64, 64)
    STACK_TRACE        = (159, 148, 211)
    CONFIDENCE_WARNING = (255, 190, 70)
    DEPRICATED_WARNING = WARN
    LOG_COLOR          = (100, 130, 140)

    DEBUG_EQUALS          = DEFAULT
    DEBUG_METADATA_DARKEN = 70
    DEBUG_TYPE_DARKEN     = 10
    DEBUG_NAME_DARKEN     = -60
    DEBUG_VALUE_DARKEN    = 0


################################### Color Utilites ###################################
def resetColor():
    print('\033[0m',  end='')
    print('\033[39m', end='')
    print('\033[49m', end='')
    # print('', end='')

#todo Add support for openGL colors (-1.0 to 1.0
#todo Consider changing the param paradigm to *rgba instead of all seperate parameters
def parseColorParams(r, g=None, b=None, a=None, bg=False) -> "((r, g, b, a), background)":
    """ Parses given color parameters and returns a tuple of equalized
        3-4 item tuple of color data, and a bool for background.
        Can take 3-4 tuple/list of color data, or r, g, and b as induvidual parameters,
        and a single int (0-5) representing a preset unique color id.
        a and bg are always available as optional or positional parameters.

        Note: Seperate color specifications for foreground and background are not currently
        supported. bg is just a bool.
    """
    #* We've been given the name of a color
    if type(r) is str:
        raise NotImplementedError(f"parseColorParams does not yet support named colors")
    #* We've been given a list of values
    elif type(r) in (tuple, list):
        if len(r) not in (3, 4):
            raise SyntaxError(f'Incorrect number ({len(r)}) of color parameters given')
        else:
            return (tuple(r), (False if g is None else g) if not bg else bg)

    #* We've been given a single basic value
    elif type(r) is int and b is None:
        return (Colors._colors[r] + ((a,) if a is not None else ()), (False if g is None else g) if not bg else bg)

    #* We've been given 3 seperate parameters
    elif type(r) is int and g is not None and b is not None:
        if type(a) is int:
            return ((r, g, b, a), bg)
        elif type(a) is bool or a is None:
            return ((r, g, b), bool(a) if not bg else bg)

    #* We've been given None
    elif r is None:
        return (Colors.DEFAULT, bg)

    #* We're not sure how to interpret the parameters given
    else:
        raise SyntaxError(f'Incorrect color parameters {tuple(type(i) for i in (r, g, b, a, bg))} given')

class coloredOutput:
    """ A class to be used with the 'with' command to print colors.
        Resets after it's done.
        @Parameters:
            Takes either a 3 or 4 list/tuple of color arguements, 3 seperate
            color arguements, or 1 color id between 0-5 representing a distinct
            color. Set the curColor parameter (must be a 3 or 4 item list/tuple)
            to have the terminal reset to that color instead of white.
        https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
    """
    def __init__(self, r, g=None, b=None, foreground=True, curColor=Colors.DEFAULT):
        color, bg = parseColorParams(r, g, b, bg=foreground)
        self.fg = bg
        self.r, self.g, self.b = color
        self.doneColor = curColor

    def __enter__(self):
        try:
            if self.fg:
                print(f'\033[38;2;{self.r};{self.g};{self.b}m', end='')
            else:
                print(f'\033[48;2;{self.r};{self.g};{self.b}m', end='')
        except:
            self.reset()

    def __exit__(self, *args):
        self.reset()

    def reset(self):
        print(f'\033[38;2;{self.doneColor[0]};{self.doneColor[1]};{self.doneColor[2]}m', end='')

def rgbToHex(rgb):
    """ Translates an rgb tuple of int to a tkinter friendly color code """
    return f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'

def darken(amount, r, g=None, b=None, a=None):
    """ Returns the given color, but darkened. Make amount negative to lighten """
    # Constrain isn't defined yet and I don't feel like moving it
    return tuple([min(255, max(0, i - amount)) for i in parseColorParams(r, g, b, a)[0]])

def lighten(amount, r, g=None, b=None, a=None):
    """ Returns the given color, but darkened. Make amount negative to darken """
    return tuple([constrain(i + amount, 0, 255) for i in parseColorParams(r, g, b, a)[0]])

def clampColor(r, g=None, b=None, a=None):
    """ Clamp a 0-255 color to a float between 1 and 0.
        Helpful for openGL commands.
    """
    rgba = parseColorParams(r, g, b, a)[0]
    return tuple(c / 255 for c in rgba)

def invertColor(r, g=None, b=None, a=None):
    """ Inverts a color """
    rgba = parseColorParams(r, g, b, a)[0]
    return tuple(255 - c for c in rgba[0])


################################### Import Utilities ###################################
def ensureImported(package:str, specificModules=[], _as=None,
                fatal:bool=False, printWarning:Union[str, bool]=True,
                _globals=globals(), _locals=locals(), level=0
                ) -> "(Union[package, (packages,)], worked)":
    if type(specificModules) is str:
        specificModules = [specificModules]
    try:
        _temp = __import__(package, _globals, _locals, specificModules, level)
    except ImportError:
        if type(printWarning) is str:
            print(printWarning)
        elif printWarning:
            if len(specificModules):
                print(f'Can\'t import {tuple(specificModules)} from {package}. Have you installed the associated pip package? (try "pip install {pacakge}")')
            else:
                print(f'Can\'t import {package}. Have you installed the associated pip package? (try "pip install {pacakge}")')
        if fatal:
            raise ImportError(package)
        return False
    else:
        if len(specificModules):
            for i in specificModules[:-1]:
                globals()[i] = _temp.__getattribute__(i)
            globals()[_as if _as else specificModules[-1]] = _temp.__getattribute__(specificModules[-1])
        else:
            globals()[_as if _as else package] = _temp
        return True

# todo
def checkImport(package:str, specificModules=[], _as=None,
                fatal:bool=False, printWarning:Union[str, bool]=True,
                _globals=globals(), _locals=locals(), level=0
                ) -> "(Union[package, (packages,)], worked)":
    # todo()
    return
    if type(specificModules) is str:
        specificModules = [specificModules]
    try:
        _temp = __import__(package, _globals, _locals, specificModules, level)
    except ImportError:
        if type(printWarning) is str:
            print(printWarning)
        elif printWarning:
            if len(specificModules):
                print(f'Can\'t import {tuple(specificModules)} from {package}. Have you installed the associated pip package? (try "pip install {pacakge}")')
            else:
                print(f'Can\'t import {package}. Have you installed the associated pip package? (try "pip install {pacakge}")')
        if fatal:
            raise ImportError(package)
        return False
    else:
        if len(specificModules):
            for i in specificModules[:-1]:
                globals()[i] = _temp.__getattribute__(i)
            globals()[_as if _as else specificModules[-1]] = _temp.__getattribute__(specificModules[-1])
        else:
            globals()[_as if _as else package] = _temp
        return True

def dependsOnPackage(package:str, specificModules=[], _as=None,
                fatal:bool=True, printWarning:Union[str, bool]=True,
                _globals=globals(), _locals=locals(), level=0):
    def wrap(func):
        def innerWrap(*funcArgs, **funcKwArgs):
            if ensureImported(package, specificModules, _as, fatal,
                           printWarning, globals, locals, level):
                return func(*funcArgs, **funcKwArgs)
            else:
                return None
        return innerWrap
    return wrap

def importpath(path, name, moduleName):
    spec = importutil.spec_from_file_location(name, path)
    module = importutil.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # This is kinda ugly and nasty, but it works. For now.
    globals()[moduleName] = importlib.import_module(name, moduleName).__getattribute__(moduleName)


################################### Debug ###################################
varnameImported = ensureImported('varname', ('ImproperUseError', 'VarnameRetrievingError', 'argname', 'nameof'), fatal=False)
DEBUGGING_DEBUG = False
_repr = repr

def _debugGetMetaData(calls=1):
    """ Gets the meta data of the line you're calling this function from.
        Calls is for how many function calls to look back from.
    """
    try:
        s = stack()[calls]
        return s
    except IndexError:
        return None

def _debugGetLink(calls=0, full=False, customMetaData=None):
    if customMetaData is not None:
        d = customMetaData
    else:
        d = _getMetaData(calls+2)

    _printLink(d.filename, d.lineno, d.function if full else None)

def _debugGetListStr(iterable: Union[tuple, list, set, dict], useMultiline:bool=True, limitToLine: bool=False, minItems: int=2, maxItems: int=50) -> str:
    """ "Cast" a tuple, list, set or dict to a string, automatically shorten
        it if it's long, and display how long it is.

        Params:
            limitToLine: if True, limit the length of list to a single line
            minItems: show at least this many items in the list
            maxItems: show at most this many items in the list
            color: a simple int color

        Note:
            If limitToLine is True, it will overrule maxItems, but *not* minItems
    """
    def getBrace(opening):
        if type(iterable) is list:
            return '[' if opening else ']'
        elif type(iterable) in (set, dict):
            return '{' if opening else '}'
        else:
            return '(' if opening else ')'

    lengthAddOn = f'(len={len(iterable)})'
    defaultStr  = str(iterable)

    # Print in lines
    if (not limitToLine and len(defaultStr) + len(lengthAddOn) > (get_terminal_size().columns / 2)) or useMultiline:
        rtnStr = f'{lengthAddOn} {getBrace(True)}'
        if type(iterable) is dict:
            for key, val in iterable.items():
                rtnStr += f'\n\t<{type(key).__name__}> {key}: <{type(val).__name__}> {val}'
        else:
            for cnt, i in enumerate(iterable):
                rtnStr += f'\n\t{cnt}: <{type(i).__name__}> {i}'
        if len(iterable):
            rtnStr += '\n'
        rtnStr += getBrace(False)
    else:
        rtnStr = defaultStr + lengthAddOn

    return rtnStr


    """
    if type(v) in (tuple, list, set) and len(v) > minItems:
        if type(v) is set:
            v = tuple(v)

        ellipsis = f', ... '
        length = f'(len={len(v)})'

        if limitToLine:
            firstHalf  = str(v[0:round(minItems/2)])[:-1]
            secondHalf = str(v[-round(minItems/2)-1:-1])[1:]
            prevFirstHalf = firstHalf
            prevSecondHalf = secondHalf
            index = 0

            # The 54 is a fugde factor. I don't know why it needs to be there, but it works.
            while (6 + 54 + len(length) + len(firstHalf) + len(secondHalf)) < get_terminal_size().columns:
                index += 1
                firstHalf  = str(v[0:round((minItems+index)/2)])[:-1]
                secondHalf = str(v[-round((minItems+index)/2)-1:-1])[1:]
                prevFirstHalf = firstHalf
                prevSecondHalf = secondHalf
                if index > 6:
                    break

            firstHalf = prevFirstHalf
            secondHalf = prevSecondHalf

        else:
            firstHalf  = str(v[0:round(maxItems/2)])[:-1]
            secondHalf = str(v[-round(maxItems/2)-1:-1])[1:]

        return firstHalf + ellipsis + secondHalf + length

    else:
        return str(v) + f'(len={len(v)})'
    """

def _debugGetTypename(var, addBraces=True):
    def getUniqueType(item):
        returnMe = type(item).__name__
        while type(item) in (tuple, list, set):
            try:
                item = item[0]
            except (KeyError, IndexError, TypeError):
                returnMe += '('
                break
            returnMe += '(' + type(item).__name__

        cnt = 0
        for i in returnMe:
            if i == '(':
                cnt += 1
        return returnMe + (')'*cnt)

    if type(var) is dict:
        if len(var) > 0:
            rtn = f'dict({type(list(var.keys())[0]).__name__}:{type(list(var.values())[0]).__name__})'
        else:
            rtn = 'dict()'
    elif type(var) in (tuple, list, set, dict):
        types = []
        for i in var:
            types.append(getUniqueType(i))
        types = sorted(set(types), key=lambda x: types.index(x))
        fullName = type(var).__name__ + str(tuple(types)).replace("'", "")
        if len(types) == 1:
            fullName = fullName[:-2] + ')'
        rtn = fullName
    else:
        rtn = type(var).__name__
    return f'<{rtn}>' if addBraces else rtn

def _debugPrintLink(filename, lineNum, function=None):
    """ Print a VSCodium clickable file and line number
        If function is specified, a full python error message style line is printed
    """
    try:
        _printColor(40, 43, 46)
        if function is None: #    \|/  Oddly enough, this double quote is nessicary
            print('\t', filename, '", line ', lineNum, '\033[0m', sep='')
        else:
            print('\tFile "', filename, '", line ', lineNum, ', in ', function, sep='')

        _resetColor()
    finally:
        _resetColor()
    _resetColor()
    print('\033[0m', end='')

def _printDebugCount(leftAdjust=2, color: int=Colors.COUNT):
    global _debugCount
    _debugCount += 1
    with coloredOutput(color):
        print(f'{str(_debugCount)+":":<{leftAdjust+2}}', end='')

def _debugManualGetVarName(var, full=True, calls=2, metadata=None):
    try:
        return research(r'(?<=debug\().+(?=,(\s)?[name color showFunc showFile showPath useRepr calls background limitToLine minItems maxItems stackTrace raiseError clr _repr trace bg throwError throw \) ])',
                         metadata.code_context[0]).group()
    except:
        return '?'

def _debugGetVarName(var, full=True, calls=1, metadata=None):
    try:
        return argname('var', frame=calls+1)
    # It's a *likely* string literal
    except Exception as e:
        if type(var) is str:
            return None
        else:
            try:
                # print('var:', var)
                # print('var type:', type(var))
                return nameof(var, frame=calls+1)
            except Exception as e:
                if VERBOSE and DEBUGGING_DEBUG and not isinstance(var, Exception):
                    raise e
                else:
                    return _debugManualGetVarName(var, full, calls+1, metadata)
    except VarnameRetrievingError as e:
        if VERBOSE:
            raise e
        else:
            return _debugManualGetVarName(var, full, calls+1, metadata)

def _debugGetAdjustedFilename(filename):
    return filename[len(ROOT)+1:]

def _debugGetContext(metadata, useVscodeStyle, showFunc, showFile, showPath):
    #* Set the stuff in the [] (the "context")
    if metadata is not None:
        if useVscodeStyle:
            s = f'["{metadata.filename if showPath else _debugGetAdjustedFilename(metadata.filename)}", line {metadata.lineno}'
            if showFunc:
                if metadata.function.startswith('<'):
                    s += ', in Global Scope'
                else:
                    s += f', in {metadata.function}()'
            s += '] '
            return s
        else:
            context = str(metadata.lineno)
            if showFunc:
                if metadata.function.startswith('<'):
                    context = 'Global Scope' + context
                else:
                    context = metadata.function + '()->' + context

            if showFile:
                context = (metadata.filename if showPath else basename(metadata.filename)) + '->' + context

            return f'[{context}] '
    else:
        return ' '

def _debugPrintStackTrace(calls, useVscodeStyle, showFunc, showFile, showPath):
    for i in reversed(stack()[3:]):
        print('\t', _debugGetContext(i, useVscodeStyle, showFunc, showFile, showPath))

def _debugBeingUsedAsDecorator(funcName, metadata=None, calls=1) -> 'Union[1, 2, 3, False]':
    """ Return 1 if being used as a function decorator, 2 if as a class decorator, 3 if not sure, and False if neither. """
    if metadata is None:
        metadata = _debugGetMetaData(calls+1)

    # print(metadata.code_context)
    line = metadata.code_context[0]

    if funcName not in line:
        if 'def ' in line:
            return 1
        elif 'class ' in line:
            return 2
        elif '@' in line:
            return 3
    elif '@' in line:
        return 3

    return False

def printContext(calls=1, color=Colors.CONTEXT, showFunc=True, showFile=True, showPath=True):
    _printDebugCount()
    with coloredOutput(color):
        print(_debugGetContext(_debugGetMetaData(1 + calls), True,
                               showFunc or DISPLAY_FUNC,
                               showFile or DISPLAY_FILE,
                               showPath or DISPLAY_PATH), end='')

def debug(var=_None,                # The variable to debug
          name: str=None,           # Don't try to get the name, use this one instead
          color=_None,              # A number (0-5), a 3 item tuple/list, or None
          showFunc: bool=True,      # Expressly show what function we're called from
          showFile: bool=True,      # Expressly show what file we're called from
          showPath: bool=False,     # Show just the file name, or the full filepath
          useRepr: bool=False,      # Whether we should print the repr of var instead of str
          calls: int=1,             # Add extra calls
          background: bool=False,   # Whether the color parameter applies to the forground or the background
          limitToLine: bool=True,   # When printing iterables, whether we should only print items to the end of the line
          minItems: int=50,         # Minimum number of items to print when printing iterables (overrides limitToLine)
          maxItems: int=-1,         # Maximum number of items to print when printing iterables, use None or negative to specify no limit
          stackTrace: bool=False,   # Print a stack trace
          raiseError: bool=False,   # If var is an error type, raise it
          clr=_None,                # Alias of color
          repr: bool=False,         # Alias of useRepr
          trace: bool=False,        # Alias of stackTrace
          bg: bool=False,           # Alias of background
          throwError: bool=False,   # Alias of raiseError
          throw: bool=False         # Alias of raiseError
    ) -> "var":
    """ Print variable names and values for easy debugging.

        Usage:
            debug()          -> Prints a standard message to just tell you that it's getting called
            debug('msg')     -> Prints the string along with metadata
            debug(var)       -> Prints the variable name, type, and value
            foo = debug(bar) -> Prints the variable name, type, and value, and returns the variable
            @debug           -> Use as a decorator to make note of when the function is called

        Args:
            var: The variable or variables to print
            name: Manully specify the name of the variable
            color: A number between 0-5, or 3 or 4 tuple/list of color data to print the debug message as
            showFunc: Ensure that the function the current call is called from is shown
            showFile: Ensure that the file the current call is called from is shown
            showPath: Show the full path of the current file, isntead of it's relative path
            useRepr: Use __repr__() instead of __str__() on the given variable
            limitToLine: If var is a list/tuple/dict/set, only show as many items as will fit on one terminal line, overriden by minItems
            minItems: If var is a list/tuple/dict/set, don't truncate more than this many items
            maxItems: If var is a list/tuple/dict/set, don't show more than this many items
            stackTrace: Prints a neat stack trace of the current call
            calls: If you're passing in a return from a function, say calls=2
            background: Changes the background color instead of the forground color
            clr: Alias of color
            _repr: Alias of useRepr
            trace: Alias of stackTrace
            bg: Alias of background
    """
    stackTrace = stackTrace or trace
    useRepr = useRepr or repr
    background = background or bg
    throwError = throw or throwError or raiseError
    useColor = (Colors.DEFAULT_DEBUG if clr is _None else clr) if color is _None else color

    if maxItems < 0 or maxItems is None:
        maxItems = 1000000

    if isinstance(var, Warning):
        useColor = Colors.WARN
    elif isinstance(var, Exception):
        useColor = Colors.ALERT

    # +1 call because we don't want to get this line, but the one before it
    metadata = _debugGetMetaData(calls+1)

    #* First see if we're being called as a decorator
    if callable(var) and _debugBeingUsedAsDecorator('debug', metadata):
        def wrap(*args, **kwargs):
            # +1 call because we don't want to get this line, but the one before it
            metadata = _debugGetMetaData(2)

            _printDebugCount()

            if stackTrace:
                with coloredOutput(Colors.STACK_TRACE):
                    _debugPrintStackTrace(2, True, showFunc, showFile, showPath)


            with coloredOutput(Colors.NOTE_CALL):
                print(_debugGetContext(metadata, True, showFunc or DISPLAY_FUNC, showFile or DISPLAY_FILE, showPath or DISPLAY_PATH), end='')
                print(f'{var.__name__}() called!')
                # print(args)
            return var(*args, **kwargs)

        return wrap

    _printDebugCount()

    if stackTrace:
        with coloredOutput(Colors.STACK_TRACE):
            _debugPrintStackTrace(calls+1, True, showFunc, showFile, showPath)

    #* Only print the "HERE! HERE!" message
    if var is _None:
        with coloredOutput(useColor if color is not _None else Colors.EMPTY, not background):
            print(_debugGetContext(metadata, True, showFunc or DISPLAY_FUNC, showFile or DISPLAY_FILE, showPath or DISPLAY_PATH), end='')
            if not metadata.function.startswith('<'):
                print(f'{metadata.function}() called ', end='')
            print('HERE!')
        return


    metadataColor = darken(Colors.DEBUG_METADATA_DARKEN,  useColor)
    typeColor     = darken(Colors.DEBUG_TYPE_DARKEN,  useColor)
    nameColor     = darken(Colors.DEBUG_NAME_DARKEN, useColor)
    equalsColor   = Colors.DEBUG_EQUALS
    valueColor    = darken(Colors.DEBUG_VALUE_DARKEN, useColor)
    #* Print the standard line
    with coloredOutput(metadataColor, not background):
        print(_debugGetContext(metadata, True,
                                showFunc or DISPLAY_FUNC,
                                showFile or DISPLAY_FILE,
                                showPath or DISPLAY_PATH), end='')

    #* Seperate the variables into a tuple of (typeStr, varString)
    varType = _debugGetTypename(var)
    if useRepr:
        varVal = _repr(var)
    else:
        if type(var) in (tuple, list, set, dict):
            varVal  = _debugGetListStr(var, limitToLine, minItems, maxItems)
        else:
            varVal  = str(var)

    with coloredOutput(nameColor, not background):
        #* Actually get the name
        varName = _debugGetVarName(var, calls=calls, metadata=metadata) if name is None else name
        # It's a string literal
        if varName is None:
            print(var)
            return

    with coloredOutput(typeColor, not background):
        print(varType, end=' ')
    with coloredOutput(nameColor, not background):
        print(varName, end=' ')
    with coloredOutput(equalsColor, not background):
        print('=', end=' ')
    with coloredOutput(valueColor, not background):
        print(varVal)

    if isinstance(var, Exception) and throwError:
        raise var

    # Does the same this as debugged used to
    return var


################################### Other Logging-Type Functions ###################################
def log(message, levelReq=LogLevel.LOGGING, color=Colors.LOG_COLOR):
    if LOG_LEVEL >= levelReq.value:
        printContext(2)
        with coloredOutput(color):
            print(message)

def warn(message):
    log(message, color=Colors.WARN)
warning = warn


################################### Decorators ###################################
def todo(featureName=None, enabled=True, blocking=False, showFunc=True, showFile=True, showPath=False):
    """ Leave reminders for yourself to finish parts of your code.
        Can be manually turned on or off with hideAllTodos(bool).
        Can also be used as a decorator (function, or class) to print a reminder
        and also throw a NotImplemented error on being called/constructed.
    """
    metadata  = _debugGetMetaData(2)
    situation = _debugBeingUsedAsDecorator('todo', metadata)
    # def decorator(*decoratorArgs, **decoratorKwArgs):
    #     def wrap(func):
    #         def innerWrap(*funcArgs, **funcKwArgs):
    #             return func(*funcArgs, **funcKwArgs)
    #         return innerWrap
    #     return wrap

    def printTodo(disableFunc):
        if not HIDE_TODO and enabled:
            _printDebugCount()
            with coloredOutput(Colors.TODO):
                print(_debugGetContext(metadata, True,
                                      (showFunc or DISPLAY_FUNC) and not disableFunc,
                                       showFile or DISPLAY_FILE,
                                       showPath or DISPLAY_PATH), end='')
                # This is coincidental, but it works
                print(f'TODO: {featureName.__name__ if disableFunc else featureName}')
            if blocking:
                raise NotImplementedError()

    # Being used as a function decorator, or we're not sure
    if situation in (1, 3):
        def wrap(func):
            def innerWrap(*funcArgs, **funcKwArgs):
                printTodo(True)
                if blocking:
                    raise NotImplementedError()
                return featureName(*funcArgs, **funcKwArgs)
            return innerWrap
        return wrap

    elif situation == 2:
        def wrap(clas):
            def raiseErr(*_, **kw_):
                raise NotImplementedError()
            printTodo(True)
            if blocking:
                featureName.__init__ = raiseErr
        return featureName
    else:
        printTodo(False)

def confidence(level, interpretAs:int=None):
    def wrap(func):
        def innerWrap(*funcArgs, **funcKwArgs):
            definiteFailResponses = ()
            possiblyFailResponses = ()
            probablyFailResponses = ()

            def getPrettyLevel():
                if type(level) is str:
                    return f'({level} confident)'
                else:
                    assert(type(level) in (int, float))
                    return f'({level}% confidence)'

            def definiteFail():
                raise UserWarning(f"{func.__name__} is going to fail. {getPrettyLevel()}")

            def probablyFail():
                printContext(3, darken(80, Colors.ALERT), showFunc=False)
                with coloredOutput(Colors.ALERT):
                    print(f"Warning: {func.__name__} will probably fail. {getPrettyLevel()}")

            def possiblyFail():
                printContext(3, darken(80, Colors.CONFIDENCE_WARNING), showFunc=False)
                with coloredOutput(Colors.CONFIDENCE_WARNING):
                    print(f"Warning: {func.__name__} might not work. {getPrettyLevel()}")

            def unknownInput():
                # If we don't understand the input, just give a soft warning (it will display what the input is, anyway)
                possiblyFail()
                return

                if interpretAs is None:
                    raise TypeError(f"I don't recognize {level} as a confidence level.")

                if interpretAs > 100:
                    raise TypeError(f"You can't be {interpretAs}% confident, that's not how it works.")
                elif interpretAs < 0:
                    # replaceLine(f'\n\t\t\t\t\t\t"{level.lower()},', offset=+2)
                    definiteFailResponses += (

                    )
                    definiteFail()
                elif interpretAs < 20:
                    # replaceLine(f'\n\t\t\t\t\t\t"{level.lower()},', offset=+2)
                    probablyFailResponses += (

                    )
                    probablyFail()
                elif interpretAs < 50:
                    # replaceLine(f'\n\t\t\t\t\t\t"{level.lower()},', offset=+2)
                    possiblyFailResponses += (

                    )
                    possiblyFail()

            if type(level) in (int, float):
                # This is a pet-peeve
                if level > 100:
                    raise TypeError(f"You can't be {level}% confident, that's not how it works.")
                elif level < 0:
                    definiteFail()
                elif level < 20:
                    probablyFail()
                elif level < 50:
                    possiblyFail()
            elif type(level) is str:
                l = level.lower()
                if l in CommonResponses.NO or l in CommonResponses.LOW_AMOUNT or l in probablyFailResponses:
                    probablyFail()
                elif l in CommonResponses.MAYBE or l in CommonResponses.SOME_AMOUNT or l in possiblyFailResponses:
                    possiblyFail()
                elif l in definiteFailResponses:
                    definitelyFail()
                elif l not in CommonResponses.YES and l not in CommonResponses.HIGH_AMOUNT and \
                     l not in CommonResponses.NA  and l not in CommonResponses.MODERATE_AMOUNT:
                    unknownInput()
            else:
                unknownInput()
            return func(*funcArgs, **funcKwArgs)
        return innerWrap
    return wrap
confident = confidence
untested = confidence(21)
tested = confidence(80)

def depricated(why=''):
    def wrap(func):
        def innerWrap(*funcArgs, **funcKwArgs):
            printContext(2, darken(80, Colors.DEPRICATED_WARNING))
            with coloredOutput(Colors.DEPRICATED_WARNING):
                print(f"{func.__name__} is Depricated{': ' if len(why) else '.'}{why}")
            return func(*funcArgs, **funcKwArgs)
        return innerWrap
    return wrap

def reprise(obj, *args, **kwargs):
    """ Sets the __repr__ function to the __str__ function of a class.
        Useful for custom classes with overloaded string functions
    """
    obj.__repr__ = obj.__str__
    return obj


################################### Iterable Utilities ###################################
def isiterable(obj, includeStr=False):
    return isinstance(obj, Iterable) and (type(obj) is not str if not includeStr else True)

def ensureIterable(obj, useList=False):
    if not isiterable(obj):
        return [obj, ] if useList else (obj, )
    else:
        return obj

def ensureNotIterable(obj, emptyBecomes=_None):
    if isiterable(obj):
        # Generators are iterable, but don't inherantly have a length
        try:
            len(obj)
        except:
            obj = list(obj)

        if len(obj) == 1:
            try:
                return obj[0]
            except TypeError:
                return list(obj)[0]
        elif len(obj) == 0:
            return obj if emptyBecomes is _None else emptyBecomes
        else:
            return obj
    else:
        return obj

def flattenList(iterable, recursive=False, useList=True):
    if recursive:
        raise NotImplementedError

    useType = list if useList else type(iterable)
    rtn = useType()
    for i in iterable:
        rtn += useType(i)
    return rtn
    # print(flattenList(('a', 'b', [1, 2, 3]), useList=False))

def removeDuplicates(iterable, method='sorted set'):
    method = method.lower()
    if method == 'set':
        return type(iterable)(set(iterable))
    elif method == 'sorted set':
        return list(sorted(set(iterable), key=lambda x: iterable.index(x)))
    elif method == 'generator':
        seen = set()
        for item in seq:
            if item not in seen:
                seen.add( item )
                yield item
    elif method == 'manual':
        dups = {}
        newlist = []
        for x in biglist:
            if x['link'] not in dups:
                newlist.append(x)
                dups[x['link']] = None
    elif method == 'other':
        seen_links = set()
        for index in len(biglist):
            link = biglist[index]['link']
            if link in seen_links:
                del(biglist[index])
            seen_links.add(link)
    else:
        raise TypeError(f'Unknown removeDuplicates method: {method}. Options are: (set, sorted set, generator, manual, other)')

# def removeRedundant(iterable):
    # """ Remove any lists with only """
    # if len(iterable) == 1:

    # for i in iterable:
    #     if isiterable(i) and

@confidence(10)
def normalizeList(iterable, ensureList=False):
    debug()
    if ensureList:
        return list(removeDuplicates(flattenList(ensureIterable(list(iterable), True))))
    else:
        return list(ensureNotIterable(removeDuplicates(flattenList(ensureIterable(list(iterable), True)))))

def getIndexWith(obj, key):
    """ Returns the index of the first object in a list in which key returns true to.
    Example: getIndexWith([ [5, 3], [2, 3], [7, 3] ], lambda x: x[0] + x[1] == 10) -> 2
    If none are found, returns None
    """
    for cnt, i in enumerate(obj):
        if key(i):
            return cnt
    return None

def invertDict(d):
    """ Returns the dict given, but with the keys as values and the values as keys. """
    return dict(zip(d.values(), d.keys()))

@todo
class LoopingList(list):
    """ It's a list, that, get this, loops!
    """
    def __getitem__(self, index):
        if index > self.__len__():
            return super().__getitem__(index % self.__len__())
        else:
            return super().__getitem__(index)


################################### Timing Utilities ###################################
timingData = {}
def timeFunc(func, accuracy=5):
    """ A function decorator that prints how long it takes for a function to run """
    def wrap(*params, **kwparams):
        global timingData

        t = process_time()

        returns = func(*params, **kwparams)

        t2 = process_time()

        elapsed_time = round(t2 - t, accuracy)
        name = func.__name__

        try:
            timingData[name] += (elapsed_time,)
        except KeyError:
            timingData[name] = (elapsed_time,)

        _printDebugCount()
        # print(name, ' ' * (10 - len(name)), 'took', elapsed_time if elapsed_time >= 0.00001 else 0.00000, '\ttime to run.')
        print(f'{name:<12} took {elapsed_time:.{accuracy}f} seconds to run.')
        #  ' ' * (15 - len(name)),
        return returns
    return wrap

def _printTimingData(accuracy=5):
    """ I realized *after* I wrote this that this is a essentially profiler. Oops. """
    global timingData
    if len(timingData):
        print()

        maxName = len(max(timingData.keys(), key=len))
        maxNum  = len(str(len(max(timingData.values(), key=lambda x: len(str(len(x)))))))
        for name, times in reversed(sorted(timingData.items(), key=lambda x: sum(x[1]))):
            print(f'{name:<{maxName}} was called {len(times):<{maxNum}} times taking {sum(times)/len(times):.{accuracy}f} seconds on average for a total of {sum(times):.{accuracy}f} seconds.')
registerExit(_printTimingData)

class getTime:
    """ A class to use with a with statement like so:
        with getTime('sleep'):
            time.sleep(10)
        It will then print how long the enclosed code took to run.
    """
    def __init__(self, name, accuracy=5):
        self.name = name
        self.accuracy = accuracy

    def __enter__(self):
        self.t = process_time()

    def __exit__(self, *args):
        # args is completely useless, not sure why it's there.
        t2 = process_time()
        elapsed_time = round(t2 - self.t, self.accuracy)
        print(self.name, ' ' * (15 - len(self.name)), 'took', f'{elapsed_time:.{self.accuracy}f}', '\ttime to run.')


################################### Misc. Useful Classes ###################################
class FunctionCall:
    """ A helpful class that represents an as-yet uncalled function call with parameters """
    def __init__(self, func=lambda: None, args=(), kwargs={}):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def call(self, *args, override_args=False, **kwargs):
        return self.__call__(*args, override_args, **kwargs)

    def __call__(self, *args, override_args=False, **kwargs):
        """ If you specify parameters and don't explicitly set override_args to True,
            then the given parameters are ignored and the previously set parameters are used.
        """
        if override_args:
            return self.func(*args, **kwargs)
        else:
            return self.func(*self.args, **self.kwargs)

class Signal:
    """ A custom Signal implementation. Connect with the connect() function """
    def __init__(self, *args, **kwargs):
        self.viableArgsCount = len(args)
        self.viableKwargs = kwargs.keys()
        self.funcs = []
        self.call = self.__call__

    def connect(self, func):
        self.funcs.append(FunctionCall(func))

    def emit(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        for f in self.funcs:
            # Verify that the parameters are valid
            if len(args) != self.viableArgsCount:
                raise TypeError("Signal emitted with incorrect number of parameters")
            for i in kwargs.keys():
                if i not in self.viableKwargs:
                    raise TypeError("Signal emitted with invalid keyword argument")
            f(*args, override_args=True, **kwargs)

class KeyStandard:
    ascii = {
        "unknown":0,
        "space":32,
        "exclamation":33,
        "doubleQuote":34,
        "pound":35,
        "dollarSign":36,
        "percent":37,
        "andersand":38,
        "singleQuote":39,
        "openParen":40,
        "closeParen":41,
        "star":42,
        "plus":43,
        "comma":44,
        "minus":45,
        "period":46,
        "slash":47,
        "one":48,
        "two":49,
        "three":50,
        "four":51,
        "five":52,
        "six":53,
        "seven":54,
        "eight":55,
        "nine":56,
        "zero":57,
        "colon":58,
        "semicolon":59,
        "lessThan":60,
        "equals":61,
        "greaterThan":62,
        "question":63,
        "attersand":64,
        "A":65,
        "B":66,
        "C":67,
        "D":68,
        "E":69,
        "F":70,
        "G":71,
        "H":72,
        "I":73,
        "J":74,
        "K":75,
        "L":76,
        "M":77,
        "N":78,
        "O":79,
        "P":80,
        "Q":81,
        "R":82,
        "S":83,
        "T":84,
        "U":85,
        "V":86,
        "W":87,
        "X":88,
        "Y":89,
        "Z":90,
        "openSquareBracket":91,
        "backslash":92,
        "closeSquareBracket":93,
        "exponent":94,
        "underscore":95,
        "sidilla":96,
        "a":97,
        "b":98,
        "c":99,
        "d":100,
        "e":101,
        "f":102,
        "g":103,
        "h":104,
        "i":105,
        "j":106,
        "k":107,
        "l":108,
        "m":109,
        "n":110,
        "o":111,
        "p":112,
        "q":113,
        "r":114,
        "s":115,
        "t":116,
        "u":117,
        "v":118,
        "w":119,
        "x":120,
        "y":121,
        "z":122,
        "openCurlyBrace":123,
        "orLine":124,
        "closeCurlyBrace":125,
        "tilde":126,
        "F1":131,
        "F2":132,
        "F3":133,
        "F4":134,
        "F5":135,
        "F6":136,
        "F7":137,
        "F8":138,
        "F9":139,
        "F10":140,
        "F11":141,
        "F12":142,
        "F13":143,
        "F14":144,
        "F15":145,
        "F16":146,
        "F17":147,
        "F18":148,
        "F19":149,
        "F20":150,
        "F21":151,
        "F22":152,
        "F23":153,
        "F24":154,
        "F25":155,
        "F26":156,
        "F27":157,
        "F28":158,
        "F29":159,
        "F30":160,
        "escape":27,
        "delete":127,
        "ctrl":200,
        "rightCtrl": 206,
        "leftCtrl": 207,
        # I don't *think* there are left and right win keys
        "win":201,
        "shift":202,
        "rightShift": 208,
        "leftShift": 209,
        "alt":203,
        "rightAlt": 204,
        "leftAlt": 205,
        "backspace":300,
        "home":301,
        "end":302,
        "enter":303,
        "insert":304,
        "pageUp":305,
        "pageDown":306,
        "up":307,
        "down":308,
        "left":309,
        "right":310,
        "printScreen":311,
        "scrollLock":312,
        "pause":313,
        "stop":314,
        "next":315,
        "previous":316,
        "break_":317,
        "numLock":318,
        "cent":319,
        "volumeDown":320,
        "volumeUp":321,
        "mute":322,
        "menu":323,
        "refresh": 324,
        "back":325,
        "touchpadLock":326,
        "muteMic":327,
        "brightnessUp":328,
        "brightnessDown":329,
        "keypad0": 330,
        "keypad1": 331,
        "keypad2": 332,
        "keypad3": 333,
        "keypad4": 334,
        "keypad5": 335,
        "keypad6": 336,
        "keypad7": 337,
        "keypad8": 338,
        "keypad9": 339,
        "accept": 340,
        "add": 341,
        "apps": 342,
        "browserForward":342,
        "browserFavorites": 343,
        "browserHome": 344,
        "search": 345,
        "browserStop": 346,
        "clear": 347,
        "convert": 348,
        "execute": 349,
        "final": 350,
        "fn": 351,
        "hanguel": 352,
        "hangul": 353,
        "hanja": 354,
        "help": 355,
        "home": 356,
        "junja": 357,
        "kana": 358,
        "kanji": 359,
        "launchApp1": 360,
        "launchApp2": 361,
        "launchMail": 362,
        "launchMediaSelect": 363,
        "changeMode": 364,
        "nonconvert": 365,
        "select": 366,
        "separator": 367,
        "sleep": 368,
        "leftWin": 369,
        "rightWin": 370,
        "yen": 371,
        "option": 372,
        "leftOption": 373,
        "rightOption": 374,
        "print": 376,
        "tab": 377,
        "capsLock": 378,
    }

    string = {
        "unknown": ascii["unknown"],
        "space": ascii["space"],
        'spacebar': ascii['space'],
        "exclamation": ascii["exclamation"],
        "exclamationMark": ascii["exclamation"],
        "exclamationPoint": ascii["exclamation"],
        "doubleQuote": ascii["doubleQuote"],
        "pound": ascii["pound"],
        "hashtag": ascii["pound"],
        "numSign": ascii["pound"],
        "numberSign": ascii["pound"],
        "dollarSign": ascii["dollarSign"],
        "dollar": ascii["dollarSign"],
        "percent": ascii["percent"],
        "andersand": ascii["andersand"],
        "and": ascii["andersand"],
        "and_": ascii["andersand"],
        "singleQuote": ascii["singleQuote"],
        "openParen": ascii["openParen"],
        "openParenthesis": ascii["openParen"],
        "closeParen": ascii["closeParen"],
        "closeParenthesis": ascii["closeParen"],
        "star": ascii["star"],
        "times": ascii["star"],
        "plus": ascii["plus"],
        "comma": ascii["comma"],
        "minus": ascii["minus"],
        "period": ascii["period"],
        "dot": ascii["period"],
        "slash": ascii["slash"],
        "forwardSlash": ascii["slash"],
        "one": ascii["one"],
        "two": ascii["two"],
        "three": ascii["three"],
        "four": ascii["four"],
        "five": ascii["five"],
        "six": ascii["six"],
        "seven": ascii["seven"],
        "eight": ascii["eight"],
        "nine": ascii["nine"],
        "zero": ascii["zero"],
        "colon": ascii["colon"],
        "semicolon": ascii["semicolon"],
        "lessThan": ascii["lessThan"],
        "less": ascii["lessThan"],
        "equals": ascii["equals"],
        "greaterThan": ascii["greaterThan"],
        "greater": ascii["greaterThan"],
        "question": ascii["question"],
        "questionMark": ascii["question"],
        "attersand": ascii["attersand"],
        "at": ascii["attersand"],
        "A": ascii["A"],
        "B": ascii["B"],
        "C": ascii["C"],
        "D": ascii["D"],
        "E": ascii["E"],
        "F": ascii["F"],
        "G": ascii["G"],
        "H": ascii["H"],
        "I": ascii["I"],
        "J": ascii["J"],
        "K": ascii["K"],
        "L": ascii["L"],
        "M": ascii["M"],
        "N": ascii["N"],
        "O": ascii["O"],
        "P": ascii["P"],
        "Q": ascii["Q"],
        "R": ascii["R"],
        "S": ascii["S"],
        "T": ascii["T"],
        "U": ascii["U"],
        "V": ascii["V"],
        "W": ascii["W"],
        "X": ascii["X"],
        "Y": ascii["Y"],
        "Z": ascii["Z"],
        "openSquareBrace": ascii["openSquareBracket"],
        "openingSquareBrace": ascii['openSquareBracket'],
        "openSquareBracket": ascii['openSquareBracket'],
        "backslash": ascii["backslash"],
        "closeSquareBrace": ascii["closeSquareBracket"],
        "closingSquareBrace": ascii["closeSquareBracket"],
        "closeSquareBracket": ascii["closeSquareBracket"],
        "exponent": ascii["exponent"],
        "underscore": ascii["underscore"],
        "sidilla": ascii["sidilla"],
        "accent": ascii["sidilla"],
        "a": ascii["a"],
        "b": ascii["b"],
        "c": ascii["c"],
        "d": ascii["d"],
        "e": ascii["e"],
        "f": ascii["f"],
        "g": ascii["g"],
        "h": ascii["h"],
        "i": ascii["i"],
        "j": ascii["j"],
        "k": ascii["k"],
        "l": ascii["l"],
        "m": ascii["m"],
        "n": ascii["n"],
        "o": ascii["o"],
        "p": ascii["p"],
        "q": ascii["q"],
        "r": ascii["r"],
        "s": ascii["s"],
        "t": ascii["t"],
        "u": ascii["u"],
        "v": ascii["v"],
        "w": ascii["w"],
        "x": ascii["x"],
        "y": ascii["y"],
        "z": ascii["z"],
        "openCurlyBrace": ascii["openCurlyBrace"],
        "openingCurlyBrace": ascii["openCurlyBrace"],
        "openCurlyBracket": ascii["openCurlyBrace"],
        "openingCurlyBracket": ascii["openCurlyBrace"],
        "orLine": ascii["orLine"],
        "line": ascii["orLine"],
        "verticalSlash": ascii["orLine"],
        "closeCurlyBrace": ascii["closeCurlyBrace"],
        "closingCurlyBrace": ascii["closeCurlyBrace"],
        "closeCurlyBracket": ascii["closeCurlyBrace"],
        "closingCurlyBracket": ascii["closeCurlyBrace"],
        "tilde": ascii["tilde"],
        "F1": ascii["F1"],
        "F2": ascii["F2"],
        "F3": ascii["F3"],
        "F4": ascii["F4"],
        "F5": ascii["F5"],
        "F6": ascii["F6"],
        "F7": ascii["F7"],
        "F8": ascii["F8"],
        "F9": ascii["F9"],
        "F10": ascii["F10"],
        "F11": ascii["F11"],
        "F12": ascii["F12"],
        "F13": ascii["F13"],
        "F14": ascii["F14"],
        "F15": ascii["F15"],
        "F16": ascii["F16"],
        "F17": ascii["F17"],
        "F18": ascii["F18"],
        "F19": ascii["F19"],
        "F20": ascii["F20"],
        "F21": ascii["F21"],
        "F22": ascii["F22"],
        "F23": ascii["F23"],
        "F24": ascii["F24"],
        "F25": ascii["F25"],
        "F26": ascii["F26"],
        "F27": ascii["F27"],
        "F28": ascii["F28"],
        "F29": ascii["F29"],
        "F30": ascii["F30"],
        "f1": ascii["F1"],
        "f2": ascii["F2"],
        "f3": ascii["F3"],
        "f4": ascii["F4"],
        "f5": ascii["F5"],
        "f6": ascii["F6"],
        "f7": ascii["F7"],
        "f8": ascii["F8"],
        "f9": ascii["F9"],
        "f10": ascii["F10"],
        "f11": ascii["F11"],
        "f12": ascii["F12"],
        "f13": ascii["F13"],
        "f14": ascii["F14"],
        "f15": ascii["F15"],
        "f16": ascii["F16"],
        "f17": ascii["F17"],
        "f18": ascii["F18"],
        "f19": ascii["F19"],
        "f20": ascii["F20"],
        "f21": ascii["F21"],
        "f22": ascii["F22"],
        "f23": ascii["F23"],
        "f24": ascii["F24"],
        "f25": ascii["F25"],
        "f26": ascii["F26"],
        "f27": ascii["F27"],
        "f28": ascii["F28"],
        "f29": ascii["F29"],
        "f30": ascii["F30"],
        "escape": ascii["escape"],
        "delete": ascii["delete"],
        "ctrl": ascii["ctrl"],
        "control": ascii["ctrl"],
        "win": ascii["win"],
        "command": ascii['win'],
        "cmd": ascii['win'],
        "windows": ascii['win'],
        "windowsKey": ascii['win'],
        "shift": ascii["shift"],
        "alt": ascii["alt"],
        "rightCtrl": ascii["rightCtrl"],
        "ctrlRight": ascii["rightCtrl"],
        "leftCtrl": ascii["leftCtrl"],
        "ctrlLeft": ascii["leftCtrl"],
        "rightShift": ascii["rightShift"],
        "shiftRight": ascii["rightShift"],
        "leftShift": ascii["leftShift"],
        "shiftLeft": ascii["leftShift"],
        "rightAlt": ascii["rightAlt"],
        "altRight": ascii["rightAlt"],
        "leftAlt": ascii["leftAlt"],
        "altLeft": ascii["leftAlt"],
        "alternate": ascii["alt"],
        "backspace": ascii["backspace"],
        "home": ascii["home"],
        "end": ascii["end"],
        "enter": ascii["enter"],
        "return": ascii["enter"],
        "insert": ascii["insert"],
        "pageUp": ascii["pageUp"],
        "pageDown": ascii["pageDown"],
        "up": ascii["up"],
        "down": ascii["down"],
        "left": ascii["left"],
        "right": ascii["right"],
        "printScreen": ascii["printScreen"],
        "prtscr": ascii['printScreen'],
        "prtScr": ascii['printScreen'],
        "PrtScr": ascii['printScreen'],
        "scrollLock": ascii["scrollLock"],
        "pause": ascii["pause"],
        "play": ascii["pause"],
        "stop": ascii["stop"],
        "forward": ascii["next"],
        "next": ascii["next"],
        "skip": ascii['next'],
        "skipAhead": ascii['next'],
        "skipForward": ascii['next'],
        "backward": ascii["previous"],
        "prev": ascii["previous"],
        "previous": ascii["previous"],
        "skipBack": ascii['previous'],
        "skipBackward": ascii['previous'],
        "break_": ascii["break_"],
        "numLock": ascii["numLock"],
        "cent": ascii["cent"],
        "\'": ascii["singleQuote"],
        "\"": ascii["doubleQuote"],
        "\\": ascii["backslash"],
        "`": ascii["sidilla"],
        "~": ascii["tilde"],
        "!": ascii["exclamation"],
        "@": ascii["attersand"],
        "#": ascii["pound"],
        "$": ascii["dollarSign"],
        "%": ascii["percent"],
        "^": ascii["exponent"],
        "&": ascii["andersand"],
        "*": ascii["star"],
        "(": ascii["openParen"],
        ")": ascii["closeParen"],
        "_": ascii["underscore"],
        "+": ascii["plus"],
        "{": ascii["openCurlyBrace"],
        "}": ascii["closeCurlyBrace"],
        "|": ascii["orLine"],
        ":": ascii["colon"],
        "<": ascii["lessThan"],
        ">": ascii["greaterThan"],
        "?": ascii["question"],
        "1": ascii["one"],
        "2": ascii["two"],
        "3": ascii["three"],
        "4": ascii["four"],
        "5": ascii["five"],
        "6": ascii["six"],
        "7": ascii["seven"],
        "8": ascii["eight"],
        "9": ascii["nine"],
        "0": ascii["zero"],
        "-": ascii["minus"],
        "=": ascii["equals"],
        "[": ascii["openSquareBracket"],
        "]": ascii["closeSquareBracket"],
        ",": ascii["comma"],
        ".": ascii["period"],
        "/": ascii["slash"],
        ';': ascii["semicolon"],
        "return": ascii["enter"],
        "and": ascii["andersand"],
        "volumeDown":ascii["volumeDown"],
        "volumeUp":ascii["volumeUp"],
        "mute":ascii["mute"],
        "menu":ascii["menu"],
        "refresh": ascii["refresh"],
        "back":ascii["back"],
        "touchpadLock":ascii["touchpadLock"],
        "lockTouchpad":ascii["touchpadLock"],
        "muteMic":ascii["muteMic"],
        "micMute":ascii["muteMic"],
        "brightnessUp":ascii["brightnessUp"],
        "brightnessDown":ascii["brightnessDown"],
        "keypad0": ascii["keypad0"],
        "kp0": ascii["keypad0"],
        "keypad1": ascii["keypad1"],
        "kp1": ascii["keypad1"],
        "keypad2": ascii["keypad2"],
        "kp2": ascii["keypad2"],
        "keypad3": ascii["keypad3"],
        "kp3": ascii["keypad3"],
        "keypad4": ascii["keypad4"],
        "kp4": ascii["keypad4"],
        "keypad5": ascii["keypad5"],
        "kp5": ascii["keypad5"],
        "keypad6": ascii["keypad6"],
        "kp6": ascii["keypad6"],
        "keypad7": ascii["keypad7"],
        "kp7": ascii["keypad7"],
        "keypad8": ascii["keypad8"],
        "kp8": ascii["keypad8"],
        "keypad9": ascii["keypad9"],
        "kp9": ascii["keypad9"],
        # Compatability with pyautogui
        '\t': ascii['tab'],
        '\\t': ascii['tab'],
        '   ': ascii['tab'],
        '\\n': ascii['enter'],
        '\n': ascii['enter'],
        '\\r': ascii['enter'],
        '\r': ascii['enter'],
        " ": ascii["space"],
        "accept": ascii["accept"],
        "add": ascii["add"],
        "alt": ascii["alt"],
        "altleft": ascii["leftAlt"],
        "altright": ascii["rightAlt"],
        "apps": ascii["apps"],
        "backspace": ascii["backspace"],
        "browserback": ascii["back"],
        "browserfavorites": ascii["browserFavorites"],
        "browserforward": ascii["browserForward"],
        "browserhome": ascii["browserHome"],
        "browserrefresh": ascii["refresh"],
        "browsersearch": ascii["search"],
        "browserstop": ascii["browserStop"],
        "capslock": ascii["capsLock"],
        "capsLock": ascii["capsLock"],
        "clear": ascii["clear"],
        "convert": ascii["convert"],
        "ctrl": ascii["ctrl"],
        "ctrlleft": ascii["leftCtrl"],
        "ctrlright": ascii["rightCtrl"],
        "decimal": ascii["period"],
        "del": ascii["delete"],
        "delete": ascii["delete"],
        "divide": ascii["slash"],
        "down": ascii["down"],
        "end": ascii["end"],
        "enter": ascii["enter"],
        "esc": ascii["escape"],
        "escape": ascii["escape"],
        "execute": ascii["execute"],
        "option": ascii["option"],
        "final": ascii["final"],
        "fn": ascii["fn"],
        "hanguel": ascii["hanguel"],
        "hangul": ascii["hangul"],
        "hanja": ascii["hanja"],
        "help": ascii["help"],
        "home": ascii["home"],
        "insert": ascii["insert"],
        "junja": ascii["junja"],
        "kana": ascii["kana"],
        "kanji": ascii["kanji"],
        "launchapp1": ascii["launchApp1"],
        "launchapp2": ascii["launchApp2"],
        "launchmail": ascii["launchMail"],
        "launchmediaselect": ascii["launchMediaSelect"],
        "left": ascii["left"],
        "modechange": ascii["changeMode"],
        "multiply": ascii["star"],
        "nexttrack": ascii["next"],
        "nonconvert": ascii["nonconvert"],
        "num0": ascii["keypad0"],
        "num1": ascii["keypad1"],
        "num2": ascii["keypad2"],
        "num3": ascii["keypad3"],
        "num4": ascii["keypad4"],
        "num5": ascii["keypad5"],
        "num6": ascii["keypad6"],
        "num7": ascii["keypad7"],
        "num8": ascii["keypad8"],
        "num9": ascii["keypad9"],
        "numlock": ascii["numLock"],
        "pagedown": ascii["pageDown"],
        "pageup": ascii["pageUp"],
        "pause": ascii["pause"],
        "pgdn": ascii["pageDown"],
        "pgup": ascii["pageUp"],
        "playpause": ascii["pause"],
        "prevtrack": ascii["back"],
        "print": ascii["print"],
        "printscreen": ascii["printScreen"],
        "prntscrn": ascii["printScreen"],
        "prtsc": ascii["printScreen"],
        "prtscr": ascii["printScreen"],
        "return": ascii["enter"],
        "right": ascii["right"],
        "scrolllock": ascii["scrollLock"],
        "select": ascii["select"],
        "separator": ascii["orLine"],
        "shift": ascii["shift"],
        "shiftleft": ascii["leftShift"],
        "shiftright": ascii["rightShift"],
        "sleep": ascii["sleep"],
        "space": ascii["space"],
        "stop": ascii["stop"],
        "subtract": ascii["minus"],
        "tab": ascii["tab"],
        "up": ascii["up"],
        "volumedown": ascii["volumeDown"],
        "volumemute": ascii["mute"],
        "volumeup": ascii["volumeUp"],
        "win": ascii["win"],
        "winleft": ascii["leftWin"],
        "winright": ascii["rightWin"],
        "yen": ascii["yen"],
        "command": ascii["win"],
        "option": ascii["option"],
        "optionleft": ascii["leftOption"],
        "optionright": ascii["rightOption"],
    }

    # Qt = ({ } if not checkImport('pyside6', 'keyboard', fatal=False, printWarning=False) else {
    # })

    pynput = ({ } if not checkImport('pynput', 'keyboard', fatal=False, printWarning=False) else {
        keyboard.Key.alt:               string['alt'],
        keyboard.Key.alt_gr:            string['alt'],
        keyboard.Key.alt_l:             string['leftAlt'],
        keyboard.Key.alt_r:             string['rightAlt'],
        keyboard.Key.backspace:         string['backspace'],
        keyboard.Key.caps_lock:         string['capsLock'],
        keyboard.Key.cmd:               string['cmd'],
        # keyboard.Key.cmd_l:             string['cmd_l'],
        # keyboard.Key.cmd_r:             string['cmd_r'],
        keyboard.Key.ctrl:              string['ctrl'],
        keyboard.Key.ctrl_l:            string['leftCtrl'],
        keyboard.Key.ctrl_r:            string['rightCtrl'],
        keyboard.Key.delete:            string['delete'],
        keyboard.Key.down:              string['down'],
        keyboard.Key.end:               string['end'],
        keyboard.Key.enter:             string['enter'],
        keyboard.Key.esc:               string['esc'],
        keyboard.Key.f1:                string['f1'],
        keyboard.Key.f2:                string['f2'],
        keyboard.Key.f3:                string['f3'],
        keyboard.Key.f4:                string['f4'],
        keyboard.Key.f5:                string['f5'],
        keyboard.Key.f6:                string['f6'],
        keyboard.Key.f7:                string['f7'],
        keyboard.Key.f8:                string['f8'],
        keyboard.Key.f9:                string['f9'],
        keyboard.Key.f10:               string['f10'],
        keyboard.Key.f11:               string['f11'],
        keyboard.Key.f12:               string['f12'],
        keyboard.Key.f13:               string['f13'],
        keyboard.Key.f14:               string['f14'],
        keyboard.Key.f15:               string['f15'],
        keyboard.Key.f16:               string['f16'],
        keyboard.Key.f17:               string['f17'],
        keyboard.Key.f18:               string['f18'],
        keyboard.Key.f19:               string['f19'],
        keyboard.Key.f20:               string['f20'],
        # keyboard.Key.f21:               string['f21'],
        # keyboard.Key.f22:               string['f22'],
        # keyboard.Key.f23:               string['f23'],
        # keyboard.Key.f24:               string['f24'],
        # keyboard.Key.f25:               string['f25'],
        # keyboard.Key.f26:               string['f26'],
        # keyboard.Key.f27:               string['f27'],
        # keyboard.Key.f28:               string['f28'],
        # keyboard.Key.f29:               string['f29'],
        # keyboard.Key.f30:               string['f30'],
        keyboard.Key.home:              string['home'],
        keyboard.Key.insert:            string['insert'],
        keyboard.Key.left:              string['left'],
        keyboard.Key.media_next:        string['next'],
        keyboard.Key.media_play_pause:  string['play'],
        keyboard.Key.media_previous:    string['previous'],
        keyboard.Key.media_volume_down: string['volumeDown'],
        keyboard.Key.media_volume_mute: string['mute'],
        keyboard.Key.media_volume_up:   string['volumeUp'],
        keyboard.Key.menu:              string['menu'],
        keyboard.Key.num_lock:          string['numLock'],
        keyboard.Key.page_down:         string['pageDown'],
        keyboard.Key.page_up:           string['pageUp'],
        keyboard.Key.pause:             string['pause'],
        keyboard.Key.print_screen:      string['printScreen'],
        keyboard.Key.right:             string['right'],
        keyboard.Key.scroll_lock:       string['scrollLock'],
        keyboard.Key.shift:             string['shift'],
        keyboard.Key.shift_l:           string['shiftLeft'],
        keyboard.Key.shift_r:           string['shiftRight'],
        keyboard.Key.space:             string['space'],
        keyboard.Key.tab:               string['tab'],
        keyboard.Key.up:                string['up'],
    })

    pyautogui = {} # Just uses strings

    pygame = ({ } if not checkImport('pygame', fatal=False, printWarning=False) else {

    })

    standards = (ascii, string, pynput, pyautogui, pygame)

class Key:
    """ A generalized Key class to bridge the gap between several standards.
        Mouse buttons are not included. Neither are controllers.
    """
    useRightLeft=False

    @staticmethod
    def dropRightLeft(key: "Key"):
        # We don't want to call __eq__() cause we use this function from there
        if key._key in (Key('rightCtrl')._key, Key('leftCtrl')._key):
            return Key('ctrl')
        elif key._key in (Key('rightShift')._key, Key('leftShift')._key):
            return Key('shift')
        elif key._key in (Key('rightAlt')._key, Key('leftAlt')._key):
            return Key('alt')
        elif key._key in (Key('rightWin')._key, Key('leftWin')._key):
            return Key('win')
        else:
            return key

    def __init__(self, key, standard=KeyStandard.string):
        if type(key) is type(self):
            self._param = key
            self.standard = key.standard
            self._key = key._key
            return
        elif key is None:
            self._param = None
            self.standard = standard
            self._key = 0
            return

        self._param = key
        self.standard = standard
        self._key = None

        if type(key) is int:
            if isBetween(key, 1, 9):
                self._key = key + 48
            elif key == 0:
                self._key = 57
        else:
            # Their error is better than mine anyway
            try:
                self._key = self.standard[key]
            except KeyError as err:
                for standard in KeyStandard.standards:
                    if key in standard.keys():
                        self._key = standard[key]
                # See if it's cased weird before we give up
                try:
                    self._key = self.standard[key.lower()]
                except: pass
                # If it's still not set, which means it's not in any standard
                if self._key is None:
                    raise err
                    # raise KeyError(f"{key} is not a valid key for the given standard (It might not be implemented yet)")

    def __eq__(self, other):
        try:
            a = self  if self.useRightLeft else self.dropRightLeft(self)
            b = other if self.useRightLeft else self.dropRightLeft(Key(other))
        except:
            raise KeyError(f"Cannot compare types of {type(other).__name__} and Key")

        # if type(other) is Key:
        return a._key == b._key

        # This *should* be replaced by just calling other with the Key constructor above
        # for standard in KeyStandard.standards:
        #     if other in standard.keys():
        #         return debug(a._key == standard[b], clr=3)


    def __hash__(self):
        return hash(self._key)

    # # This doesn't work, I don't know why.
    def __getattr__(self, key):
        debug()
        return Key(key)

    def __str__(self):
        return invertDict(KeyStandard.ascii)[self._key]

    @todo("Figure out prototyping (look here: https://www.geeksforgeeks.org/prototype-method-python-design-patterns/)")
    def __add__(self, other):
        if type(other) is Key:
            return KeyShortcut(self, other)
        elif type(other) is KeyShortcut:
            other.keys += (self,)
        else:
            raise TypeError(f"Can only add Keys (not {type(other).__name__}) together to make KeyShortcuts")

# TODO add an entered trigger and exited trigger for on pressed vs on released
class KeyShortcut:
    """ Triggers when all the specified keys are being held at the same time """
    def __init__(self, *keys:Key, useRightLeft=False):
        self.useRightLeft = useRightLeft
        self.triggered = Signal()
        self.keys = keys
        self.heldKeys = set()
        if Key('shift') in self.keys:
            debug("Shift (just shift) doesn't work with KeyShortcuts at the moment. No clue why. Please fix my code or use a different key.", clr=RED)

    def update(self, key, pressed):
        key = Key(key) if self.useRightLeft else Key.dropRightLeft(Key(key))

        debug(key, 'shortcut update key', clr=2)

        # debug(self.heldKeys, clr=3)
        if pressed:
            self.heldKeys.add(key)
        else:
            # if key == Key('shift'):
                # self.heldKeys.clear()
            # else:
            try:
                self.heldKeys.remove(key)
            except KeyError:
                self.heldKeys.clear()


        # debug(self.heldKeys, clr=4)

        valid = True
        for i in self.keys:
            if i not in self.heldKeys:
                valid = False
                break

        if valid:
            debug("Shortcut Triggered!")
            self.triggered.call()

@todo
class KeyChord:
    pass

class KeySequence:
    triggered = Signal()
    def __init__(self, *keySequence):
        self.sequence = keySequence
        self.activeMods = dict(zip(modifierKeys, (False,) * len(modifierKeys)))
        self.currentSequence = []

    def update(self, key, pressed):
        todo('not finished', blocking=True)
        if key in modifierKeys:
           self.activeMods[key] = pressed

        self.currentSequence.append(key)

        valid = True
        for i in self.keys:
            if i in modifierKeys:
                if not self.activeMods[i]:
                    valid = False
            else:
                if key != i:
                    valid = False

        if valid:
            triggered.call()
        else:
            self.currentSequence = []

# TODO add the __roperator__ functions
class MappingList(list):
    """ An iterable that functions exactly like a list, except any operators applied to it
        are applied equally to each of it's memebers, and return a mapping list instance.
    """
    unmatchedLenError = TypeError('Cannot evaluate 2 MappingLists of differing length')
    def __init__(self, *args):
        super().__init__(ensureIterable(ensureNotIterable(args)))

    def apply(self, func:Union[Callable, 'MappingList'], *args, **kwargs):
        """ Call a function with parameters on each item """
        if type(func) is MappingList and len(func) == len(self):
            self = MappingList([i(k, *args, **kwargs) for i, k in (func, self)])
        else:
            for i in range(len(self)):
                self[i] = func(self[i], *args, **kwargs)
        return self

    def call(self, func:Union[str, 'MappingList'], *args, **kwargs):
        """ Call a member function with parameters on each item """
        if type(func) is MappingList and len(func) == len(self):
            self = MappingList([i.__getattribute__(k)(*args, **kwargs) for i, k in (func, self)])
        else:
            for i in range(len(self)):
                self[i] = self[i].__getattribute__(func)(*args, **kwargs)
        return self

    def attr(self, attr:Union[str, 'MappingList']):
        """ Replace each item with it's attribute """
        if type(attr) is MappingList and len(attr) == len(self):
            self = MappingList([i.__getattribute__(k) for i, k in (attr, self)])
        else:
            for i in range(len(self)):
                self[i] = self[i].__getattribute__(attr)
        return self

    def lengths(self):
        return ensureNotIterable(MappingList([len(i) for i in self]))

    def __getattr__(self, name):
        self = MappingList([i.__getattribute__(name) for i in self])
        return self

    def __call__(self, *args, **kwargs):
        for i in range(len(self)):
            self[i] = self[i](*args, **kwargs)
        return self

    def __hash__(self):
        return hash(tuple(self))

    def __cmp__(self, other):
        if type(other) is MappingList:
            if len(other) == len(self):
                return MappingList([i.__cmp__(k) for i, k in (other, self)])
            else:
                raise MappingList.unmatchedLenError
        else:
            return super().__cmp__(other)

    def __pos__(self):
        for i in range(len(self)):
            self[i] = +self[i]
        return self

    def __neg__(self):
        for i in range(len(self)):
            self[i] = -self[i]
        return self

    def __abs__(self):
        for i in range(len(self)):
            self[i] = abs(self[i])
        return self

    def __invert__(self):
        for i in range(len(self)):
            self[i] = ~self[i]
        return self

    def __round__(self, n):
        for i in range(len(self)):
            self[i] = round(self[i], n)
        return self

    def __floor__(self):
        for i in range(len(self)):
            self[i] = self[i].__floor__()
        return self

    def __ceil__(self):
        for i in range(len(self)):
            self[i] = self[i].__ceil__()
        return self

    def __trunc__(self):
        for i in range(len(self)):
            self[i] = self[i].__trunc__()
        return self

    def __add__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.__add__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__add__(other) for i in self])

    def __sub__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.__sub__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__sub__(other) for i in self])

    def __mul__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.__mul__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__mul__(other) for i in self])

    def __floordiv__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.____(__floordiv__) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__floordiv__(other) for i in self])

    def __truediv__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.____(__truediv__) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__truediv__(other) for i in self])

    def __mod__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.__mod__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__mod__(other) for i in self])

    def __divmod__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.__divmod__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__divmod__(other) for i in self])

    def __pow__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.__pow__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__pow__(other) for i in self])

    def __lshift__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.__lshift__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__lshift__(other) for i in self])

    def __rshift__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.__rshift__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__rshift__(other) for i in self])

    def __and__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.__and__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__and__(other) for i in self])

    def __or__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.__or__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__or__(other) for i in self])

    def __xor__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            return MappingList(*[i.__xor__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            return MappingList(*[i.__xor__(other) for i in self])

    def __iadd__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__add__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__add__(other) for i in self])
        return self

    def __isub__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__sub__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__sub__(other) for i in self])
        return self

    def __imul__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__mul__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__mul__(other) for i in self])
        return self

    def __ifloordiv__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__floordiv__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__floordiv__(other) for i in self])
        return self

    def __itruediv__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__truediv__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__truediv__(other) for i in self])
        return self

    def __imod__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__mod__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__mod__(other) for i in self])
        return self

    def __idivmod__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__divmod__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__divmod__(other) for i in self])
        return self

    def __ipow__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__pow__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__pow__(other) for i in self])
        return self

    def __ilshift__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__lshift__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__lshift__(other) for i in self])
        return self

    def __irshift__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__rshift__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__rshift__(other) for i in self])
        return self

    def __iand__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__and__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__and__(other) for i in self])
        return self

    def __ior__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__or__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__or__(other) for i in self])
        return self

    def __ixor__(self, other):
        if type(other) is MappingList and len(other) == len(self):
            self = MappingList(*[i.__xor__(k) for i, k in zip(other, self)])
        elif type(other) is MappingList:
            raise MappingList.unmatchedLenError
        else:
            self = MappingList(*[i.__xor__(other) for i in self])
        return self

    def __int__(self):
        for i in range(len(self)):
            self[i] = self[i].__int__()
        return self

    def __long__(self):
        for i in range(len(self)):
            self[i] = self[i].__long__()
        return self

    def __float__(self):
        for i in range(len(self)):
            self[i] = self[i].__float__()
        return self

    def __complex__(self):
        for i in range(len(self)):
            self[i] = self[i].__complex__()
        return self

    def __oct__(self):
        for i in range(len(self)):
            self[i] = self[i].__oct__()
        return self

    def __hex__(self):
        for i in range(len(self)):
            self[i] = self[i].__hex__()
        return self

    def __index__(self):
        for i in range(len(self)):
            self[i] = self[i].__index__()
        return self

    def __trunc__(self):
        for i in range(len(self)):
            self[i] = self[i].__trunc__()
        return self

    def __coerce__(self, other):
        for i in range(len(self)):
            self[i] = self[i].__coerce__(other)
        return self


################################### Misc. Useful Functions ###################################
struct = namedtuple # I'm used to C
@todo('Make this use piping and return the command output', False)
def runCmd(args):
    """ Run a command and terminate if it fails. """
    try:
        ec = subprocess.call(' '.join(args), shell=True)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)
        ec = 1
    if ec:
        sys.exit(ec)

def percent(percentage):
    ''' Usage:
        if (percent(50)):
            <code that has a 50% chance of running>
    '''
    return randint(1, 100) < percentage

def randbool():
    """ Returns, randomly, either True or False """
    return bool(randint(0, 1))

def closeEnough(a, b, tolerance):
    """ Returns True if a is within tolerance range of b """
    return a <= b + tolerance and a >= b - tolerance

def findClosestValue(target, comparatorList) -> "value":
    """ Finds the value in comparatorList that is closest to target """
    # dist = max_distance
    # value = None
    # index = 0
    # for cnt, current in enumerate(comparatorList):
    #     currentDist = abs(target - current)
    #     if currentDist < dist:
    #         dist = currentDist
    #         value = current
    #         index = cnt
    # return (value, index)
    return min(comparatorList, key=lambda x: abs(target - x))

def findFurthestValue(target, comparatorList) -> "value":
    """ Finds the value in comparatorList that is furthest from target """
    return max(comparatorList, key=lambda x: abs(target - x))

def absdeg(angle):
    """ If an angle (in degrees) is not within 360, then this cuts it down to within 0-360 """
    angle = angle % 360.0
    if angle < 0:
        angle += 360
    return angle

def absrad(angle):
    """ If an angle (in radians) is not within 2Pi, then this cuts it down to within 0-2Pi """
    angle = angle % math.tau
    if angle < 0:
        angle += math.tau
    return angle

def center(string):
    """ Centers a string for printing in the terminal """
    from os import get_terminal_size
    for _ in range(int((get_terminal_size().columns - len(string)) / 2)): string = ' ' + string
    return string

def isPowerOf2(x):
    """ Returns true if x is a power of 2 """
    return (x != 0) and ((x & (x - 1)) == 0)

def isBetween(val, start, end, beginInclusive=False, endInclusive=False):
    """ Returns true if val is between start and end """
    return (val >= start if beginInclusive else val > start) and \
           (val <= end   if endInclusive   else val < end)

def insertChar(string, index, char):
    """ Returns the string with char inserted into string at index. Freaking python string are immutable. """
    return string[:index] + char + string[index+1:]

def constrain(val, low, high):
    """ Constrains val to be within low and high """
    return min(high, max(low, val))

def translate(value, fromStart, fromEnd, toStart, toEnd):
    return ((abs(value - fromStart) / abs(fromEnd - fromStart)) * abs(toEnd - toStart)) + toStart

def frange(start, stop, skip=1.0, accuracy=10000000000000000):
    return [x / accuracy for x in range(int(start*accuracy), int(stop*accuracy), int(skip*accuracy))]

def getDist(ax, ay, bx, by):
    return math.sqrt(((bx - ax)**2) + ((by - ay)**2))

def deg2rad(a, symbolic=False):
    if symbolic:
        if ensureImported('sympy', 'pi', fatal=True):
            return (a * pi / 180).simplify()
    else:
        return a * PI / 180.0

def rad2deg(a, symbolic=False):
    if symbolic:
        if ensureImported('sympy', 'pi', fatal=True):
            return (a * 180 / pi).simplify()
    else:
        return a * 180.0 / PI

def normalize2rad(a):
    while a < 0: a += math.tau
    while a >= math.tau: a -= math.tau
    return a

def normalize2deg(a):
    while a < 0: a += 360
    while a >= 360: a -= 360
    return a

def portFilename(filename):
    return join(*filename.split('/'))

def assertValue(param, *values, blocking=True):
    paramName = _debugGetVarName(param)
    if not _debugBeingUsedAsDecorator('assertValue'):
        if param not in values:
            err = TypeError(f"Invalid value for {paramName}, must be one of: {values}")
            if blocking:
                raise err
            else:
                debug(err)
    else:
        todo('usage as a decorator')

@confidence(72)
def replaceLine(line, offset=0, keepTabs=True, convertTabs=True, additionalCalls=0):
    """ Replaces the line of code this is called from with the give line parameter.
        This is probably a very bad idea to actually use.
        Don't forget to add tabs! Newline is already taken care of (unless you want to add more).
    """
    meta = _debugGetMetaData(calls=2 + additionalCalls)

    with open(meta.filename, 'r') as f:
        file = f.readlines()

    # Not really sure of the reason for the -1.
    if file[meta.lineno-1] == meta.code_context[0]:
        if keepTabs:
            tabs = rematch(file[meta.lineno-1 + offset], r'\s+')
            if tabs:
                line = tabs.string + line

        if convertTabs:
            line = line.replace('\t', '    ')

        file[meta.lineno-1 + offset] = line + '\n'

    else:
        debug(f"Error: lines don't match, not replacing line.\n\tMetadata: \"{meta.code_context}\"\n\tFile: \"{file[meta.lineno-1]}\"", clr=Colors.ERROR)
        return

    with open(meta.filename, 'w') as f:
        f.writelines(file)

@confidence(85)
def fancyComment(title='', char='#', endChar='#', lineLimit=80):
    """ Replaces the call with a nicely formatted comment line """
    halfLen = ((lineLimit / len(char)) - len(title) - 1 - (2 if len(title) else 0) - len(endChar)) / 2
    seperateChar = ' ' if len(title) else ''
    replaceLine('#' + (char * ceil(halfLen)) + seperateChar + title.title() + seperateChar + (char * floor(halfLen)) + endChar, keepTabs=False, additionalCalls=1)

# No, English does not make any sense.
def umpteenthName(i:int) -> "1st, 2nd, 3rd, etc.":
    i = str(i)
    if i[-1] == '1' and (i != '11'):
        return i + 'st'
    elif i[-1] == '2' and (i[0] != '1'):
        return i + 'nd'
    elif i[-1] == '3' and (i[0] != '1'):
        return i + 'rd'
    else:
        return i + 'th'


################################### API Specific functions ###################################
#* PrettyTable
@dependsOnPackage('prettytable', 'PrettyTable')
def quickTable(listOfLists, interpretAsRows=True, fieldNames=None, returnStr=False, sortByField:str=False, sortedReverse=False):
    """ A small, quick wrapper for the prettytable library """
    t = PrettyTable()
    if interpretAsRows:
        if fieldNames is not None:
            t.field_names = fieldNames
        t.add_rows(listOfLists)
    else:
        if fieldNames is not None:
            for i, name in zip(listOfLists, fieldNames):
                t.add_column(name, i)
        else:
            for i in listOfLists:
                t.add_column(str(i[0]), i[1:])

    if sortByField:
        t.sortby = sortByField
        t.reversesort = sortedReverse

    return t.get_string() if returnStr else t

#* My Own Point Classes
@dependsOnPackage('Point', 'Point')
def findClosestXPoint(target, comparatorList, offsetIndex = 0):
    """ I've forgotten what *exactly* this does. I think it finds the point in a list of
        points who's x point is closest to the target
    """
    finalDist = 1000000
    result = 0

    # for i in range(len(comparatorList) - offsetIndex):
    for current in comparatorList:
        # current = comparatorList[i + offsetIndex]
        currentDist = abs(target.x - current.x)
        if currentDist < finalDist:
            result = current
            finalDist = currentDist

    return result

@dependsOnPackage('Point', ('Point', 'Pointi', 'Pointf'))
def getPointsAlongLine(p1, p2):
    """ I don't remember what this does. """
    p1 = Pointi(p1)
    p2 = Pointi(p2)

    returnMe = []

    dx = p2.x - p1.x
    dy = p2.y - p1.y

    for x in range(p1.x, p2.x):
        y = p1.y + dy * (x - p1.x) / dx
        returnMe.append(Pointf(x, y))

    return returnMe

@dependsOnPackage('Point', 'Point')
def rotatePoint(p, angle, pivotPoint, radians = False):
    """ This rotates one point around another point a certain amount, and returns it's new position """
    if not radians:
        angle = math.radians(angle)
    # p -= pivotPoint
    # tmp = pygame.math.Vector2(p.data()).normalize().rotate(amount)
    # return Pointf(tmp.x, tmp.y) + pivotPoint

    dx = p.x - pivotPoint.x
    dy = p.y - pivotPoint.y
    newX = dx * math.cos(angle) - dy * math.sin(angle) + pivotPoint.x
    newY = dx * math.sin(angle) + dy * math.cos(angle) + pivotPoint.y

    return Pointf(newX, newY)

@dependsOnPackage('Point', 'Point')
def getMidPoint(p1, p2):
    """ Returns the halfway point between 2 given points """
    assert type(p1) == type(p2)
    # return Pointf((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
    return p1._initCopy((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

@dependsOnPackage('Point', 'Point')
def findClosestPoint(target, comparatorList):
    """ Finds the closest point in the list to what it's given"""
    finalDist = 1000000

    for i in comparatorList:
        current = getDist(target, i)
        if current < finalDist:
            finalDist = current

    return finalDist

@dependsOnPackage('Point', 'Point')
def collidePoint(topLeft: 'Point', size: Union[tuple, list, 'Size'], target, inclusive=True):
    """ Returns true if target is within the rectangle given by topLeft and size """
    return isBetween(target.x, topLeft.x, size[0], beginInclusive=inclusive, endInclusive=inclusive) and \
           isBetween(target.y, topLeft.y, size[1], beginInclusive=inclusive, endInclusive=inclusive)

@dependsOnPackage('Point', 'Point')
def getPointDist(a: 'Point', b: 'Point'):
    return math.sqrt(((b.x - a.x)**2) + ((b.y - a.y)**2))

#* Pygame
@dependsOnPackage('pygame')
def loadImage(filename):
    # if pygame.image.get_extended():
    filename = '/' + portableFilename(DATA + '/' + filename)

    image = pygame.image.load(filename)
    # self.image = self.image.convert()
    image = image.convert_alpha()
    # else:
    #     assert(not f"Cannot support the file extension {}")
    return image

def loadAsset(dir, name, extension='png'):
    return loadImage(dir + name + '.' + extension)

@dependsOnPackage('pygame')
def rotateSurface(surface, angle, pivot, offset):
    """Rotate the surface around the pivot point.

    Args:
        surface (pygame.Surface): The surface that is to be rotated.
        angle (float): Rotate by this angle.
        pivot (tuple, list, pygame.math.Vector2): The pivot point.
        offset (pygame.math.Vector2): This vector is added to the pivot.
    """
    rotated_image = pygame.transform.rotozoom(surface, -angle, 1)  # Rotate the image.
    rotated_offset = offset.rotate(angle)  # Rotate the offset vector.
    # Add the offset vector to the center/pivot point to shift the rect.
    rect = rotated_image.get_rect(center=pivot+rotated_offset)
    return rotated_image, rect  # Return the rotated image and shifted rect.


################################### Old Functions I Don't Want to Delete ###################################
# I don't remember what this does and I'm scared to delete it
@confidence(45)
@dependsOnPackage('tkinter', _as='tk')
@dependsOnPackage('tkinter.ttk', _as='ttk')
@dependsOnPackage('contextlib', 'redirect_stdout')
@dependsOnPackage('ttkthemes')
def stylenameElementOptions(stylename):
    '''Function to expose the options of every element associated to a widget
       stylename.'''
    with open('tmp.del', 'a') as f:
        with redirect_stdout(f):
            print('\n-----------------------------------------------------------------------------\n')
            try:
                # Get widget elements
                style = ttk.Style()
                layout = str(style.layout(stylename))
                print('Stylename = {}'.format(stylename))
                print('Layout    = {}'.format(layout))
                elements=[]
                for n, x in enumerate(layout):
                    if x=='(':
                        element=""
                        for y in layout[n+2:]:
                            if y != ',':
                                element=element+str(y)
                            else:
                                elements.append(element[:-1])
                                break
                print('\nElement(s) = {}\n'.format(elements))
                # Get options of widget elements
                for element in elements:
                    print('{0:30} options: {1}'.format(
                        element, style.element_options(element)))
            except tk.TclError:
                print('_tkinter.TclError: "{0}" in function'
                    'widget_elements_options({0}) is not a regonised stylename.'
                    .format(stylename))

    # for i in ['TButton', 'TCheckbutton', 'TCombobox', 'TEntry', 'TFrame', 'TLabel', 'TLabelFrame', 'TMenubutton', 'TNotebook', 'TPanedwindow', 'Horizontal.TProgressbar', 'Vertical.TProgressbar', 'TRadiobutton', 'Horizontal.TScale', 'Vertical.TScale', 'Horizontal.TScrollbar', 'Vertical.TScrollbar', 'TSeparator', 'TSizegrip', 'Treeview', 'TSpinbox']:
    #     stylenameElementOptions('test.' + i)

    # stylenameElementOptions('me.TButton')

@depricated
def ref(obj):
    return pointer(py_object(obj))

@depricated
def deref(ptr):
    return ptr.contents.value


################################### Notes ###################################
""" DECORATOR SYNTAX:

def decorator(*decoratorArgs, **decoratorKwArgs):
    def wrap(functionBeingDecorated):
        def innerWrap(*decoratedArgs, **decoratedKwArgs):
            return functionBeingDecorated(*decoratedArgs, **decoratedKwArgs)
        return innerWrap
    return wrap

COPY version:

def decorator(*decoratorArgs, **decoratorKwArgs):
    def wrap(func):
        def innerWrap(*funcArgs, **funcKwArgs):
            return func(*funcArgs, **funcKwArgs)
        return innerWrap
    return wrap
"""

################################### Tests ###################################
if ENABLE_TESTING:
    displayAllPaths()
    #* parseColorParams tests
    if False:
        # setVerbose(True)
        # debug(parseColorParams((5, 5, 5)) )
        # debug(parseColorParams((5, 5, 5), True) )
        # debug(parseColorParams((5, 5, 5, 6)) )
        # debug(parseColorParams((5, 5, 5, 6), True) )
        # debug(parseColorParams([5, 5, 5, 6]) )
        # debug(parseColorParams(5, 5, 5) )
        # debug(parseColorParams(5, 5, 5, True) )
        # debug(parseColorParams(5, 5, 5, 6) )
        # debug(parseColorParams(5, 5, 5, bg=True) )
        # debug(parseColorParams(5, 5, 5, 6, True) )
        # debug(parseColorParams(3) )
        # debug(parseColorParams(3, bg=True)
        # debug(parseColorParams((3,)) ) # Succeeded
        # debug(parseColorParams(3, a=6) )
        # debug(parseColorParams(3, a=6, bg=True) )
        # debug(parseColorParams(None) )
        # debug(parseColorParams(None, bg=True) )
        pass

    #* debug tests
    if False:
        # setVerbose(True)
        """
        a = 6
        s = 'test'
        j = None
        def testFunc():
            print('testFunc called')

        debug(a)
        debug(a, 'apple')

        debug('test3')
        debug(s)

        debug(j)
        debug()

        debug(testFunc)

        foo = debug(a)
        debug(foo)

        debug(parseColorParams((5, 5, 5)) )

        debug(SyntaxError('Not an error'))
        try:
            debug(SyntaxError('Not an error'), raiseError=True)
        except SyntaxError:
            print('SyntaxError debug test passed!')
        else:
            print('SyntaxError debug test failed.')

        debug(UserWarning('Not a warning'))
        try:
            debug(UserWarning('Not a warning'), raiseError=True)
        except UserWarning:
            print('UserWarning debug test passed!')
        else:
            print('UserWarning debug test failed.')

        @debug
        def testFunc2():
            print('testFunc2 (decorator test) called')

        debug()

        testFunc2()

        debug(None)
        """
        TUPLE = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        LIST  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        DICT  = {'a':1, 'b':2, 'c': 3}
        TYPE_LIST = ['a', 2, 7.4, 3]
        TYPE_TUPLE = ('a', 2, 7.4, 3)

        debug([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], raiseError=True)
        debug((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), raiseError=True)
        debug({'a':1, 'b':2, 'c': 3}, raiseError=True)
        debug(['a', 2, 7.4, 3], raiseError=True)
        debug(('a', 2, 7.4, 3), raiseError=True)
        debug()
        debug(TUPLE, raiseError=True)
        debug(LIST, raiseError=True)
        debug(DICT, raiseError=True)
        debug(TYPE_LIST, raiseError=True)
        debug(TYPE_TUPLE, raiseError=True)

        debug(())
        debug([])
        debug({})
        debug(set())

    #* todo tests
    if False:
        todo('testing todo')
        todo('testing todo 2', False)

        @todo
        def unfinishedFunc():
            print("this func is unfin")

        try:
            unfinishedFunc()
        except NotImplementedError:
            print("func decorator test worked!")
        else:
            print("func decorator test failed.")

        @todo(blocking=False)
        def unfinishedFunc2():
            print("this non Blocking func is unfin")

        unfinishedFunc2()

        @todo
        class unfinishedClass:
            def __init__(self):
                print('this class is unfin')

        try:
            x = unfinishedClass()
        except NotImplementedError:
            print("class decorator test worked!")
        else:
            print("class decorator test failed.")

    #* Decorator Testing
    if False:
        def decorator(*decoratorArgs, **decoratorKwArgs):
            def wrap(functionBeingDecorated):
                def innerWrap(*decoratedArgs, **decoratedKwArgs):
                    debug(decoratorArgs)
                    debug(decoratorKwArgs)
                    debug(functionBeingDecorated)
                    debug(decoratedArgs)
                    debug(decoratedKwArgs)
                    return functionBeingDecorated(*decoratedArgs, **decoratedKwArgs)
                return innerWrap
            return wrap

        @decorator("decoratorArg1", "decoratorArg2", decoratorKwArg="decoratorKwValue")
        def testFunc(funcArg1, funcArg2, funcKwArg='funcKwArg'):
            debug(funcArg1)
            debug(funcArg2)
            debug(funcKwArg)

        testFunc("calledArg1", 'calledArg2', funcKwArg='calledKwArg')

    #* Confidence Testing
    if False:
        @confidence(29)
        def testFunc(funcArg1, funcArg2, funcKwArg='funcKwArg'):
            debug(funcArg1)
            debug(funcArg2)
            debug(funcKwArg)

        @confident(102)
        def testFunc2(funcArg1, funcArg2, funcKwArg='funcKwArg'):
            debug(funcArg1)
            debug(funcArg2)
            debug(funcKwArg)

        @confident(16)
        def testFunc3(funcArg1, funcArg2, funcKwArg='funcKwArg'):
            debug(funcArg1)
            debug(funcArg2)
            debug(funcKwArg)

        @confidence('super')
        def testFunc4(): pass

        @confidence('not very')
        def testFunc5(): pass

        @confidence('none')
        def testFunc6(): pass

        @confidence('low')
        def testFunc7(): pass

        @confidence('Sorta')
        def testFunc8(): pass

        @confidence('asfgs')
        def testFunc9(): pass

        @confidence('confident', 100)
        def testFunc10(): pass

        @confidence('not confident', 0)
        def testFunc11(): pass

        try:
            testFunc2("calledArg1", 'calledArg2', funcKwArg='calledKwArg')
        except TypeError:
            print('testFunc2 worked')

        testFunc( "calledArg1", 'calledArg2', funcKwArg='calledKwArg')
        testFunc3("calledArg1", 'calledArg2', funcKwArg='calledKwArg')
        testFunc4()
        testFunc5()
        testFunc6()
        testFunc7()
        testFunc8()
        testFunc10()
        testFunc11()
        try:
            testFunc9()
        except TypeError:
            print('testFunc9 worked')

    #* Mapping list tests
    if False:
        debug(MappingList())
        debug(MappingList(1, 2, 3))
        debug(MappingList((1, 2, 3)))
        debug(MappingList([1, 2, 3]))
        m = MappingList(1, 2, 3)
        debug(m)
        debug(m+3)
        debug(m-3)
        m += 4
        debug(m)

        m = MappingList('hello', 'world')
        debug(m)
        m += '!'
        debug(m)
        try:
            debug(m / 4)
        except AttributeError:
            print('First Error test worked')
        else:
            print('First Error test failed')

        debug(MappingList(1, 2, 3) * MappingList(2, 2, 4))
        # try:
            # debug(MappingList(1, 2, 3) * MappingList(2, 2))
        # except TypeError:
            # print('Second error test worked')
        # else:
            # print('Second error test failed')
        # debug(MappingList(1, 2, 3) * MappingList(2))
        debug(MappingList(1, 2, 3) * 2)

        t = MappingList('testing')
        debug(t)
        debug(t+' success')
        debug(t.istitle)
        debug(t.replace('t', '|'))
        debug(t.upper())
        t += ' tests'
        debug(t)

    #* Replace Line tests
    if False:

        # replaceLine("\t\t# This Line has been replaced! 1", -1)
        # replaceLine("\t\t# This Line has been replaced! 2")
        # replaceLine("# This Line has been replaced! 3")

        # replaceLine("\t\t# This Line has been replaced! 1", -1)
        # replaceLine("\t\t# This Line has been replaced! 2")
        # replaceLine("# This Line has been replaced! 3")

        fancyComment()
        fancyComment(char='~')
        fancyComment(lineLimit=30)
        fancyComment('Seperator!')
        fancyComment('Seperator!', '~')
        fancyComment('Seperator!', '~', '{')
        fancyComment('Seperator!', '~', '{', 50)

        # fancyComment()
        # fancyComment(char='~')
        # fancyComment(lineLimit=30)
        # fancyComment('Seperator!')
        # fancyComment('Seperator!', '~')
        # fancyComment('Seperator!', '~', '{')
        # fancyComment('Seperator!', '~', '{', 50)

    #* Key testing
    if False:
        debug(Key('one'))
        debug(Key('exclamation'))
        # debug(Key('exclamationmark'))
        debug(Key('exclamationPoint'))
        # These SHOULD work. __getattr__ is a fickle beast.
        # debug(Key.one)
        # debug(Key.s)
        # debug(Key.ctrl)

        from pynput import keyboard

        def on_press(key):
            debug(key)

            try:
                key = key.char
            except AttributeError:
                pass

            debug(key)
            debug(Key(key))
            debug(Key(key) == key)

        def on_release(key):
            try:
                key = key.char
            except AttributeError:
                pass
            if key == keyboard.Key.esc:
                return False

        # Collect events until released
        with keyboard.Listener(
                on_press=on_press,
                on_release=on_release) as listener:
            listener.join()
