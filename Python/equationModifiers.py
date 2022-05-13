from collections import namedtuple

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
peak = UnitModifier(
    lambda d: 'peak' + d,
    lambda abbrev: (),
    lambda abbrev: abbrev + '_m'
)