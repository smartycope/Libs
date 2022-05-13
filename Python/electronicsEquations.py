from Equation import Equation, Unit
from namespaces import electronics
from include import series, parallel

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
invertingOpAmpGain    = Equation("Av == -(Rf / Rin)", electronics.replace(
    Unit('Av', 'Voltage Gain'),
    Unit('Rf', 'the resistor connecting the op amp Vout and the op amp negative in terminals'),
    Unit('Rin', 'the resistor between Vin and the op amp negative in'),
    Unit('Vout', 'the output voltage of the op amp'),
    Unit('Vin', 'voltage connected to Rin'),
))
noninvertingOpAmpGain = Equation('Av == 1 + (Rf / R)', electronics.replace(
    Unit('Av', 'Voltage Gain'),
    Unit('Rf', 'the resistor connecting the op amp Vout and the op amp negative in terminals'),
    Unit('R', 'the resistor between ground and the op amp negative in'),
    Unit('Vout', 'the output voltage of the op amp'),
    Unit('Vin', 'voltage connected to the op amp positive in'),
))

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
someImportantRCEquation = Equation('x_t == x_f+ (x_i - x_f)*_e**(-t/tc)', electronics)

# i_L(t) == I_0*e*(-t/tc)
RLDischargeCurrent = Equation('i_t == ii * _e**(-t/tc)', electronics)
RLTimeConstant = Equation('tc == L/Req', electronics)
# RLChargeCurrent = Equation('i_t == (vs/r) + ((ii - (vs/r)) * _e**(-t/tc))', electronics)
# RLChargeCurrent = Equation('v_t == vs * _e**(-t/tc)', electronics)
# RLChargeCurrent = Equation('v_t == vs * _e**(-t/tc)', electronics)


# inductorCurrent = iL(t) == (1/L) * Integral(high=t, low=0, vL(dummy)ddummy) + i_0
#? Vp = -Np * Derivative(phi, t)
# inductorEnergy = w=integral(high=t, low=t_0, p(t))
#? capacitorVoltage ( vc(t) = 1/C * integrral(t, 0, ???))
#? w = L*integral(i(t), low=0, ???)
# watts == joules / second
# Coulomb = amps / second

#* AC Current
# ACPowerLoss = Equation*('P_loss == i**2 * R_loss', electronics)
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
seriesRLImpedance = Equation('Z_eq_RL == R+_i*X_L', electronics)
inductorReactance = Equation('X_L == ohmega*L', electronics)

#? X_L=delta(ohmega*L)

# For capacitors
capacitorImpedance = Equation('phZ_C == phV_C/phi_C', electronics)
capacitorImpedance2 = Equation('phZ_C == 1/(_i*ohmega*C)', electronics)
capacitorImpedance3 = Equation('phZ_C == -_i*X_C', electronics)
seriesRCImpedance = Equation('Z_eq_RC == R-_i*X_C', electronics)
capacitorReactance = Equation('X_C == 1/(ohmega*C)', electronics)
capacitorReactance2 = Equation('X_C == V_m/I_m', electronics)
# Equation('X_C == 1/(2*pi*f*C)', electronics)(f=5*k, )

resistorImpedance = Equation('phZ == r', electronics)
inductorImpedance = Equation('phZ == _i*X', electronics)
inductorImpedance2 = Equation('phZ == _i*ohmega*L', electronics)
capacitorImpedance4 = Equation('phZ == -_i*X', electronics)
capacitorImpedance5 = Equation('phZ == 1/(_i*ohmega*C)', electronics)

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
instantPowerSine = Equation('p_t == (1/2) * v_m*i_m * cos(dph) + (1/2)*v_m*i_m * cos(2*ohmega*t + theta_v + theta_i)', electronics)
instantVoltage = Equation('v_t == v_m * cos(ohmega*t + theta_v)', electronics)
instantCurrent = Equation('i_t == i_m * cos(ohmega*t + theta_i)', electronics)

# The average power, in watts, is the average of the instantaneous power over one period.
averagePower = Equation('p == (1/T) * Integral(p_t, (t, 0, T))', electronics)
# Integral(val, (var, low, high))
averagePowerExpanded = Equation('p == (1/T) * Integral((1/2) * v_m*i_m * cos(dph), (t, 0, T)) + (1/T) * Integral((1/2)*v_m*i_m * cos(2*ohmega*t + theta_v + theta_i), (t, 0, T))', electronics)
averagePower2 = Equation('p == (1/2) * v_m * i_m * cos(dph)', electronics, defaults={'theta_i':0, 'theta_v':0})
averagePowerPhasors = Equation('p == (1/2) * re(phV * phI)', electronics)

pureResistivePower = Equation('p == (1/2) * i_m**2 * r', electronics)
pureReactivePower = Equation('p == 0', electronics)

rms = Equation('i_eff == i_rms', electronics)
rmsCurrent = Equation('i_eff == sqrt((1/T) * Integral(i_t**2, (t, 0, T)))', electronics)
rmsVoltage = Equation('i_eff == sqrt((1/T) * Integral(v_t**2, (t, 0, T)))', electronics)
rmsVoltageExpanded = Equation('i_eff == sqrt((1/T) * Integral(i_m**2 * cos(2*pi*f*t), (t, 0, T)))', electronics)
averagePower3 = Equation('p == v_rms*i_rms*cos(dph)', electronics)

#! ADD THESE!
# v_r = phI*r
# phI = phV_s / Z_eq
# P_r = 1/2 * |v_r| * I_m * cos(dph)

deciblePowerScale = Equation('G_db == 10*log(P_out/P_in, 10)', electronics)
decibleVoltageScale = Equation('A_vdb == 20*log(V_out/V_in, 10)', electronics)
systemTransferFunction = Equation('phH == phIn/phOut', electronics)
frequencyDependantVoltageDivider = Equation('V_out == V_s * (Z_C / (r + Z_C))', electronics)
# 1/sqrt(2) -- .707 -- -3db --some important constant?
# when a circuit has a voltage gain magnitude of -3db, the output voltage has a smaller magnitude than the input voltage

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

# q = C*V_v
# dq/dt = C * dv_c/dt
# i_c(t) = C * d*v_c/dt
# q=Cv_

# series connected capacitors all have the same seperated charge
# 2 capacitors in series --
# The imaginary part of impedance is reactance

#! # for constant dc current, a capaciter behaves like an open circuit -- and no current passes through

# powerDelivered to a capacitor = v*i = v * (C*dv/dt)

# power is the energy rate, p = dw/dt
# energy = w = integral(high=t, low=t_0, p(0) * something he changed the slide)


#* Misc:
# coulombsLaw = Equation()
# magneticFlux =


parallelCapacitors = llCap = series
seriesCapacitors = sCap = parallel

parallelInductors = llInd = parallel
seriesInductors = sInd = series

seriesImpedances = series
parallelImpedances = series

seriesAdmittances = parallel
parallelAdmittances = series
