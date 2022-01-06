#!/bin/bash

echo Installing Libs...
export PYTHONPATH=${PYTHONPATH}:${HOME}/libs/Python
export C_INCLUDE_PATH=${C_INCLUDE_PATH}:${HOME}/libs/C
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${HOME}/libs/C++

# Q is the charge?
# 1 coulomb/second = 1 Amp
# Q(t) = integral(high=t, low=t_0, i(t))
# i = current = Amps = derivative(high=q, low=t, coulomb/second)
