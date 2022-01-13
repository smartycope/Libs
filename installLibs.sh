#!/bin/bash

# echo Installing Libs...
libsPath=${HOME}/hello/Libs
export PYTHONPATH=${PYTHONPATH}:$libsPath/Python
export C_INCLUDE_PATH=${C_INCLUDE_PATH}:$libsPath/C
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:$libsPath/C++
# echo Finished Installing Libs
# export PYTHONPATH="$PYTHONPATH:$HOME/Libs/Python"

# Q is the charge?
# 1 coulomb/second = 1 Amp
# Q(t) = integral(high=t, low=t_0, i(t))
# i = current = Amps = derivative(high=q, low=t, coulomb/second)
