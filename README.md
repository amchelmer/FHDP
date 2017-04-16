# Demo code for Flexible Heuristic Dynamic Programming for Reinforcement Learning in Quad-Rotors


# Requirements
- Python 2.7
- Numpy
- Matplotlib
- Scipy
- Scikit-optimize
- iPython

all can be installed through the following command in the terminal
`pip install numpy scipy matplotlib scikit-optimize ipython`

Before being able to run any script, the `CACHE_PATH` needs to be defined in `lib/env/__init__.py`.

# Functioning
Run any script in `/scripts/` using `ipython /scripts/whatever_script.py -- -arg1 value1 -arg2 value2`
Scripts can also be run using `python` instead of `ipython`, but then the folder of the library needs to be added to the path. This can be done by adding the following code to the top of the script.

```
import sys
sys.path.extend(["./", "../"])
```

# Scripts
Several scripts are available. `training` scripts train one or more agents. The `skopt` scripts are used to optimize learning parameters. By passing a set of variables to the `x0` parameter in the optimize function, one can force a set of parameters to be tried. When making new scripts, ensure that the numpy error catching is set properly. This is done by adding the following line to the top of a script.

```
np.seterr(divide="raise", invalid="raise")
```

# Testing
Unit tests can be run through `nosetest lib/tests/`. Not all tests have been implemented yet. Some are missing for the following classes:

- `ActorCriticController`
- `ControllerSet`
- `KDTreeWrapper`
- `RewardSet`

# Multi-Processing
The library can leverage the power of multiithreading. By passing the argument `-j` to a script, multiple threads are started automatically. Also hyperthreading is supported.