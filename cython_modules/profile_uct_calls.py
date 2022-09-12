import pstats, cProfile
import pickle

import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

import cython_test

cProfile.runctx("cython_test.profile()", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
