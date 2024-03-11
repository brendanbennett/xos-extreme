import pstats, cProfile

import sim

cProfile.runctx("sim.profiling()", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats(50)
