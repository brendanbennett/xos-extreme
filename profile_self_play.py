import pstats, cProfile

import sim

cProfile.runctx("sim.profiling_self_play()", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
