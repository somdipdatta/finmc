# Description: This file is used to profile the code using cProfile

import cProfile
import pstats
from pstats import SortKey

from tests.blackscholes.examples.option import run_model  # noqa: F401

if __name__ == "__main__":
    run_model()  # Run once for cold start imports
    cProfile.run("run_model()", "restats")
    p = pstats.Stats("restats")
    p.sort_stats(SortKey.CUMULATIVE).print_stats(30)
