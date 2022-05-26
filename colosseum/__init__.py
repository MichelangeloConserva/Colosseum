import os
from glob import glob

from colosseum import benchmark

bench_f = benchmark.__file__

benchmark_dirs = [x for x in glob(bench_f[: bench_f.rfind(os.sep) + 1] + "benc*")]
