import collections
import glob
import math
import os
import pandas

from scipy.interpolate import interp1d, LinearNDInterpolator

def memoize(f):
    memo = {}
    def helper(*x):
        if x not in memo:
            memo[x] = f(*x)
        return memo[x]
    return helper

class Application(object):
    def __init__(self, trace_dir, max_steps=None):
        self.name = os.path.basename(trace_dir)

        self.max_steps = max_steps

        df = pandas.read_csv('./traces/atari-impala/placement.csv')

        self.max_workers = max(df.num_workers)

        xs = ["num_workers", "num_nodes"]
        ys = ["throughput"]
        df = df.groupby(xs)[xs + ys].mean()
        self.interpolator = LinearNDInterpolator(df[xs].values, df[ys].values)
    
    @memoize
    def get_throughput(self, num_workers, num_nodes):
        ret = self.interpolator(num_workers, num_nodes)
        return ret


TRACES_DIR = os.path.join(os.path.dirname(__file__), "traces")
APPLICATIONS = {
    "atari-impala": Application(os.path.join(TRACES_DIR, "atari-impala"), max_steps=100000000),
    "doom": Application(os.path.join(TRACES_DIR, "doom"), max_steps=100000000),
    "mujoco": Application(os.path.join(TRACES_DIR, "mujoco"), max_steps=100000000),
}

if __name__ == "__main__":
    app = APPLICATIONS["atari-impala"]
    ret = app.get_throughput(28, 1)
    print(ret)
