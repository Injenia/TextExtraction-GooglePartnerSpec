# This Python file uses the following encoding: utf-8
from joblib import Parallel, delayed
from multiprocessing import cpu_count 

def parallel_run(fn, args_list, n_jobs=cpu_count()):
    return Parallel(n_jobs)(delayed(fn)(i) for i in args_list)

def parallel_run_args(fn, args_list, n_jobs=cpu_count()):
    return Parallel(n_jobs)(delayed(fn)(*i) for i in args_list)

def parallelize(fn, n_jobs=cpu_count()):
    def wrapper(vect, *args, **kwargs):
        return Parallel(n_jobs)(delayed(fn)(e, *args, **kwargs) for e in vect)
    return wrapper