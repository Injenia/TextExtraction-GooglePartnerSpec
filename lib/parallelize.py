# This Python file uses the following encoding: utf-8
from __future__ import print_function
from functools import wraps
from joblib import Parallel, delayed
from multiprocessing import cpu_count

def parallelize(fn, n_jobs=cpu_count()):
    '''
    Defines a parallelized version of the input function that takes a list (and other optional arguments)
    instead of the first argument and returns a list of results.

    Parameters:
        - fn: a callable that doesn't interfere with the other calls to the same function
        - n_jobs: the number of threads to use (default=number of cpus)
    Returns:
        - parallelized: the parallelized version of the function
    '''
    @wraps(fn)
    def parallalelized(vect, *args, **kwargs):
        return Parallel(n_jobs)(delayed(fn)(e, *args, **kwargs) for e in vect)
    parallalelized.__doc__ = 'Parallelized!!! ' + (fn.__doc__ if fn.__doc__ else '')
    return parallalelized

if __name__ == '__main__':
    # Example with pow
    print('pow(2,3)')
    print(pow(2,3))
    print('parallelize(pow)([3,2,3,5,6,4],3)')
    print(parallelize(pow)([3,2,3,5,6,4],3))