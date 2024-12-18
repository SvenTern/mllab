"""
Contains the logic from chapter 20 on multiprocessing and vectorization.
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import datetime as dt

def lin_parts(num_atoms, num_threads):
    """
    Partitions a list of atoms into subsets of equal size.
    
    :param num_atoms: (int) Number of atoms
    :param num_threads: (int) Number of processors
    :return: (np.array) Partition of atoms
    """
    num_threads = min(num_threads, num_atoms)
    parts = np.linspace(0, num_atoms, num_threads + 1).astype(int)
    return parts

def nested_parts(num_atoms, num_threads, upper_triangle=False):
    """
    Enables parallelization of nested loops.

    :param num_atoms: (int) Number of atoms
    :param num_threads: (int) Number of processors
    :param upper_triangle: (bool) Flag to order atoms as an upper triangular matrix
    :return: (list) Partition of atoms
    """
    parts, num_threads = [0], min(num_threads, num_atoms)
    for num in range(num_threads):
        part = 1 + 4 * (parts[-1]**2 + parts[-1] + num_atoms * (num_atoms + 1) / num_threads)
        part = (-1 + part**0.5) / 2
        parts.append(min(num_atoms, int(part)))
    parts = np.array(parts).astype(int)
    if upper_triangle:
        jobs = []
        for i in range(1, len(parts)):
            jobs.append((parts[i - 1], parts[i]))
        return jobs
    return parts

def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True, verbose=True, **kwargs):
    """
    Parallelizes jobs, returning a pandas DataFrame or Series.
    
    :param func: (function) Callback function to execute in parallel
    :param pd_obj: (tuple) ("molecule", list of atoms)
    :param num_threads: (int) Number of threads
    :param mp_batches: (int) Number of parallel batches
    :param lin_mols: (bool) Use linear partitioning
    :param verbose: (bool) Verbose output
    :param kwargs: Additional arguments for func
    :return: (pd.DataFrame or pd.Series) Results
    """
    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), num_threads * mp_batches)
    else:
        parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches)

    jobs = []
    for i in range(1, len(parts)):
        job = {
            "func": func,
            "kwargs": {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], **kwargs}
        }
        jobs.append(job)

    if num_threads == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads, verbose=verbose)

    if not out:  # Ensure out is not empty
        return pd.DataFrame() if isinstance(pd_obj[1], pd.DataFrame) else pd.Series()

    if isinstance(out[0], pd.DataFrame):
        return pd.concat(out)
    elif isinstance(out[0], pd.Series):
        return pd.concat(out)
    else:
        return out

def process_jobs_(jobs):
    """
    Processes jobs sequentially for debugging.

    :param jobs: (list) Jobs
    :return: (list) Results of jobs
    """
    out = []
    for job in jobs:
        out_ = expand_call(job)
        out.append(out_)
    return out

def expand_call(kargs):
    """
    Executes the callback function for a job.

    :param kargs: (dict) Job arguments
    :return: Result of the job
    """
    func = kargs['func']
    del kargs['func']
    return func(**kargs)

def report_progress(job_num, num_jobs, time0, task):
    """
    Reports progress of asynchronous jobs.

    :param job_num: (int) Current job number
    :param num_jobs: (int) Total number of jobs
    :param time0: (time) Start time
    :param task: (str) Task description
    :return: (None)
    """
    msg = [
        task,
        f'{job_num}/{num_jobs} completed',
        f'{(job_num / num_jobs):.2%} progress',
        f'elapsed time: {time.time() - time0:.2f}s',
        f'estimated remaining time: {(time.time() - time0) * (num_jobs / job_num - 1):.2f}s'
    ]
    print(', '.join(msg))

def process_jobs(jobs, task=None, num_threads=24, verbose=True):
    """
    Executes jobs in parallel using multiprocessing.

    :param jobs: (list) Jobs
    :param task: (str) Task description
    :param num_threads: (int) Number of threads
    :param verbose: (bool) Verbose output
    :return: (list) Results of jobs
    """
    if verbose:
        time0 = time.time()
        job_num = 0

    pool = mp.Pool(processes=num_threads)
    out = []
    for i, job in enumerate(jobs):
        out.append(pool.apply_async(expand_call, args=(job,)))

    pool.close()
    pool.join()

    out = [o.get() for o in out]

    if verbose:
        report_progress(len(jobs), len(jobs), time0, task)

    return out
