import numpy as np
import time
import string
import re

from collections import Counter
from functools import wraps
from tensorboardX import SummaryWriter


def progress(a, b):
    _progress = a / b
    bar_length = 5  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(_progress, int):
        _progress = float(_progress)
    if not isinstance(_progress, float):
        _progress = 0
        status = "error: progress var must be float\r\n"
    if _progress < 0:
        _progress = 0
        status = "Halt...\r\n"
    if _progress >= 1:
        _progress = 1
        status = ""
    block = int(round(bar_length * _progress))
    text = "[{}]\t{}/{} {}".format(
            "#" * block + " " * (bar_length-block), int(a), b, status)

    return text


def var_str(variable):
    return str(variable.data.cpu().numpy()) + ' ' + str(variable.size())


PROF_DATA = {}


# decorator for execution time measurement
class profile(object):
    def __init__(self, prefix):
        self.prefix = prefix

    def __call__(self, fn):
        def with_profiling(*args, **kwargs):
            global PROF_DATA
            start_time = time.time()
            ret = fn(*args, **kwargs)

            elapsed_time = time.time() - start_time
            key = '[' + self.prefix + '].' + fn.__name__

            if key not in PROF_DATA:
                PROF_DATA[key] = [0, []]
            PROF_DATA[key][0] += 1
            PROF_DATA[key][1].append(elapsed_time)

            return ret
        return with_profiling


def print_prof_data():
    for fname, data in sorted(PROF_DATA.items()):
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        total_time = sum(data[1])
        print("\n{} => called {} times.".format(fname, data[0]))
        print("Time total: {:.3f}, max: {:.3f}, avg: {:.3f}".format(
            total_time, max_time, avg_time))


def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
