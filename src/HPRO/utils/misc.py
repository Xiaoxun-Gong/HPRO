import sys
import time, datetime
from os.path import join, dirname
import json
from collections import namedtuple
from math import ceil
from tqdm import tqdm
from tqdm.utils import disp_len, _unicode
import numpy as np

from .mpi import is_master

class mytqdm(tqdm):
    def __init__(self, iterable=None, **kwargs):
        if 'total' in kwargs:
            total = kwargs['total']
        else:
            total = len(iterable)
        miniters = ceil(total / 10)
        kwargs['miniters'] = miniters
        kwargs['file'] = sys.stdout
        kwargs['delay'] = 1e-5
        kwargs['leave'] = True # total%miniters!=0
        kwargs['bar_format'] = '{l_bar}{bar:40}{r_bar}{bar:-10b}'
        disable = kwargs.pop('disable', False)
        kwargs['disable'] = not is_master() or disable
        # prevent tqdm from changing miniters by itself,
        # see /home1/09019/xiaoxun/.local/lib/python3.9/site-packages/tqdm/_monitor.py", line 82
        kwargs['mininterval'] = 1
        kwargs['maxinterval'] = 1e10 
        super().__init__(iterable=iterable, **kwargs)
        # self.dynamic_miniters = False
        # self.miniters = miniters
        self.start_time = time.time()
    
    # @property
    # def miniters(self):
    #     return self._miniters
    
    # @miniters.setter
    # def miniters(self, v):
    #     self._miniters = v
    #     traceback.print_stack()
    
    
    @staticmethod
    def status_printer(file):
        """
        Manage the printing and in-place updating of a line of characters.
        Note that if the string is longer than a line, then in-place
        updating may not work (it will print a new line at each refresh).
        """
        fp = file
        fp_flush = getattr(fp, 'flush', lambda: None)  # pragma: no cover
        if fp in (sys.stderr, sys.stdout):
            getattr(sys.stderr, 'flush', lambda: None)()
            getattr(sys.stdout, 'flush', lambda: None)()

        def fp_write(s):
            fp.write(_unicode(s))
            fp_flush()

        last_len = [0]

        def print_status(s):
            len_s = disp_len(s)
            fp_write(s + (' ' * max(last_len[0] - len_s, 0)) + '\n') # always write new bar in new line
            last_len[0] = len_s

        return print_status

    def close(self):
        """Cleanup and (if leave=False) close the progressbar."""
        if self.disable:
            return

        # Prevent multiple closures
        self.disable = True

        # decrement instance pos and remove from internal set
        pos = abs(self.pos)
        self._decr_instances(self)

        if self.last_print_t < self.start_t + self.delay:
            # haven't ever displayed; nothing to clear
            return

        # GUI mode
        if getattr(self, 'sp', None) is None:
            return

        # annoyingly, _supports_unicode isn't good enough
        def fp_write(s):
            self.fp.write(_unicode(s))

        try:
            fp_write('')
        except ValueError as e:
            if 'closed' in str(e):
                return
            raise  # pragma: no cover

        leave = pos == 0 if self.leave is None else self.leave

        with self._lock:
            if leave:
                # stats for overall rate (no weighted average)
                self._ema_dt = lambda: None
                self.display(pos=0)
                # fp_write('\n')
            # else:
                # clear previous display
                # if self.display(msg='', pos=pos) and not pos:
                    # fp_write('\r')
        
        elapsed = time.time() - self.start_time
        fp_write(f'Done, elapsed time: {elapsed:4.1f}s.\n\n') # write timing message

    # def set_postfix(self, ordered_dict=None, refresh=False, **kwargs):
    #     return super().set_postfix(self, ordered_dict=ordered_dict, refresh=refresh, **kwargs)
    
    def update(self, n=1):
        if not self.disable:
            if self.n + n >= self.total:
                self.n += n
                self.last_print_n = self.n
                self.last_print_t = time.time()
                return
        return super().update(n=n)
    
    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            for obj in iterable:
                yield obj
            return

        try:
            for obj in iterable:
                yield obj
                self.update()
        finally:
            self.close()


with open(join(dirname(dirname(__file__)), "data", 'periodic_table.json'), 'r') as f:
    periodic_table = json.load(f)
    
def atom_name2number(element_names):
    return [periodic_table[spc]["Atomic no"] for spc in element_names]

def atom_number2name(atomic_numbers):
    element_names = []
    for n in atomic_numbers:
        for k, v in periodic_table.items():
            if v["Atomic no"] == n:
                element_names.append(k)
                break
    return element_names


KGData = namedtuple('KGData', ['ng', 'nbnd', 'miller_idc', 'unkg', 'kgcart'])
# miller_idc(ng, 3): int
# unkg(nbnd, ng): complex
# kgcart(ng, 3): float, already scaled by 2pi
# These are all distributed among different processors
VKBGData = namedtuple('VKBGData', ['ng', 'miller_idc', 'vkbg', 'kgcart'])
# miller_idc(ng, 3)
# vkbg: int->(nh, ng), \beta_{ah'sk}
# kgcart(ng, 3)


def set_range(xrange, xmin, xmax):
    if xrange is None:
        return (xmin, xmax)
    else:
        return (xmin if xrange[0] is None else xrange[0],
                xmax if xrange[1] is None else xrange[1])

def slice_same(array):
    '''
    array must be 1D
    Example: [1,1,2,2,2,3,3] -> [0, 2, 5, 7]
    '''
    return np.nonzero(np.r_[1, np.diff(array), 1])[0]

def simple_timer(formatstr):
    '''
    {t} in formatstr will be replaced by the time used by the function
    For example, "Total wall time: {t}" will be replaced by "Total wall time: 01:23:45"
    '''
    def timer_decorator(f):
        def g(*args, **kwargs):
            start_time = time.time()
            ret = f(*args, **kwargs)
            if is_master():
                total_time = datetime.timedelta(seconds=int(time.time()-start_time))
                # print(f'total wall time: {str(total_time)}\n')
                print(formatstr.format(t=str(total_time)))
            return ret
        return g
    return timer_decorator

def index_traverse(*indices):
    return np.stack(np.meshgrid(*indices, indexing='ij'), axis=-1).reshape(-1, len(indices))

def unique_nosort(arr):
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]

def _wrap_rank(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    av = np.abs(v)
    M  = av.max()
    rank = av.astype(np.int8)
    neg  = (v < 0)
    rank[neg] = (M + 1) + (M - av[neg])
    return rank

def sort_translations_siesta(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 2

    tx = arr[:, 0]
    ty = arr[:, 1]
    tz = arr[:, 2]

    rx = _wrap_rank(tx).astype(np.uint64)
    ry = _wrap_rank(ty).astype(np.uint64)
    rz = _wrap_rank(tz).astype(np.uint64)

    Bx = rx.max() + 1
    By = ry.max() + 1
    Bz = rz.max() + 1

    key = rz
    key = key * By + ry
    key = key * Bx + rx

    order = np.argsort(key, kind='stable')
    return arr[order]

class Timer:
    header = 'Routine           Calls     Tot.Time'

    def __init__(self, name):
        self.start_time = 0.
        self.total_time = 0.
        self.name = name
        self.ncalls = 0

    def start(self):
        self.start_time = time.time()
        self.ncalls += 1

    def stop(self):
        self.total_time += time.time() - self.start_time
    
    def __repr__(self):
        return f'{self.name:15} {self.ncalls:7d} {self.total_time:12.3f}'
    