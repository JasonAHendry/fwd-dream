import numpy as np
from itertools import chain, combinations, product


def default(obj):
    """
    JSON serialize numpy arrays for json.dump()
    
    `delve()` produces dictionaries
    of numpy arrays as output. These cannot
    be written to .json unless the numpy
    arrays or serialized.
    
    """
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))