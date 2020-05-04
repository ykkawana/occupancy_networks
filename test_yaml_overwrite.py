# %%
import yaml
import os
from collections import defaultdict
import copy
import collections.abc


# %%
def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# %%
base = {
    'data': {
        'a': True,
        'b': False,
        'c': None,
        'd': 'str',
        'e': 5.0
    },
    'trainer': {
        'kwargs': {
            'a2': 3.0
        }
    }
}

old_base = copy.deepcopy(base)

unknown_args = [
    '--explicit', 'true', '--new_key', '32', '--data.a', 'false',
    '--trainer.kwargs.a2', '4'
]

print(old_base)
print(base)

# %%
