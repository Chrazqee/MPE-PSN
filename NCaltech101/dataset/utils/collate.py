import pathlib
import sys
# from pprint import pprint
from typing import Any

import torch as th

pl = pathlib.Path(__file__).parent.parent
if pl not in sys.path:
    sys.path.append(str(pl))
if pl.parent not in sys.path:
    sys.path.append(str(pl.parent))
# pprint(sys.path)


class ncaltech101_collate:
    def __init__(self):
        self.origin_data = None
        self.image = None

def custom_collate(batch: Any):
    """
    Args:
        batch: (ndarray(164534, 4), tensor(2, 3, 224, 224), Tensor(224, 224, 3), dict: 2 {"bbox": tensor([ , , , ]), "label": }
    将 tensor(2, 3, 224, 224) stack 起来，dict 中的 bbox 和 label stack 起来
    Returns:
    """
    origin_data, data, image, target = tuple(zip(*batch))
    data = th.stack(data, dim=0)
    image = th.stack(image, dim=0)
    # origin_data = np.stack(origin_data, axis=0)
    extra = {"image": image, "origin_data": origin_data}
    return data, target, extra
