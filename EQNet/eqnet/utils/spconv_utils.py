from typing import Set

try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

import torch.nn as nn


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out