import numpy as np
import json


# Custom encoder to write int64 into JSON (serializable)
class CustomEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        return super(CustomEncoder, self).default(obj)
