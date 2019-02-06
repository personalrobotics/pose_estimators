
class DetectedItem:
    def __init__(self, namespace, idx, db_key,
                 x, y, z, ox, oy, oz, ow,
                 info_map=dict()):
        self.namespace = namespace
        self.id = idx
        self.info_map = info_map
        self.info_map['db_key'] = db_key
        self.x = x
        self.y = y
        self.z = z
        self.ox = ox
        self.oy = oy
        self.oz = oz
        self.ow = ow


