from collections import OrderedDict

class TileCache:
    def __init__(self, capacity=256):
        self.capacity = capacity
        self.store = OrderedDict()

    def get(self, key):
        if key not in self.store:
            return None
        value = self.store.pop(key)
        self.store[key] = value
        return value

    def put(self, key, value):
        if key in self.store:
            self.store.pop(key)
        self.store[key] = value

        if len(self.store) > self.capacity:
            self.store.popitem(last=False)
