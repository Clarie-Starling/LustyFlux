# Python
from collections import OrderedDict

class PipelineLRU:
    def __init__(self, capacity=2):
        self.capacity = capacity
        self.store = OrderedDict()

    def get(self, key):
        v = self.store.get(key)
        if v is not None:
            self.store.move_to_end(key)
        return v

    def put(self, key, pipe):
        if key in self.store:
            self.store.move_to_end(key)
            self.store[key] = pipe
            return
        if len(self.store) >= self.capacity:
            old_key, old_pipe = self.store.popitem(last=False)
            try:
                unload_pipeline(old_pipe)
            except Exception:
                pass
            safe_cuda_cleanup("LRU-evict")
        self.store[key] = pipe