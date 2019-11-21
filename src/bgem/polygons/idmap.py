"""
A dictionary that automaticaly assign unique IDs to appended objects
- IDs are set to the objects
- Objects can be compared by IDs.
TODO: Make auxiliary class for producing IDs and allow
several IdMaps to source from common ID source
"""
class IdObject:

    def __init__(self):
        self.attr = None

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


class IdSource:
    pass

class IdMap(dict):
    def __init__(self):
        self._next_id = -1
        # Source of new object IDs. The last used ID.
        self._free_ids = []
        # List of free ids
        super().__init__()


    def get_new_id(self):
        if self._free_ids:
            return self._free_ids.pop()
        self._next_id += 1
        return self._next_id

    def append(self, obj):
        id = self.get_new_id()
        obj.id = id
        self[obj.id] = obj
        return obj

    def remove(self, obj):
        del self[obj.id]
        self._free_ids.append(obj.id)
