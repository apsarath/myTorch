import myTorch
from myTorch.environment.BlocksworldMatrix import Block


class Tower(object):
    def __init__(self, block_id):
        self._table_loc = None
        self._blocks = [block_id]
        self._is_grounded = False

    def place_on_top(self, target_tower):
        assert(not target_tower.is_grounded)
        self._blocks = target_tower.blocks + self._blocks

    def ground(self):
        self._is_grounded = True

    @property
    def loc(self):
        return self._table_loc

    @property
    def blocks(self):
        return self._blocks

    @property
    def is_grounded(self):
        return self._is_grounded

