class Route:
    def __init__(self):
        self.required_arc_list = []  # [[edge, reverse]]
        self.load = 0
        self.cost = 0


class Solution:
    def __init__(self):
        self.route_list = []
        self.quality = 0
