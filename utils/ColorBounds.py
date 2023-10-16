
# Class and objects for color masking aliens out
# The idea is that this will be used to "quick and dirty" collect lots of pics of the aliens, which will then be
# manually sorted through. Just not consistent enough to feed directly into some game-playing ai...

class ColorBounds:
    def __init__(self, lh, uh, ls, us, lv, uv, k, sig):
        self.uh = uh
        self.lh = lh
        self.us = us
        self.ls = ls
        self.uv = uv
        self.lv = lv
        self.k = k
        self.sig = sig

    def get_hue_bounds(self):
        return self.lh, self.uh

    def get_sat_bounds(self):
        return self.ls, self.us

    def get_lower_bounds(self):
        return self.lh, self.ls, self.lv

    def get_upper_bounds(self):
        return self.uh, self.us, self.uv

    def get_k(self):
        return self.k

    def get_sigma(self):
        return self.sig


BLUE_ALIEN = ColorBounds(120, 122, 183, 228, 83, 104, 45, 0)
YELLOW_ALIEN = ColorBounds(16, 23, 198, 230, 116, 148, 59, 0)
