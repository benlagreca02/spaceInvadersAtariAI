# Class and objects for color masking aliens out
# The idea is that this will be used to "quick and dirty" collect lots of pics of the aliens, which will then be
# manually sorted through. Just not consistent enough to feed directly into some game-playing ai...

class ColorBounds:
    def __init__(self, lh, uh, ls, us, lv, uv, kx, ky, sig=0):
        self.uh = uh
        self.lh = lh
        self.us = us
        self.ls = ls
        self.uv = uv
        self.lv = lv
        self.kx = kx
        self.ky = ky
        self.sig = sig

    def get_hue_bounds(self):
        return self.lh, self.uh

    def get_sat_bounds(self):
        return self.ls, self.us

    def get_lower_bounds(self):
        return self.lh, self.ls, self.lv

    def get_upper_bounds(self):
        return self.uh, self.us, self.uv

    def get_k_dims(self):
        return self.kx, self.ky

    def get_sigma(self):
        return self.sig


MIN_ALIEN_SIZE = 70
# constant values for roughly detecting the color of the aliens
BLUE_ALIEN = ColorBounds(119, 120, 183, 247, 63, 95, 31, 29)
YELLOW_ALIEN = ColorBounds(17, 22, 172, 246, 91, 132, 59, 59)

ALIENS = {"blue": BLUE_ALIEN, "yellow": YELLOW_ALIEN}
