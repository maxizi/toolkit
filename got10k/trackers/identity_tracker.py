from __future__ import absolute_import

from . import Tracker


class IdentityTracker(Tracker):

    def __init__(self):
        super(IdentityTracker, self).__init__(
            name='IdentityTracker',
            is_deterministic=True)

    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box

class IdentityTrackerRGBD(Tracker):

    def __init__(self):
        super(IdentityTrackerRGBD, self).__init__(
            name='IdentityTrackerRGBD',
            is_deterministic=True)

    def init(self, image, box, depth):
        self.box = box

    def update(self, image, depth):
        return self.box