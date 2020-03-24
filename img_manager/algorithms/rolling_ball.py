"""
Rolling ball Algorithm based on proposal from
https://github.com/scikit-image/scikit-image/issues/3538
"""

import numpy as np
from skimage.filters import threshold_local


def subtract_rolling_ball(image, radius):
    """Subtracts background from image using the Rolling Ball algorithm."""
    subtract = SubtractBall(radius)
    new_radius = subtract.ball.width
    background = threshold_local(image, new_radius, method='generic',
                                 param=subtract.bg)
    return image - background


class SubtractBall:
    """
        Generates a RollingBall of the selected radius and uses it's profile to
        calculate and subtract background.
    """
    def __init__(self, radius):
        self.ball = RollingBall(radius)

    def bg(self, surrounding_area):  # note that surrounding area is flattened
        return np.min(surrounding_area - self.ball.profile) + np.max(
            self.ball.profile)


class RollingBall:
    """
        A rolling ball (or actually a square part thereof)
        Here it is also determined whether to shrink the image

        Taken from https://github.com/mbalatsko/opencv-rolling-ball/blob/41462eec83ec652af0ee35ef9193049fa7e56e91/cv2_rolling_ball/background_subtractor.py#L447
    """

    def __init__(self, radius):
        """Builds a ball of the selected radius and calculates the shrink
        factor and arc_trim."""

        self.profile = []
        self.width = 0

        if radius <= 10:
            self.shrink_factor = 1
            arc_trim_per = 24
        elif radius <= 30:
            self.shrink_factor = 2
            arc_trim_per = 24
        elif radius <= 100:
            self.shrink_factor = 4
            arc_trim_per = 32
        else:
            self.shrink_factor = 8
            arc_trim_per = 40
        self.build(radius, arc_trim_per)

    def build(self, ball_radius, arc_trim_per):
        """Builds the ball into the profile attribute."""
        small_ball_radius = ball_radius / self.shrink_factor

        if small_ball_radius < 1:
            small_ball_radius = 1

        r_square = small_ball_radius * small_ball_radius
        x_trim = int(arc_trim_per * small_ball_radius / 100)
        half_width = round(small_ball_radius - x_trim)
        self.width = 2 * half_width + 1
        self.profile = [0] * (self.width * self.width)

        p = 0
        for y in range(self.width):
            for x in range(self.width):
                x_val = x - half_width
                y_val = y - half_width

                temp = r_square - x_val * x_val - y_val * y_val
                self.profile[p] = np.sqrt(temp) if temp > 0 else 0

                p += 1

        self.profile = np.array(self.profile)
        self.profile /= max(self.profile)

    def get_ball(self):
        """Returns an array with the values of the built ball."""
        return self.profile.reshape(self.width, -1)
