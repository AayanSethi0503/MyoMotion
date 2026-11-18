"""2-D geometry primitives shared by the biomechanics modules (NumPy / stdlib only).

Image coordinates: x to the right, **y downwards** (OpenCV / MediaPipe pixel space).
Angles are in degrees unless a name says otherwise.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

# calculate the angle at vertex b of the triangle formed by points a, b, c, with b being the center point.
# uses atan2(x,y) to calculate the angle, which is more numerically stable than using acos.
# atan2 is a piece wise function that returns the angle in radians
def joint_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Interior angle (0-180 deg) at vertex *b* of the path a-b-c; NaN if degenerate.

    Uses atan2(|cross|, dot), which is numerically stable over the whole 0-180 deg
    range (acos loses precision near 0/180 deg, exactly where full extension lives).
    Source: ported unchanged from the original deadlift tracker (``calculate_angle``).
    """
    v1x, v1y = a[0] - b[0], a[1] - b[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    if math.hypot(v1x, v1y) < 1e-6 or math.hypot(v2x, v2y) < 1e-6:
        return float("nan")
    cross = v1x * v2y - v1y * v2x
    dot = v1x * v2x + v1y * v2y
    return math.degrees(math.atan2(abs(cross), dot))



# calculates how much the body is leaning forward or backward, based solely off of the angle between the lower and upper segments of the body.
def inclination_from_vertical(lower: Sequence[float], upper: Sequence[float], facing: int) -> float:
    """Sagittal inclination of the segment lower->upper from vertical-up, in degrees.

    Positive = the *upper* end leans toward the direction the person faces (forward);
    0 = vertical; 90 = horizontal; negative = leaning backwards.
    ``facing`` is +1 if the person faces image-right, -1 if image-left.
    """
    dx = (upper[0] - lower[0]) * facing
    dy_up = lower[1] - upper[1]            # image y is down, so "up" is lower.y - upper.y

    #check if the upper segment is not too far from the lower segment, if it is close, return Nan
    if math.hypot(dx, dy_up) < 1e-9:
        return float("nan")
    return math.degrees(math.atan2(dx, dy_up))

#calcualtes angle of 2 points connected by a line, the above function calculates angle between 3 points, connected by 2 lines
def direction_deg(frm: Sequence[float], to: Sequence[float], facing: int) -> float:
    """Direction of frm->to measured like :func:`inclination_from_vertical` (vertical-up = 0,
    forward-positive). Same maths, named separately for tangents and curve segments."""
    return inclination_from_vertical(frm, to, facing)


#normalize angles so it stays within the range of -180 to 180 degrees
def wrap_deg(angle: float) -> float:
    """Wrap an angle to (-180, 180]."""
    a = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a

# calculates distance through the pythagoras theorem
def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# converts vectors to unit vectors
def unit(v: np.ndarray) -> np.ndarray:
    """Unit vector; zero vector stays zero."""
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-12 else np.zeros(2)
