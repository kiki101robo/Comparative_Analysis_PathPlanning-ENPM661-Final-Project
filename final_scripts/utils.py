import math
import cv2
import numpy as np

def compute_path_length(path):
    """
    Computes the total Euclidean length of a given path.
    """
    if not path or len(path) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(path) - 1):
        pt1 = path[i]
        pt2 = path[i + 1]

        x1, y1 = (pt1[0], pt1[1]) if isinstance(pt1, tuple) else (pt1.x, pt1.y)
        x2, y2 = (pt2[0], pt2[1]) if isinstance(pt2, tuple) else (pt2.x, pt2.y)

        total_length += np.hypot(x2 - x1, y2 - y1)

    return total_length


def compute_path_jerkiness(path):
    """
    Measures how jerky the path is based on changes in heading angle between segments.

    Returns:
        float: Average absolute angle change (in radians) between path segments.
    """
    if not path or len(path) < 3:
        return 0.0

    angle_changes = []
    for i in range(1, len(path) - 1):
        pt_prev = path[i - 1]
        pt_curr = path[i]
        pt_next = path[i + 1]

        x1, y1 = (pt_prev[0], pt_prev[1]) if isinstance(pt_prev, tuple) else (pt_prev.x, pt_prev.y)
        x2, y2 = (pt_curr[0], pt_curr[1]) if isinstance(pt_curr, tuple) else (pt_curr.x, pt_curr.y)
        x3, y3 = (pt_next[0], pt_next[1]) if isinstance(pt_next, tuple) else (pt_next.x, pt_next.y)

        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x3 - x2, y3 - y2])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue

        unit_v1 = v1 / norm1
        unit_v2 = v2 / norm2
        dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        angle = math.acos(dot)
        angle_changes.append(angle)

    return np.mean(angle_changes) if angle_changes else 0.0
