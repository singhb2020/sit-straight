# Debug Functions


# ------------------ Importing Libraries ------------------ #
import numpy as np
import cv2


# ------------------ Drawing Utilities ------------------ #
def draw_keypoints(frame, keypoints, confidence_threshold):
    """
    draw_keypoint: 
        Used to draw the keypoint outputs. Used only when debugging or calibrating.
    """

    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for kp in shaped:
        ky, kx, conf = kp
        if conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    """
    draw_connections: 
        Used to draw the edges between keypoint outputs. Used only when debugging or calibrating.
    """

    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if min(c1, c2) > confidence_threshold:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 1)


def get_edge_dictionary():
    """
    get_edge_dictionary: 
        Used to map pairs of keypoints to create an edge
    """

    return {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
    }