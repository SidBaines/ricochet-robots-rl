import numpy as np

# Directions: 0: North (-1, 0), 1: East (0, 1), 2: South (1, 0), 3: West (0, -1)
DIRECTIONS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int8)
DIRECTION_NAMES = ['N', 'E', 'S', 'W']
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

# For visualization
ROBOT_COLORS = ['R', 'G', 'B', 'Y', 'P', 'O'] # Red, Green, Blue, Yellow, Purple, Orange
ROBOT_COLORS_PLOTLY = ['red', 'green', 'blue', 'gold', 'purple', 'orange']
TARGET_MARKER = 'T'
EMPTY_CELL = ' '
WALL_HORIZONTAL = '---'
WALL_VERTICAL = '|'
CORNER = '+'

DEFAULT_BOARD_SIZE = 16
DEFAULT_NUM_ROBOTS = 4 

def action_to_human_readable(action):
    robot_idx = action // 4
    direction = DIRECTION_NAMES[action % 4]
    return f"{ROBOT_COLORS_PLOTLY[robot_idx]} Robot moving {direction}"