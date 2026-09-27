DEFAULT_ITERATIONS = 1000000

ROBOT_RRT_RANGES = {
    "sphere": 1,
    "ur5": 1.5,
    "panda": 1.0,
    "fetch": 1.0,
    "baxter": 0.5,
    }

ROBOT_FIRST_JOINT_LOCATIONS = {
    "baxter": [0.0, 0.0, 0.0],
    "fetch": [0.0, 0.0, 0.4],
    "ur5": [0.0, 0.0, 0.91],
    "panda": [0.0, 0.0, 0.0],
    }

ROBOT_MAX_RADII = {
    "baxter": 1.31,
    "ur5": 1.2,
    "fetch": 1.5,
    "panda": 1.19,
    }

POINT_RADIUS = 0.0025
