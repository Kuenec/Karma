import numpy as np
from math import sqrt
from rlgym_sim.utils.gamestates.physics_object import PhysicsObject


def sign(x: float) -> int:
    return -1 if x < 0 else 1


def clamp(max_range: float, min_range: float, number: float) -> float:
    return max((min_range, min((number, max_range))))


def normalize(x: np.array) -> np.array:
    norm = np.linalg.norm(x)
    if norm == 0:
       return x
    return x / norm


def distance(x: np.array, y: np.array) -> float:
    return np.linalg.norm(x - y)


def distance2D(x: np.array, y: np.array)->float:
    x[2] = 0
    y[2] = 0
    return distance(x, y)


def relative_velocity(vec1: np.array, vec2: np.array)-> np.array:
    return vec1 - vec2


def relative_velocity_mag(vec1: np.array, vec2: np.array)-> float:
    return np.linalg.norm(relative_velocity(vec1, vec2))


def simple_physics_object_mirror(base: PhysicsObject) -> dict:
    """ mirrors data on the y axis """
    return {
        'position': np.array([-base.postion[0], base.position[1], base.postion[2]]),
        'linear_velocity': np.array([-base.linear_velocity[0], base.position[1], base.linear_velocity[2]]),
        'angular_velocity': np.array([-base.angular_velocity[0], base.angular_velocity[1], -base.angular_velocity[2]])
    }


if __name__ == "__main__":
    pass