import math
import random
from collections import namedtuple
from typing import List, Union

import numpy as np
from rlgym_sim.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, \
    CAR_MAX_ANG_VEL, \
    BALL_MAX_SPEED, BLUE_TEAM, ORANGE_TEAM, BOOST_LOCATIONS
from rlgym_sim.utils.gamestates import PhysicsObject
from rlgym_sim.utils.math import rand_vec3
from rlgym_sim.utils.state_setters.random_state import RandomState
from rlgym_sim.utils.state_setters import StateWrapper, StateSetter
from rlgym_sim.utils.state_setters.wrappers import CarWrapper

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

GOAL_X_MAX = 800.0
GOAL_X_MIN = -800.0

PLACEMENT_BOX_X = 5000
PLACEMENT_BOX_Y = 2000
PLACEMENT_BOX_Y_OFFSET = 3000

GOAL_LINE = 5100

YAW_MAX = np.pi

DEG_TO_RAD = np.pi / 180

from numpy import random as rand

X_MAX = 7000
Y_MAX = 9000
Z_MAX_CAR = 1900
GOAL_HEIGHT = 642.775
PITCH_MAX = np.pi / 2
ROLL_MAX = np.pi
BLUE_GOAL_POSITION = np.asarray([0, -5120, 0])
ORANGE_GOAL_POSITION = np.asarray([0, 5120, 0])


class LeviathanState(StateSetter):
    def __init__(self, team_sizes: tuple = (1, 2, 3)):
        self.team_sizes = team_sizes
        self.iteration = 0

        self.default_state = DefaultState()

    def build_wrapper(self, max_team_size: int, spawn_opponents: bool) -> StateWrapper:
        team_size = self.team_sizes[self.iteration]

        self.iteration += 1
        if self.iteration == len(self.team_sizes):
            self.iteration = 0

        return StateWrapper(team_size, team_size)

    def reset(self, state_wrapper: StateWrapper):
        # state setter here
        self.default_state.reset(state_wrapper)


class HoopsLikeSetter(StateSetter):

    def __init__(self, spawn_radius: float = 800):
        """
        Hoops-like kickoff constructor

        :param spawn_radius: Float determining how far away to spawn the cars from the ball.
        """
        super().__init__()
        self.spawn_radius = spawn_radius
        self.yaw_vector = np.asarray([-1, 0, 0])

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to contain random values the ball and each car.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        self._reset_ball_random(state_wrapper)
        self._reset_cars_random(state_wrapper, self.spawn_radius)

    def _reset_ball_random(self, state_wrapper: StateWrapper):
        """
        Resets the ball according to a uniform distribution along y-axis and normal distribution along x-axis.

        :param state_wrapper: StateWrapper object to be modified.
        """
        new_x = rand.uniform(-4000, 4000)
        new_y = rand.triangular(-5000, 0, 5000)
        new_z = rand.uniform(GOAL_HEIGHT, CEILING_Z - 2 * BALL_RADIUS)
        z = rand.uniform(400, 600)
        state_wrapper.ball.set_pos(new_x, new_y, z)
        state_wrapper.ball.set_ang_vel(0, 0, 0)
        state_wrapper.ball.set_lin_vel(0, 0, 0)

    def _reset_cars_random(self, state_wrapper: StateWrapper, spawn_radius: float):
        """
        Function to set all cars inbetween the ball and net roughly facing the ball. The other cars will be spawned
        randomly.

        :param state_wrapper: StateWrapper object to be modified.
        :param spawn_radius: Float determining how far away to spawn the cars from the ball.
        """
        orange_done = False
        blue_done = False
        for i, car in enumerate(state_wrapper.cars):
            if car.team_num == BLUE_TEAM and not blue_done:
                # just shorthands for ball_x and ball_y
                bx = state_wrapper.ball.position[0]
                by = state_wrapper.ball.position[1]

                # add small variation to spawn radius
                car_spawn_radius = rand.triangular(0.8, 1, 1.2) * spawn_radius

                # calculate distance from ball to goal.
                R = ((BLUE_GOAL_POSITION[0] - bx) ** 2 + (BLUE_GOAL_POSITION[1] - by) ** 2) ** 0.5

                # use similarity of triangles to calculate offsets
                x_offset = ((BLUE_GOAL_POSITION[0] - bx) / R) * car_spawn_radius
                y_offset = ((BLUE_GOAL_POSITION[1] - by) / R) * car_spawn_radius

                # offset the car's positions
                car.set_pos(bx + x_offset, by + y_offset, 17)

                # compute vector from ball to car
                rel_ball_car_vector = car.position - state_wrapper.ball.position

                # calculate the angle between the yaw vector and relative vector and use that as yaw.

                yaw = np.arccos(np.dot(rel_ball_car_vector / np.linalg.norm(rel_ball_car_vector), self.yaw_vector))

                # then sprinkle in more variation by offsetting by random angle up to pi/8 radians (this is arbitrary)
                max_offset_angle = np.pi / 8
                yaw += rand.random() * max_offset_angle * 2 - max_offset_angle

                # random at least slightly above 0 boost amounts
                car.boost = rand.uniform(0.2, 1)

                # make car face the ball roughly
                car.set_rot(pitch=0, roll=0, yaw=yaw)

                # x angular velocity (affects pitch) set to 0
                # y angular velocity (affects roll) set to 0
                # z angular velocity (affects yaw) set to 0
                car.set_ang_vel(x=0, y=0, z=0)
                car.set_lin_vel(x=0, y=0, z=0)
                blue_done = True
            elif car.team_num == ORANGE_TEAM and not orange_done:
                # just shorthands for ball_x and ball_y
                bx = state_wrapper.ball.position[0]
                by = state_wrapper.ball.position[1]

                # add small variation to spawn radius
                car_spawn_radius = rand.triangular(0.8, 1, 1.2) * spawn_radius

                # calculate distance from ball to goal.
                R = ((ORANGE_GOAL_POSITION[0] - bx) ** 2 + (ORANGE_GOAL_POSITION[1] - by) ** 2) ** 0.5

                # use similarity of triangles to calculate offsets
                x_offset = ((ORANGE_GOAL_POSITION[0] - bx) / R) * car_spawn_radius
                y_offset = ((ORANGE_GOAL_POSITION[1] - by) / R) * car_spawn_radius

                # offset the car's positions
                car.set_pos(bx + x_offset, by + y_offset, 17)

                # compute vector from ball to car
                rel_ball_car_vector = car.position - state_wrapper.ball.position

                # calculate the angle between the yaw vector and relative vector and use that as yaw.

                yaw = np.arccos(np.dot(rel_ball_car_vector / np.linalg.norm(rel_ball_car_vector), self.yaw_vector))

                # then sprinkle in more variation by offsetting by random angle up to pi/8 radians (this is arbitrary)
                max_offset_angle = np.pi / 8
                yaw += rand.random() * max_offset_angle * 2 - max_offset_angle

                # random at least slightly above 0 boost amounts
                car.boost = rand.uniform(0.2, 1)

                # make car face the ball roughly
                car.set_rot(pitch=0, roll=0, yaw=yaw - np.pi if yaw > 0 else yaw + np.pi)

                # x angular velocity (affects pitch) set to 0
                # y angular velocity (affects roll) set to 0
                # z angular velocity (affects yaw) set to 0
                car.set_ang_vel(x=0, y=0, z=0)
                car.set_lin_vel(x=0, y=0, z=0)
                orange_done = True
            else:
                car.set_pos(rand.uniform(-4096, 4096)), rand.uniform(0, -1 ** (car.team_num - 1) * 5120, 17)

                car.boost = rand.uniform(0.2, 1)

                # make car face the ball roughly
                car.set_rot(pitch=0, roll=0, yaw=rand.uniform(-np.pi, np.pi))

                # x angular velocity (affects pitch) set to 0
                # y angular velocity (affects roll) set to 0
                # z angular velocity (affects yaw) set to 0
                car.set_ang_vel(x=0, y=0, z=0)
                car.set_lin_vel(x=0, y=0, z=0)

def mirror(car: CarWrapper, ball_x, ball_y):
    my_car = namedtuple('my_car', 'pos lin_vel rot ang_vel')
    if ball_x == ball_y == 0:
        my_car.pos = -car.position[0], -car.position[1], car.position[2]
        my_car.lin_vel = -car.linear_velocity[0], -car.linear_velocity[1], car.linear_velocity[2]
        my_car.rot = car.rotation[0], -car.rotation[1], car.rotation[2]
        my_car.ang_vel = -car.angular_velocity[0], -car.angular_velocity[1], car.angular_velocity[2]
    elif ball_x == 0:
        my_car.pos = -car.position[0], car.position[1], car.position[2]
        my_car.lin_vel = -car.linear_velocity[0], car.linear_velocity[1], car.linear_velocity[2]
        my_car.rot = car.rotation[0], -car.rotation[1], car.rotation[2]
        my_car.ang_vel = -car.angular_velocity[0], -car.angular_velocity[1], car.angular_velocity[2]
    elif ball_y == 0:
        my_car.pos = car.position[0], -car.position[1], car.position[2]
        my_car.lin_vel = -car.linear_velocity[0], car.linear_velocity[1], car.linear_velocity[2]
        my_car.rot = car.rotation[0], -car.rotation[1], car.rotation[2]
        my_car.ang_vel = -car.angular_velocity[0], -car.angular_velocity[1], car.angular_velocity[2]
    elif ball_x == ball_y and car.position[0] > car.position[1]:
        my_car.pos = -car.position[0], -car.position[1], car.position[2]
        my_car.lin_vel = car.linear_velocity[1], car.linear_velocity[0], car.linear_velocity[2]
        my_car.rot = car.rotation[0] - np.pi / 2, car.rotation[1], car.rotation[2]
        my_car.ang_vel = -car.angular_velocity[0], -car.angular_velocity[1], car.angular_velocity[2]
    else:
        return None
    return my_car


def set_pos(end_object: PhysicsObject, x: float = None, y: float = None, z: float = None):
    """
    Sets position.
    :param end_object: object to set
    :param x: Float indicating x position value.
    :param y: Float indicating y position value.
    :param z: Float indicating z position value.
    """
    if x is not None and y is not None and z is not None:
        if x == y == z == -1:
            end_object.position[0] = -1
            end_object.position[1] = -1
            end_object.position[2] = -1
            return

    if x is not None:
        end_object.position[0] = max(min(x, 4096), -4096)
    if y is not None:
        end_object.position[1] = max(min(y, 3800), -3800)
    if z is not None:
        end_object.position[2] = max(min(z, 1700), 350)


def random_valid_loc() -> np.ndarray:
    rng = np.random.default_rng()
    rand_x = rng.uniform(-4000, 4000)
    if abs(rand_x) > (4096 - 1152):
        rand_y = rng.uniform(-5120 + 1152, 5120 - 1152)
    else:
        rand_y = rng.uniform(-5020, 5020)
    rand_z = rng.uniform(20, 2000)
    return np.asarray([rand_x, rand_y, rand_z])


class AerialStateSetter(StateSetter):
    MID_AIR = 900
    LOW_AIR = 150
    HIGH_AIR = 1500
    CEILING = CEILING_Z


def compare_arrays_inferior(arr1: np.array, arr2: np.array) -> bool:
    return all(arr1[0:2] < arr2[0:2])


def _check_positions(state: StateWrapper) -> bool:
    CAR_DIM = np.array((50, 50, 50))

    # Checking all others for each car
    for car in state.cars:

        current_car = car

        for other in state.cars:
            if other == current_car:
                continue

            if compare_arrays_inferior(abs(current_car.position - other.position), CAR_DIM):
                print("Car too close, resetting")
                return False

        if compare_arrays_inferior(abs(current_car.position - state.ball.position), CAR_DIM):
            return False

    return True


class CustomStateSetter(StateSetter):
    """def reset(self, state_wrapper: StateWrapper):
        # Set up our desired spawn location and orientation. Here, we will only change the yaw, leaving the remaining
        for car in state_wrapper.cars:

            desired_car_pos = [random.randint(-4000, 4000), random.randint(-5000, 5000), random.randint(50, int(CEILING_Z) - int(BALL_RADIUS) * 2)]
            desired_yaw = np.pi / 2

            #print(desired_car_pos)

            if car.team_num == BLUE_TEAM:
                pos = desired_car_pos
                yaw = desired_yaw

            elif car.team_num == ORANGE_TEAM:  # invert
                pos = [-1 * coord for coord in desired_car_pos]
                yaw = -1 * desired_yaw

            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = random.randint(0, 100)

        state_wrapper.ball.set_pos(random.randint(-3000, 3000), random.randint(-3000, 3000),
                                   random.randint(0 + int(BALL_RADIUS) * 2, int(CEILING_Z) - int(BALL_RADIUS) * 2))"""

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(
            x=np.random.uniform(-LIM_X, LIM_X),
            y=np.random.uniform(-LIM_Y, LIM_Y),
            z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
        )

        # 99.9% chance of below ball max speed
        ball_speed = np.random.exponential(-BALL_MAX_SPEED / np.log(1 - 0.999))
        vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED))
        state_wrapper.ball.set_lin_vel(*vel)

        ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
        state_wrapper.ball.set_ang_vel(*ang_vel)

        for car in state_wrapper.cars:
            # On average 1 second at max speed away from ball
            ball_dist = np.random.exponential(BALL_MAX_SPEED)
            ball_car = rand_vec3(ball_dist)
            car_pos = state_wrapper.ball.position + ball_car
            if abs(car_pos[0]) < LIM_X \
                    and abs(car_pos[1]) < LIM_Y \
                    and 0 < car_pos[2] < LIM_Z:
                car.set_pos(*car_pos)
            else:  # Fallback on fully random
                car.set_pos(
                    x=np.random.uniform(-LIM_X, LIM_X),
                    y=np.random.uniform(-LIM_Y, LIM_Y),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
                )

            vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
            car.set_lin_vel(*vel)

            car.set_rot(
                pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
            )

            ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
            car.set_ang_vel(*ang_vel)
            car.boost = np.random.uniform(0, 1)


class DribblingStateSetter(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            selected_car = random.choice(state_wrapper.blue_cars())

            # Place the ball on top of the selected car
            state_wrapper.ball.set_pos(
                x=selected_car.position[0],
                y=selected_car.position[1],
                z=selected_car.position[2] + BALL_RADIUS + 20  # Car height + ball radius
            )

            # Set the ball velocity to zero
            state_wrapper.ball.set_lin_vel(0, 0, -0.1)
            state_wrapper.ball.set_ang_vel(0, 0, -0.1)

            # Set the car's position and velocity
            selected_car.set_pos(
                x=random.uniform(-LIM_X, LIM_X),
                y=random.uniform(-LIM_Y, LIM_Y),
                z=17
            )

            selected_car.set_lin_vel(0, 0, 0)
            selected_car.set_ang_vel(0, 0, 0)

            selected_car.set_rot(
                pitch=0,
                yaw=random.uniform(-YAW_LIM, YAW_LIM),
                roll=0
            )

            selected_car.boost = .52


class ShotState(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                car.set_pos(
                    random.uniform(-4096, 4096),
                    random.uniform(0, 3000),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

                state_wrapper.ball.set_pos(
                    x=np.random.uniform(max(car.position.item(0) - 1000, -LIM_X),
                                        min(car.position.item(0) + 1000, LIM_X)),
                    y=np.random.uniform(car.position.item(1) + 1000, car.position.item(1) + 100),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z / 2),
                )

                ball_speed = np.random.exponential(-(BALL_MAX_SPEED / 3) / np.log(1 - 0.999))
                vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED / 3))
                state_wrapper.ball.set_lin_vel(*vel)

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
                state_wrapper.ball.set_ang_vel(*ang_vel)

            if car.team_num == ORANGE_TEAM:
                car.set_pos(
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)


class JumpShotState(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                car.set_pos(
                    random.uniform(-4096, 4096),
                    random.uniform(0, 2500),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=0,
                    yaw=90,
                    roll=0
                )

                ang_vel = (0, 0, 0)  # rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

                state_wrapper.ball.set_pos(
                    x=np.random.uniform(max(car.position.item(0) - 1000, -LIM_X),
                                        min(car.position.item(0) + 1000, LIM_X)),
                    y=np.random.uniform(car.position.item(1) + 1500, car.position.item(1) + 500),
                    z=CEILING_Z / 2
                )

                ball_speed = np.random.uniform(100, BALL_MAX_SPEED / 2)
                vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED / 2))
                state_wrapper.ball.set_lin_vel(*vel)

                ang_vel = (0, 0, 0)
                state_wrapper.ball.set_ang_vel(*ang_vel)

            if car.team_num == ORANGE_TEAM:
                car.set_pos(
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)


class SaveState(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == ORANGE_TEAM:
                car.set_pos(
                    random.uniform(-4096, 4096),
                    random.uniform(0, -3000),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

                state_wrapper.ball.set_pos(
                    x=np.random.uniform(max(car.position.item(0) - 1000, -LIM_X),
                                        min(car.position.item(0) + 1000, LIM_X)),
                    y=np.random.uniform(car.position.item(1) - 1000, car.position.item(1) - 100),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z / 2),
                )

                ball_speed = np.random.exponential(-(BALL_MAX_SPEED / 3) / np.log(1 - 0.999))
                vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED / 3))
                state_wrapper.ball.set_lin_vel(*vel)

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
                state_wrapper.ball.set_ang_vel(*ang_vel)

            if car.team_num == BLUE_TEAM:
                car.set_pos(
                    random.randint(-2900, 2900),
                    -random.randint(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)


class AirDribble2Touch(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):

        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:

                car_x = random.randint(-3500, 3500)
                car_y = random.randint(-2000, 4000)
                car_z = np.random.uniform(500, LIM_Z)

                car.set_pos(
                    car_x,
                    car_y,
                    car_z
                )

                car_pitch_rot = random.uniform(-1, 1) * math.pi
                car_yaw_rot = random.uniform(-1, 1) * math.pi
                car_roll_rot = random.uniform(-1, 1) * math.pi

                car.set_rot(
                    car_pitch_rot,
                    car_yaw_rot,
                    car_roll_rot
                )

                car_lin_y = np.random.uniform(300, CAR_MAX_SPEED)

                car.set_lin_vel(
                    0 + random.uniform(-150, 150),
                    car_lin_y,
                    0 + random.uniform(-150, 150)
                )

                state_wrapper.ball.set_pos(
                    car_x + random.uniform(-150, 150),
                    car_y + random.uniform(0, 150),
                    car_z + random.uniform(0, 150)
                )

                ball_lin_y = car_lin_y + random.uniform(-150, 150)

                state_wrapper.ball.set_lin_vel(
                    0 + random.uniform(-150, 150),
                    ball_lin_y,
                    0 + random.uniform(-150, 150)
                )
                car.boost = np.random.uniform(0.4, 1)

            else:
                car.set_pos(
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)


class DefaultState(StateSetter):
    SPAWN_BLUE_POS = [[-2048, -2560, 17], [2048, -2560, 17],
                      [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]
    SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                      0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
    SPAWN_ORANGE_POS = [[2048, 2560, 17], [-2048, 2560, 17],
                        [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]
    SPAWN_ORANGE_YAW = [-0.75 * np.pi, -0.25 *
                        np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        rand_default_or_diff = random.randint(0, 3)
        # rand_default_or_diff = 0
        # DEFAULT KICKOFFS POSITIONS
        if rand_default_or_diff == 0:

            # possible kickoff indices are shuffled
            spawn_inds = [0, 1, 2, 3, 4]
            random.shuffle(spawn_inds)

            blue_count = 0
            orange_count = 0
            for car in state_wrapper.cars:
                pos = [0, 0, 0]
                yaw = 0
                # team_num = 0 = blue team
                if car.team_num == 0:
                    # select a unique spawn state from pre-determined values
                    pos = self.SPAWN_BLUE_POS[spawn_inds[blue_count]]
                    yaw = self.SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                    blue_count += 1
                # team_num = 1 = orange team
                elif car.team_num == 1:
                    # select a unique spawn state from pre-determined values
                    pos = self.SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                    yaw = self.SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                    orange_count += 1
                # set car state values
                car.set_pos(*pos)
                car.set_rot(yaw=yaw)
                car.boost = 0.33
        else:
            """SPAWN POS RANGE
            Y = CENTER BACK KICKOFF [-4608, 0], [4608, 0]; BACK LEFT/RIGHT KICKOFFS [-3000, 0], [3000,0]"""

            # print("Advanced Kickoffs")
            # SPAWN POSITION
            spawn_inds = [0, 1, 2, 3, 4]
            spawn_pos = np.random.choice(spawn_inds)

            same_pos = random.randint(0, 1)
            # print(f"Same position = {same_pos}")

            # RIGHT CORNER
            if spawn_pos == 0:
                # print("Right Corner")
                # SAME POS
                if same_pos == 1:
                    slope = -2560 / -2048
                    posX = np.random.uniform(-2048, 0)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 200)
                    blue_spawn_pos = (posX, posY, posZ)
                    orange_spawn_pos = (abs(posX), abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -2560 / -2048
                        blue_posX = np.random.uniform(-2048, 0)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        blue_spawn_pos = (blue_posX, blue_posY, posZ)

                        slope = 2560 / 2048
                        orange_posX = np.random.uniform(2048, 0)
                        orange_posY = slope * orange_posX
                        posZ = np.random.uniform(17, 500)
                        orange_spawn_pos = (orange_posX, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            # LEFT CORNER
            elif spawn_pos == 1:
                # print("Left Corner")
                # SAME POS
                if same_pos == 1:
                    slope = -2560 / 2048
                    posX = np.random.uniform(2048, 0)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 500)
                    blue_spawn_pos = (posX, posY, posZ)
                    orange_spawn_pos = (-abs(posX), abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -2560 / 2048
                        blue_posX = np.random.uniform(2048, 0)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        blue_spawn_pos = (blue_posX, blue_posY, posZ)

                        slope = 2560 / -2048
                        orange_posX = np.random.uniform(-2048, 0)
                        orange_posY = slope * orange_posX
                        posZ = np.random.uniform(17, 500)
                        orange_spawn_pos = (orange_posX, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        #  print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            elif spawn_pos == 2:
                # print("Back Right")
                # SAME POS
                if same_pos == 1:
                    slope = -3840 / -256
                    posX = np.random.uniform(-256, 0)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 500)

                    if posY < -3000:
                        blue_spawn_pos = (posX, posY, posZ)
                        orange_spawn_pos = (abs(posX), abs(posY), posZ)
                    else:
                        blue_spawn_pos = (0, posY, posZ)
                        orange_spawn_pos = (0, abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -3840 / -256
                        blue_posX = np.random.uniform(-256, 0)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        if blue_posY < -3000:
                            blue_spawn_pos = (blue_posX, blue_posY, posZ)
                        else:
                            blue_spawn_pos = (0, blue_posY, posZ)

                        slope = 3840 / 256
                        orange_posX = np.random.uniform(256, 0)
                        orange_posY = slope * orange_posX
                        if orange_posX > 3000:
                            orange_spawn_pos = (orange_posX, orange_posY, posZ)
                        else:
                            orange_spawn_pos = (0, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            elif spawn_pos == 3:
                # print("Back Left")
                # SAME POS
                if same_pos == 1:
                    slope = -3840 / 256
                    posX = np.random.uniform(256, 0)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 500)

                    if posY < -3000:
                        blue_spawn_pos = (posX, posY, posZ)
                        orange_spawn_pos = (-abs(posX), abs(posY), posZ)
                    else:
                        blue_spawn_pos = (posX, 0, 17)
                        orange_spawn_pos = (0, abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -3840 / 256
                        blue_posX = np.random.uniform(256, 0)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        if blue_posY < -3000:
                            blue_spawn_pos = (blue_posX, blue_posY, posZ)
                        else:
                            blue_spawn_pos = (0, blue_posY, posZ)

                        slope = 3840 / -256
                        orange_posX = np.random.uniform(-256, 0)
                        orange_posY = slope * orange_posX
                        posZ = np.random.uniform(17, 500)
                        if orange_posX > 3000:
                            orange_spawn_pos = (orange_posX, orange_posY, posZ)
                        else:
                            orange_spawn_pos = (0, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        #  print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            elif spawn_pos == 4:
                # print("Far Back Center")
                # SAME POS
                if same_pos == 1:
                    posX = 0
                    posY = np.random.uniform(-4608, 0)
                    posZ = np.random.uniform(17, 500)
                    blue_spawn_pos = (posX, posY, posZ)
                    orange_spawn_pos = (posX, abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        blue_posX = 0
                        blue_posY = np.random.uniform(-4608, 0)
                        posZ = np.random.uniform(17, 500)
                        blue_spawn_pos = (blue_posX, blue_posY, posZ)

                        orange_posX = 0
                        orange_posY = np.random.uniform(4608, 0)
                        posZ = np.random.uniform(17, 500)
                        orange_spawn_pos = (orange_posX, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)


class NaNState(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(
            x=1943.54,
            y=-776.12,
            z=406.78,
        )
        vel = (122.79422, -234.43246, 259.7227)
        state_wrapper.ball.set_lin_vel(*vel)

        ang_vel = (-0.11868675, 0.10540164, -0.05797875)
        state_wrapper.ball.set_ang_vel(*ang_vel)

        for car in state_wrapper.cars:

            if car.team_num == 0:
                car.set_pos(
                    x=1943.54,
                    y=-776.12,
                    z=406.78,
                )

                vel = (-303.3699951171875, -214.94998168945312, -410.7699890136719)
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = (-0.91367, -1.7385, 0.28524998)
                car.set_ang_vel(*ang_vel)
                car.boost = 0.593655776977539
            else:
                car.set_pos(
                    x=1917.4401,
                    y=-738.27997,
                    z=462.01,
                )

                vel = (142.98796579, 83.07732574, -20.63784965)
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = (-0.22647299, 0.10713767, 0.24121591)
                car.set_ang_vel(*ang_vel)
                car.boost = 0.8792996978759766


class SideHighRoll(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        sidepick = random.randrange(2)

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1

        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        ball_x_pos = 3000 * side_inverter
        ball_y_pos = random.randrange(1500) - 750
        ball_z_pos = BALL_RADIUS
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (2000 + random.randrange(1000) - 500) * side_inverter
        ball_y_vel = random.randrange(1500) - 750
        ball_z_vel = random.randrange(300)
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        wall_car_blue = [car for car in state_wrapper.cars if car.team_num == 0][0]

        # blue car setup
        blue_pitch_rot = 0 * DEG_TO_RAD
        blue_yaw_rot = 90 * DEG_TO_RAD
        blue_roll_rot = 90 * side_inverter * DEG_TO_RAD
        wall_car_blue.set_rot(blue_pitch_rot, blue_yaw_rot, blue_roll_rot)

        blue_x = 4096 * side_inverter
        blue_y = -2500 + (random.randrange(500) - 250)
        blue_z = 600 + (random.randrange(400) - 200)
        wall_car_blue.set_pos(blue_x, blue_y, blue_z)
        wall_car_blue.boost = 100

        # orange car setup
        wall_car_orange = None
        if len(state_wrapper.cars) > 1:
            wall_car_orange = [car for car in state_wrapper.cars if car.team_num == 1][0]
            # orange car setup
            orange_pitch_rot = 0 * DEG_TO_RAD
            orange_yaw_rot = -90 * DEG_TO_RAD
            orange_roll_rot = -90 * side_inverter * DEG_TO_RAD
            wall_car_orange.set_rot(orange_pitch_rot, orange_yaw_rot, orange_roll_rot)

            orange_x = 4096 * side_inverter
            orange_y = 2500 + (random.randrange(500) - 250)
            orange_z = 400 + (random.randrange(400) - 200)
            wall_car_orange.set_pos(orange_x, orange_y, orange_z)
            wall_car_orange.boost = 100

        for car in state_wrapper.cars:
            if len(state_wrapper.cars) == 1 or car is wall_car_orange or car is wall_car_blue:
                continue

            # set all other cars randomly in the field
            car.set_pos(random.randrange(2944) - 1472, random.randrange(3968) - 1984, 0)
            car.set_rot(0, (random.randrange(360) - 180) * (3.1415927 / 180), 0)


class SaveShot(StateSetter):
    def __init__(self, ball_speed=1500):
        self.ball_speed = ball_speed
        self.rng = np.random.default_rng()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                if random.uniform(True, False):
                    car_x = -900
                    car_rot_yaw = 0
                else:
                    car_x = 900
                    car_rot_yaw = 180 * DEG_TO_RAD
                car_y = -5030
                car_z = 17
                car.set_pos(car_x, car_y, car_z)

                car_rot_pitch = 0
                car_rot_roll = 0
                car.set_rot(car_rot_pitch, car_rot_yaw, car_rot_roll)

                car.set_lin_vel(0.1, 0, 0)
                car.set_ang_vel(0.1, 0, 0)

                car.boost = self.rng.uniform(0.3, 0.4)

            else:
                car.set_pos(
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

        ball_x = self.rng.uniform(-800, 800)
        ball_y = -10240 / 2 + 2500
        ball_z = self.rng.uniform(500, 700)
        state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)

        ball_vel_x = 0
        ball_vel_y = -self.ball_speed * 1.5
        ball_vel_z = 150
        state_wrapper.ball.set_lin_vel(ball_vel_x, ball_vel_y, ball_vel_z)
        state_wrapper.ball.set_ang_vel(0, 0, 0)


class AirdribbleSetup(StateSetter):
    def __init__(self, ball_vel_mult=1, ball_zero_z=False):
        self.ball_vel_mult = ball_vel_mult
        self.ball_zero_z = ball_zero_z
        self.rng = np.random.default_rng()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                # Randomly choose which wall to use
                if self.rng.choice([True, False]):
                    wall_x = SIDE_WALL_X - 700
                else:
                    wall_x = -SIDE_WALL_X + 700

                ball_x = wall_x - BALL_RADIUS - 10 if wall_x > 0 else wall_x + BALL_RADIUS + 10
                ball_y = self.rng.uniform(-3000, 3000)
                ball_z = 94

                # Set the position of the ball
                state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)

                # Set the ball velocity towards the goal at a small angle
                ball_vel_x = self.rng.uniform(1000, 1200) * (-1 if wall_x < 0 else 1)
                ball_vel_y = self.rng.uniform(50, 500) * (-1 if ball_y > 0 else 1)
                state_wrapper.ball.set_lin_vel(ball_vel_x, ball_vel_y, 0)
                state_wrapper.ball.set_ang_vel(0, 0, 0)

                # Set the position of the car behind the ball
                car_x = ball_x - 500 * (-1 if wall_x < 0 else 1)
                car_y = ball_y - ball_vel_y * 0.5
                car_z = 17

                # Set the car's rotation to face the wall and towards the goal at a small angle
                car_rot_yaw = 0 if ball_y == 0 else -math.atan2(ball_vel_y, ball_vel_x)
                car_rot_pitch = 0
                car_rot_roll = 0

                car.set_pos(car_x, car_y, car_z)
                car.set_rot(car_rot_pitch, car_rot_yaw, car_rot_roll)

                # Set the car's linear and angular velocities to match the ball's velocity
                car.set_lin_vel(ball_vel_x, ball_vel_y, 0)
                car.set_ang_vel(0, 0.1, 0)

                # Set the boost amount for the car
                car.boost = self.rng.uniform(.45, 1.000001)
            else:
                car.set_pos(
                    self.rng.uniform(-2900, 2900),
                    self.rng.uniform(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=self.rng.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = self.rng.uniform(0, 1)


class ShortGoalRoll(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        if len(state_wrapper.cars) > 1:
            defense_team = random.randrange(2)
        else:
            defense_team = 0
        sidepick = random.randrange(2)

        defense_inverter = 1
        if defense_team == 0:
            # change side
            defense_inverter = -1

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1

        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        x_random = random.randrange(446)
        ball_x_pos = (-2850 + x_random) * side_inverter
        ball_y_pos = (5120 - BALL_RADIUS) * defense_inverter
        ball_z_pos = 1400 + random.randrange(400) - 200
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (1000 + random.randrange(400) - 200) * side_inverter
        ball_y_vel = 0
        ball_z_vel = 550
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        wall_car = [car for car in state_wrapper.cars if car.team_num == defense_team][0]

        wall_car_x = (2000 - random.randrange(500)) * side_inverter
        wall_car_y = 5120 * defense_inverter
        wall_car_z = 1000 + (random.randrange(500) - 500)
        wall_car.set_pos(wall_car_x, wall_car_y, wall_car_z)

        wall_pitch_rot = (0 if side_inverter == -1 else 180) * DEG_TO_RAD
        wall_yaw_rot = 0 * DEG_TO_RAD
        wall_roll_rot = -90 * defense_inverter * DEG_TO_RAD
        wall_car.set_rot(wall_pitch_rot, wall_yaw_rot, wall_roll_rot)
        wall_car.boost = 25

        if len(state_wrapper.cars) > 1:
            challenge_car = [car for car in state_wrapper.cars if car.team_num != defense_team][0]
            challenge_car.set_pos(0, 1000 * defense_inverter, 0)

            challenge_pitch_rot = 0 * DEG_TO_RAD
            challenge_yaw_rot = 90 * defense_inverter * DEG_TO_RAD
            challenge_roll_rot = 0 * DEG_TO_RAD
            challenge_car.set_rot(challenge_pitch_rot, challenge_yaw_rot, challenge_roll_rot)
            challenge_car.boost = 100

        for car in state_wrapper.cars:
            if len(state_wrapper.cars) == 1 or car is wall_car or car is challenge_car:
                continue

            car.set_pos(random.randrange(2944) - 1472, (-4500 + random.randrange(500) - 250) * defense_inverter, 0)
            car.set_rot(0, (random.randrange(360) - 180) * DEG_TO_RAD, 0)


class StandingBallState(StateSetter):

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(x=random.Random().randint(-SIDE_WALL_X + 500, SIDE_WALL_X - 500),
                                   y=0,
                                   z=random.Random().randint(1000, 1300))

        state_wrapper.ball.set_lin_vel(x=random.uniform(-1000, 1000),
                                       y=random.uniform(-1000, 1000),
                                       z=random.uniform(400, 800)
                                       )

        blue_team = []
        orange_team = []

        for player in state_wrapper.cars:

            x = 0
            y = 0
            yaw = 0
            if player.team_num == ORANGE_TEAM:

                y = random.uniform(1000, 2000)
                yaw = -0.5 * np.pi

                if len(orange_team) >= 1:
                    choice = random.choice(orange_team)

                    if choice.position.item(0) > 3000:
                        x = random.Random().randint(choice.position.item(0) - 500,
                                                    choice.position.item(0) - 200)

                    elif choice.position.item(0) < -3000:
                        x = random.Random().randint(choice.position.item(0) + 200,
                                                    choice.position.item(0) + 500)
                    else:
                        x = random.Random().randint(choice.position.item(0) - 500,
                                                    choice.position.item(0) + 500)

                else:
                    x = random.Random().randint(state_wrapper.ball.position.item(0) - 300,
                                                state_wrapper.ball.position.item(0) + 300)

                orange_team.append(player)
            else:
                y = random.uniform(-2000, -1000)
                yaw = 0.5 * np.pi

                if len(blue_team) >= 1:
                    choice = random.choice(blue_team)

                    if choice.position.item(0) > 3000:
                        x = random.Random().randint(choice.position.item(0) - 500,
                                                    choice.position.item(0) - 200)

                    elif choice.position.item(0) < -3000:
                        x = random.Random().randint(choice.position.item(0) + 200,
                                                    choice.position.item(0) + 500)
                    else:
                        x = random.Random().randint(choice.position.item(0) - 500,
                                                    choice.position.item(0) + 500)

                else:
                    x = random.Random().randint(
                        state_wrapper.ball.position.item(0) - 300,
                        state_wrapper.ball.position.item(0) + 300)

            player.set_pos(
                x=x,
                y=y,
                z=30)
            player.set_rot(yaw=yaw)
            player.boost = .85


class AerialBallState(StateSetter):
    def __init__(self):
        super().__init__()
        self.observators = []

    def reset(self, state_wrapper: StateWrapper):

        if len(self.observators) != 0:
            distances = []
            for player in state_wrapper.cars:
                dist = math.sqrt((state_wrapper.ball.position.item(0) - player.position.item(0)) ** 2 +
                                 (state_wrapper.ball.position.item(1) - player.position.item(1)) ** 2 +
                                 (state_wrapper.ball.position.item(2) - player.position.item(2)) ** 2)
                distances.append(dist)

            distance = np.mean(distances)

            for obs in self.observators:
                obs.update(distance=float(distance))

        xthreshold = 1800
        ythreshold = 1200
        zthreshold = 500

        # Ball position
        state_wrapper.ball.set_pos(x=np.random.randint(-SIDE_WALL_X + xthreshold,
                                                       SIDE_WALL_X - xthreshold),

                                   y=np.random.randint(-BACK_WALL_Y + ythreshold,
                                                       BACK_WALL_Y - ythreshold),

                                   z=np.random.randint(zthreshold,
                                                       CEILING_Z - 2 * zthreshold)
                                   )

        # Ball speed
        state_wrapper.ball.set_lin_vel(x=np.random.randint(500, 1000),
                                       y=np.random.randint(500, 1000),
                                       z=np.random.randint(500, 1000)
                                       )

        for player in state_wrapper.cars:
            player.set_pos(x=np.random.randint(-SIDE_WALL_X + xthreshold,
                                               SIDE_WALL_X - xthreshold),
                           y=np.random.randint(-BACK_WALL_Y + ythreshold,
                                               BACK_WALL_Y - ythreshold),
                           z=30)

            player.boost = .85

            # player.set_rot(pitch=random.Random().random() * np.pi,
            #                yaw=random.Random().random() * np.pi,
            #                roll=random.Random().random() * np.pi
            #                )
            #
            # player.set_lin_vel(x=np.random.randint(0, constants.CAR_MAX_SPEED),
            #                    y=np.random.randint(0, constants.CAR_MAX_SPEED),
            #                    z=np.random.randint(0, constants.CAR_MAX_SPEED)
            #                    )

    '''def bind_to(self, ball_register: BallDistRegisterer):
        self.observators.append(ball_register)'''


def probs_function(number):
    if number < 0.5:
        return 0
    else:
        return -4 * pow(number + 0.5, 2) + 1


class RecoverySetter(StateSetter):
    def __init__(self, zero_boost_weight=0, zero_ball_vel_weight=0, ball_vel_mult=1, ball_zero_z=False):
        self.ball_zero_z = ball_zero_z
        self.ball_vel_mult = ball_vel_mult
        self.zero_boost_weight = zero_boost_weight
        self.zero_ball_vel_weight = zero_ball_vel_weight
        self.rng = np.random.default_rng()
        self.big_boosts = [BOOST_LOCATIONS[i] for i in [3, 4, 15, 18, 29, 30]]
        self.big_boosts = np.asarray(self.big_boosts)
        self.big_boosts[:, -1] = 18
        # self.end_object_tracker = end_object_tracker

    def reset(self, state_wrapper: StateWrapper):

        if self.rng.uniform() > self.zero_boost_weight:
            boost = self.rng.uniform(0, 1.000001)
        else:
            boost = 0

        for car in state_wrapper.cars:
            car.set_pos(*random_valid_loc())
            car.set_rot(self.rng.uniform(-np.pi / 2, np.pi / 2), self.rng.uniform(-np.pi, np.pi),
                        self.rng.uniform(-np.pi, np.pi))
            car.set_lin_vel(self.rng.uniform(-2000, 2000), self.rng.uniform(-2000, 2000), self.rng.uniform(-2000, 2000))
            car.set_ang_vel(self.rng.uniform(-4, 4), self.rng.uniform(-4, 4), self.rng.uniform(-4, 4))
            car.boost = boost

        # if self.end_object_tracker is not None and self.end_object_tracker[0] != 0:
        loc = random_valid_loc()
        state_wrapper.ball.set_pos(x=loc[0], y=loc[1], z=94)
        if self.rng.uniform() > self.zero_ball_vel_weight:
            state_wrapper.ball.set_lin_vel(self.ball_vel_mult * self.rng.uniform(-200, 200),
                                           self.ball_vel_mult * self.rng.uniform(-200, 200),
                                           0 if self.zero_ball_vel_weight else self.rng.uniform(-200, 200))
        else:
            state_wrapper.ball.set_lin_vel(0, 0, 0)
        state_wrapper.ball.set_ang_vel(0, 0, 0)


class HalfFlip(StateSetter):
    def __init__(self, zero_boost_weight=0, zero_ball_vel_weight=0, ball_vel_mult=1, ball_zero_z=False):
        self.ball_zero_z = ball_zero_z
        self.ball_vel_mult = ball_vel_mult
        self.zero_ball_vel_weight = zero_ball_vel_weight
        self.zero_boost_weight = zero_boost_weight
        self.rng = np.random.default_rng()

    def reset(self, state_wrapper: StateWrapper):
        assert len(state_wrapper.cars) < 3
        zero_ball_vel = True
        if self.rng.uniform() > self.zero_ball_vel_weight:
            zero_ball_vel = False
        y = 0
        x = self.rng.uniform(-1500, 1500)
        state_wrapper.ball.set_pos(x, y, 94)
        if zero_ball_vel:
            state_wrapper.ball.set_lin_vel(0, 0, 0)
        else:
            state_wrapper.ball.set_lin_vel(self.ball_vel_mult * self.rng.uniform(-600, 600) if y == 0 and x != 0 else 0,
                                           self.ball_vel_mult * self.rng.uniform(-600, 600) if x == 0 and y != 0 else 0,
                                           0 if self.zero_ball_vel_weight else self.rng.uniform(-200, 200))
        state_wrapper.ball.set_ang_vel(0, 0, 0)
        if self.rng.uniform() > self.zero_boost_weight:
            boost = self.rng.uniform(0, 1.000001)
        else:
            boost = 0
        for car in state_wrapper.cars:
            if car.id == 1:
                car.set_pos(x, y - 2500)
                car.set_rot(0, (-np.pi * 0.5) + self.rng.uniform(-0.04, 0.04) * np.pi, 0)
                car.set_lin_vel(0, 0, 0)
                car.set_ang_vel(0, 0, 0)
            else:
                values = mirror(state_wrapper.cars[0], x, y)
                # values_pos = [*values.pos]
                # values_pos[0] += 100  # stop dropping on top of each other
                car.set_pos(*values.pos)
                car.set_rot(*values.rot)
                car.set_lin_vel(*values.lin_vel)
                car.set_ang_vel(*values.ang_vel)
            car.boost = boost


class Wavedash(StateSetter):
    def __init__(self, zero_boost_weight=0, zero_ball_vel_weight=0, ball_vel_mult=1, ball_zero_z=False):
        self.ball_zero_z = ball_zero_z
        self.ball_vel_mult = ball_vel_mult
        self.zero_boost_weight = zero_boost_weight
        self.zero_ball_vel_weight = zero_ball_vel_weight
        self.rng = np.random.default_rng()

    def reset(self, state_wrapper: StateWrapper):
        assert len(state_wrapper.cars) < 3
        zero_ball_vel = True
        if self.rng.uniform() > self.zero_ball_vel_weight:
            zero_ball_vel = False
        y = 0
        x = self.rng.uniform(-1500, 1500)
        state_wrapper.ball.set_pos(x, y, 94)
        if zero_ball_vel:
            state_wrapper.ball.set_lin_vel(0, 0, 0)
        else:
            state_wrapper.ball.set_lin_vel(self.ball_vel_mult * self.rng.uniform(-600, 600) if y == 0 and x != 0 else 0,
                                           self.ball_vel_mult * self.rng.uniform(-600, 600) if x == 0 and y != 0 else 0,
                                           0 if self.zero_ball_vel_weight else self.rng.uniform(-200, 200))
        state_wrapper.ball.set_ang_vel(0, 0, 0)
        if self.rng.uniform() > self.zero_boost_weight:
            boost = self.rng.uniform(0, 1.000001)
        else:
            boost = 0
        for car in state_wrapper.cars:
            if car.id == 1:
                car.set_pos(x + self.rng.uniform(-500, 500), y - 2500, self.rng.uniform(50, 350))
                car.set_rot(0, np.pi * 0.5, 0)
                car.set_lin_vel(0, 0, 0)
                car.set_ang_vel(0, 0, 0)
            else:
                values = mirror(state_wrapper.cars[0], x, y)
                # values_pos = [*values.pos]
                # values_pos[0] += 100  # stop dropping on top of each other
                car.set_pos(*values.pos)
                car.set_rot(*values.rot)
                car.set_lin_vel(*values.lin_vel)
                car.set_ang_vel(*values.ang_vel)
            car.boost = boost


class Chaindash(StateSetter):
    def __init__(self, zero_boost_weight=0, zero_ball_vel_weight=0, ball_vel_mult=1, ball_zero_z=False):
        self.ball_zero_z = ball_zero_z
        self.ball_vel_mult = ball_vel_mult
        self.zero_boost_weight = zero_boost_weight
        self.zero_ball_vel_weight = zero_ball_vel_weight
        self.rng = np.random.default_rng()

    def reset(self, state_wrapper: StateWrapper):
        assert len(state_wrapper.cars) < 3
        zero_ball_vel = True
        if self.rng.uniform() > self.zero_ball_vel_weight:
            zero_ball_vel = False
        if self.rng.choice([False, True]):
            y = self.rng.uniform(-1500, 1500)
            x = 0
        else:
            y = 0
            x = self.rng.uniform(-1500, 1500)
        state_wrapper.ball.set_pos(x, y, 94)
        if zero_ball_vel:
            state_wrapper.ball.set_lin_vel(0, 0, 0)
        else:
            state_wrapper.ball.set_lin_vel(self.ball_vel_mult * self.rng.uniform(-600, 600) if y == 0 and x != 0 else 0,
                                           self.ball_vel_mult * self.rng.uniform(-600, 600) if x == 0 and y != 0 else 0,
                                           0 if self.zero_ball_vel_weight else self.rng.uniform(-200, 200))
        state_wrapper.ball.set_ang_vel(0, 0, 0)
        if self.rng.uniform() > self.zero_boost_weight:
            boost = self.rng.uniform(0, 1.000001)
        else:
            boost = 0
        for car in state_wrapper.cars:
            if car.id == 1:
                car.set_pos(x + self.rng.uniform(-500, 500), max(-3900, y - self.rng.uniform(3000, 5000)),
                            self.rng.uniform(50, 350))
                car.set_rot(self.rng.uniform(-np.pi / 8, np.pi / 8),
                            self.rng.uniform(-np.pi, np.pi),
                            self.rng.uniform(-np.pi / 8, np.pi / 8))
                ball_sign = -1 if state_wrapper.cars[0].position[1] - y > 0 else 1
                car.set_lin_vel(self.rng.uniform(-50, 50),
                                ball_sign * self.rng.uniform(600, 2000),
                                self.rng.uniform(-50, 1))
                car.set_ang_vel(self.rng.uniform(-1, 1), self.rng.uniform(-1, 1), self.rng.uniform(-1, 1))
            else:
                values = mirror(state_wrapper.cars[0], x, y)
                car.set_pos(*values.pos)
                car.set_rot(*values.rot)
                car.set_lin_vel(*values.lin_vel)
                car.set_ang_vel(*values.ang_vel)
            car.boost = boost


class RandomEvenRecovery(StateSetter):
    def __init__(self, zero_boost_weight=0, zero_ball_vel_weight=0, ball_vel_mult=1, ball_zero_z=False):
        self.ball_zero_z = ball_zero_z
        self.ball_vel_mult = ball_vel_mult
        self.zero_boost_weight = zero_boost_weight
        self.zero_ball_vel_weight = zero_ball_vel_weight
        self.rng = np.random.default_rng()

    def reset(self, state_wrapper: StateWrapper):
        assert len(state_wrapper.cars) < 3
        zero_ball_vel = True
        if self.rng.uniform() > self.zero_ball_vel_weight:
            zero_ball_vel = False
        if self.rng.choice([False, True]):
            y = self.rng.uniform(-1500, 1500)
            x = 0
        else:
            y = 0
            x = self.rng.uniform(-1500, 1500)
        if y >= 0:
            ball_sign = 1
        else:
            ball_sign = -1
        state_wrapper.ball.set_pos(x, y, 94)
        if zero_ball_vel:
            state_wrapper.ball.set_lin_vel(0, 0, 0)
        else:
            state_wrapper.ball.set_lin_vel(self.ball_vel_mult * self.rng.uniform(-600, 600) if y == 0 and x != 0 else 0,
                                           self.ball_vel_mult * self.rng.uniform(-600, 600) if x == 0 and y != 0 else 0,
                                           0 if self.zero_ball_vel_weight else self.rng.uniform(-200, 200))
        state_wrapper.ball.set_ang_vel(0, 0, 0)
        if self.rng.uniform() > self.zero_boost_weight:
            boost = self.rng.uniform(0, 1.000001)
        else:
            boost = 0
        for car in state_wrapper.cars:
            if car.id == 1:
                car.set_pos(self.rng.uniform(-1000, 1000), y - 2500, self.rng.uniform(50, 350))
                car.set_rot(self.rng.uniform(-np.pi / 2, np.pi / 2),
                            self.rng.uniform(-np.pi, np.pi),
                            self.rng.uniform(-np.pi / 2, np.pi / 2))
                car.set_lin_vel(self.rng.uniform(-1500, 1500),
                                ball_sign * self.rng.uniform(-1500, 1500),
                                self.rng.uniform(-50, -1))
                car.set_ang_vel(self.rng.uniform(-4, 4), self.rng.uniform(-4, 4), self.rng.uniform(-4, 4))
            else:
                values = mirror(state_wrapper.cars[0], x, y)
                car.set_pos(*values.pos)
                car.set_rot(*values.rot)
                car.set_lin_vel(*values.lin_vel)
                car.set_ang_vel(*values.ang_vel)
            car.boost = boost


class Curvedash(StateSetter):
    def __init__(self, zero_boost_weight=0, zero_ball_vel_weight=0, ball_vel_mult=1, ball_zero_z=False):
        self.ball_zero_z = ball_zero_z
        self.ball_vel_mult = ball_vel_mult
        self.zero_boost_weight = zero_boost_weight
        self.zero_ball_vel_weight = zero_ball_vel_weight
        self.rng = np.random.default_rng()

    def reset(self, state_wrapper: StateWrapper):
        assert len(state_wrapper.cars) < 3
        zero_ball_vel = True
        if self.rng.uniform() > self.zero_ball_vel_weight:
            zero_ball_vel = False
        if self.rng.choice([False, True]):
            ball_y = self.rng.uniform(-3000, 3000)
            ball_x = 0
        else:
            ball_x = self.rng.uniform(-2500, 2500)
            ball_y = 0
        state_wrapper.ball.set_pos(ball_x, ball_y, 94)
        if zero_ball_vel:
            state_wrapper.ball.set_lin_vel(0, 0, 0)
        else:
            state_wrapper.ball.set_lin_vel(
                self.ball_vel_mult * self.rng.uniform(-600, 600) if ball_y == 0 and ball_x != 0 else 0,
                self.ball_vel_mult * self.rng.uniform(-600, 600) if ball_x == 0 and ball_y != 0 else 0,
                0 if self.zero_ball_vel_weight else self.rng.uniform(-200, 200))
        state_wrapper.ball.set_ang_vel(0, 0, 0)
        if self.rng.uniform() > self.zero_boost_weight:
            boost = self.rng.uniform(0, 1.000001)
        else:
            boost = 0
        for car in state_wrapper.cars:
            if car.id == 1:
                neg = self.rng.choice([-1, 1])
                car.set_pos(neg * (SIDE_WALL_X - 17),
                            ball_y - (neg * self.rng.uniform(0, 1000)),
                            self.rng.uniform(600, 1000))
                car.set_rot((-90 + self.rng.uniform(-30, 30)) * DEG_TO_RAD, 90 * DEG_TO_RAD, 90 * DEG_TO_RAD * neg)
                car.set_lin_vel(0, 0, -self.rng.uniform(300, 1000))
                car.set_ang_vel(0, 0, 0)
            else:
                values = mirror(state_wrapper.cars[0], ball_x, ball_y)
                car.set_pos(*values.pos)
                car.set_rot(*values.rot)
                car.set_lin_vel(*values.lin_vel)
                car.set_ang_vel(*values.ang_vel)
            car.boost = boost


class Walldash(StateSetter):
    def __init__(self, zero_boost_weight=0, zero_ball_vel_weight=0, ball_vel_mult=1, ball_zero_z=False,
                 end_object: PhysicsObject = None,
                 location: str = None,
                 min_car_vel=0,
                 max_car_vel=CAR_MAX_SPEED,
                 ):

        self.max_car_vel = max_car_vel
        self.min_car_vel = min_car_vel
        assert self.min_car_vel < self.max_car_vel
        self.ball_zero_z = ball_zero_z
        self.ball_vel_mult = ball_vel_mult
        self.zero_boost_weight = zero_boost_weight
        self.zero_ball_vel_weight = zero_ball_vel_weight
        self.rng = np.random.default_rng()
        self.end_object = end_object
        self.location = location

    def reset(self, state_wrapper: StateWrapper):
        assert len(state_wrapper.cars) < 3
        zero_ball_vel = True
        if self.location is None:
            if self.rng.uniform() > self.zero_ball_vel_weight:
                zero_ball_vel = False
            ball_x = 0
            ball_y = self.rng.uniform(-2500, 2500)
            state_wrapper.ball.set_pos(ball_x, ball_y, 94)
            if zero_ball_vel:
                state_wrapper.ball.set_lin_vel(0, 0, 0)
            else:
                state_wrapper.ball.set_lin_vel(0,
                                               self.ball_vel_mult * self.rng.uniform(-600, 600) if ball_y != 0 else 0,
                                               0 if self.ball_zero_z else self.rng.uniform(-200, 200))
            state_wrapper.ball.set_ang_vel(0, 0, 0)
            if ball_y >= 0:
                ball_sign = 1
            else:
                ball_sign = -1
        else:
            ball_y = 0
            ball_x = 0
            ball_sign = 1
            state_wrapper.ball.set_pos(0, 0, 94)
        if self.rng.uniform() > self.zero_boost_weight:
            boost = self.rng.uniform(0, 1.000001)
        else:
            boost = 0
        for car in state_wrapper.cars:
            if car.id == 1:
                neg = self.rng.choice([-1, 1])
                if self.location is None:
                    car.set_pos(neg * (SIDE_WALL_X - 17),
                                ball_y - (ball_sign * self.rng.uniform(800, 1500)),
                                self.rng.uniform(300, 1700))
                    car.set_rot(self.rng.uniform(-30, 30) * DEG_TO_RAD, 90 * DEG_TO_RAD, 90 * DEG_TO_RAD * neg)
                    car.set_lin_vel(0, ball_sign * self.rng.uniform(300, 1000), 0)
                    car.set_ang_vel(0, 0, 0)
                elif self.location == "90":
                    #object_y = self.rng.choice([-1, 1])
                    x = neg * (SIDE_WALL_X - 17)
                    y = self.rng.uniform(-3500, 3500)
                    car.set_pos(x,
                                y,
                                self.rng.uniform(300, 600),
                                )
                    car.set_rot((90 + self.rng.uniform(-10, 10)) * DEG_TO_RAD, 90 * DEG_TO_RAD, 90 * DEG_TO_RAD * neg)
                    car.set_lin_vel(0, 0, self.rng.uniform(200, 600))
                    set_pos(end_object=self.end_object, x=x, y=y, z=1750)
                elif self.location == "45":
                    object_y = self.rng.choice([-1, 1])
                    object_pos_45 = self.rng.choice([False, True])
                    dist_yz = 2300
                    x = neg * (SIDE_WALL_X - 17)
                    y = self.rng.uniform(-3500, 2000) * object_y
                    z = self.rng.uniform(300, 700) if object_pos_45 else self.rng.uniform(1350, 1750)
                    car.set_pos(x,
                                y,
                                z,
                                )
                    if object_pos_45:
                        pitch_mod = object_y
                    else:
                        pitch_mod = -object_y
                    car.set_rot(
                        ((180 if object_y == -1 else 0) + (45 * pitch_mod) + self.rng.uniform(-10, 10)) * DEG_TO_RAD,
                        90 * DEG_TO_RAD,
                        90 * DEG_TO_RAD * neg)
                    speed = self.rng.uniform(self.min_car_vel, self.max_car_vel)
                    car.set_lin_vel(0, speed * object_y * 0.707, speed * 0.707 * (1 if object_pos_45 else -1))
                    set_pos(end_object=self.end_object, x=x, y=y + (dist_yz * object_y * 0.707),
                            z=z + (dist_yz * 0.5 * (1 if object_pos_45 else -1)))
                elif self.location == "same_z":
                    object_y = self.rng.choice([-1, 1])
                    dist_yz = 2400
                    x = neg * (SIDE_WALL_X - 17)
                    y = self.rng.uniform(-3500, 1500) * object_y
                    z = self.rng.uniform(300, 1600)
                    car.set_pos(x,
                                y,
                                z,
                                )
                    car.set_rot(((180 if object_y == -1 else 0) + self.rng.uniform(-10, 10)) * DEG_TO_RAD,
                                90 * DEG_TO_RAD,
                                90 * DEG_TO_RAD * neg)
                    speed = self.rng.uniform(self.min_car_vel, self.max_car_vel)
                    car.set_lin_vel(0, speed * object_y, 0)
                    set_pos(end_object=self.end_object, x=x, y=y + (dist_yz * object_y), z=z)
                elif self.location == "ball":
                    object_y = self.rng.choice([-1, 1])
                    zero_ball_vel = True
                    dist_yz = 2400
                    if self.rng.uniform() > self.zero_ball_vel_weight:
                        zero_ball_vel = False
                    ball_x = neg * (SIDE_WALL_X - BALL_RADIUS)
                    ball_y = self.rng.uniform(-1500, 3500) * object_y
                    ball_z = self.rng.uniform(300, 1700)
                    state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)
                    if zero_ball_vel:
                        state_wrapper.ball.set_lin_vel(0, 0, 0)
                    else:
                        state_wrapper.ball.set_lin_vel(0,
                                                       self.ball_vel_mult * self.rng.uniform(-600,
                                                                                             600),
                                                       self.ball_vel_mult * self.rng.uniform(-200, 200))
                    state_wrapper.ball.set_ang_vel(0, 0, 0)
                    x = neg * (SIDE_WALL_X - 17)
                    y = ball_y - (dist_yz * object_y)
                    z = self.rng.uniform(300, 1600)
                    car.set_pos(x,
                                y,
                                z,
                                )
                    car.set_rot(((180 if object_y == -1 else 0) + self.rng.uniform(-30, 30)) * DEG_TO_RAD,
                                90 * DEG_TO_RAD,
                                90 * DEG_TO_RAD * neg)
                    speed = self.rng.uniform(self.min_car_vel, self.max_car_vel)
                    car.set_lin_vel(0, speed * object_y, 0)
                    if self.end_object is not None:
                        set_pos(end_object=self.end_object, x=-1, y=-1, z=-1)
                elif self.location == "back_boost":
                    object_y = self.rng.choice([-1, 1])
                    dist_yz = 3000
                    ball_x = neg * 3072
                    ball_y = 4096 * object_y
                    ball_z = 17
                    state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)
                    state_wrapper.ball.set_lin_vel(0, 0, 0)
                    state_wrapper.ball.set_ang_vel(0, 0, 0)
                    x = neg * (SIDE_WALL_X - 17)
                    y = ball_y - (dist_yz * object_y)
                    z = self.rng.uniform(300, 1600)
                    car.set_pos(x,
                                y,
                                z,
                                )
                    car.set_rot(((180 if object_y == -1 else 0) + self.rng.uniform(-30, 30)) * DEG_TO_RAD,
                                90 * DEG_TO_RAD,
                                90 * DEG_TO_RAD * neg)
                    speed = self.rng.uniform(self.min_car_vel, self.max_car_vel)
                    car.set_lin_vel(0, speed * object_y, 0)
                    if self.end_object is not None:
                        set_pos(end_object=self.end_object, x=-1, y=-1, z=-1)
            else:
                values = mirror(state_wrapper.cars[0], ball_x, ball_y)
                car.set_pos(*values.pos)
                car.set_rot(*values.rot)
                car.set_lin_vel(*values.lin_vel)
                car.set_ang_vel(*values.ang_vel)
            car.boost = boost


all_states = [DefaultState()]


# DynamicScoredReplaySetter(
#         "../replays/states_scores_duels.npz",
#         "../replays/states_scores_doubles.npz",
#         "../replays/states_scores_standard.npz"
#     )

class ProbabilisticStateSetter(StateSetter):

    def __init__(self, states, probs, verbose: int = 1, training: bool = True):
        self.old_state = None
        self.verbose = verbose
        self.training = training
        self.states, self.probs = states, probs

    def reset(self, state_wrapper: StateWrapper):

        if self.training:
            selected_state = random.choices(self.states, weights=self.probs, k=1)[0]
        else:
            selected_state = all_states.pop(0)

        self.old_state = selected_state

        # while not _check_positions(state_wrapper):
        selected_state.reset(state_wrapper)
