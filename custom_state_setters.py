from dataclasses import dataclass
from subprocess import NORMAL_PRIORITY_CLASS
from tkinter import CallWrapper
from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters.wrappers import CarWrapper
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z
import numpy as np
import random as rnd
import math
from rlgym_sim.utils.terminal_conditions import TerminalCondition
from rlgym_sim.utils.gamestates import GameState
from scipy.stats import truncnorm


def get_truncated_normal(mean=0.0, sd=1.0, low=0.0, upp=10.0):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class BallTouchedCondition(TerminalCondition):

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        """
        Return `True` if the last touch does not have the same ID as the last touch from the initial state.
        """
        if any(current_state.ball.position[0:1] != [0.0, 0.0]) or any(p.ball_touched for p in current_state.players):
            return True
        else:
            return False

@dataclass
class CarData:
    car_x: float
    car_y: float
    car_z: float
    car_yaw: float
    car_boost: float

class BackwardsStateSetter(StateSetter):

    def reset(self, state_wrapper: StateWrapper):

        state_wrapper.ball.position = np.array([0.0, 0.0, 92.75])

        car_data = CarData(3457.8198, -1.1, 17.01, 0.0, 100.0)

        car = state_wrapper.cars[0]

        car.set_pos(car_data.car_x, car_data.car_y, car_data.car_z)
        car.set_rot(yaw = car_data.car_yaw)
        car.boost = car_data.car_boost

class RandomMirrorSetterOneVOne(StateSetter):
    X_RANGE = [1024.0, 3500.0]
    Y_RANGE = [-3700.0, -1024.0]
    YAW_RANGE = [0, 2 * math.pi]

    @staticmethod
    def rnd_sign():
        return 1 if rnd.random() > 0.5 else -1

    def reset(self, state_wrapper: StateWrapper):
        
        state_wrapper.ball.position = np.array([0.0, 0.0, 92.75])
        
        car_boost = 100

        #car_location = [3457.8198, -1.1, 17.01]

        car_x = rnd.uniform(*self.X_RANGE) * self.rnd_sign()
        car_y = rnd.uniform(*self.Y_RANGE)
        car_z = 17.01 # At rest height for octane
        car_yaw = rnd.uniform(*self.YAW_RANGE)

        for car in state_wrapper.cars:

            if car.team_num == ORANGE_TEAM:
                car_x = -car_x
                car_y = -car_y
                car_yaw = -car_yaw

            car.set_pos(car_x, car_y, car_z)
            car.set_rot(yaw = car_yaw)
            car.boost = car_boost

class RandomMirrorSetterFlexible(StateSetter):
    X_RANGE = [1024.0, 3500.0]
    Y_RANGE = [-3700.0, -1024.0]
    YAW_RANGE = [0, 2 * math.pi]
    
    DISTANCE = 3500.0
    MAX_THETA = math.pi
    
    @staticmethod
    def rnd_sign():
        return 1 if rnd.random() > 0.5 else -1

    @staticmethod
    def gen_car_location():
        car_x = rnd.uniform(*RandomMirrorSetterFlexible.X_RANGE) * RandomMirrorSetterFlexible.rnd_sign()
        car_y = rnd.uniform(*RandomMirrorSetterFlexible.Y_RANGE)
        car_z = 17.01
        car_boost = 100
        car_yaw = rnd.uniform(*RandomMirrorSetterFlexible.YAW_RANGE)
        ball_pos = [0.0, 0.0, 92.75]

        return CarData(car_x, car_y, car_z, car_yaw, car_boost)

    @staticmethod
    def gen_car_loc_fixed_distance(idx = 0, num_cars = 1):
        rho = RandomMirrorSetterFlexible.DISTANCE

        theta_partition = RandomMirrorSetterFlexible.MAX_THETA / num_cars
        theta = rnd.uniform(0, theta_partition) + (idx * theta_partition)
        
        car_x = rho * np.cos(theta)
        car_y = -(rho * np.sin(theta))
        car_z = 17.01
        car_boost = 100

        #yaw_theta = 2 * math.pi - theta
        car_yaw = rnd.uniform(*RandomMirrorSetterFlexible.YAW_RANGE)
        
        # center the yaw range to the car->ball angle, then normalize to [0, 2pi]
        #car_yaw = ((car_yaw + yaw_theta - (3/2 * math.pi)) + 2 * math.pi) % (2 * math.pi)

        return CarData(car_x, car_y, car_z, car_yaw, car_boost)

    @staticmethod
    def gen_cal_loc_fixed_distance_and_yaw(idx = 0, num_cars = 1, relative_yaw = 0.0):
        rho = RandomMirrorSetterFlexible.DISTANCE

        theta_partition = RandomMirrorSetterFlexible.MAX_THETA / num_cars
        theta = rnd.uniform(0, theta_partition) + (idx * theta_partition)
        
        car_x = rho * np.cos(theta)
        car_y = -(rho * np.sin(theta))
        car_z = 17.01
        car_boost = 100

        car_ball_angle = 2 * math.pi - theta
        car_yaw = car_ball_angle - math.pi + relative_yaw
        car_yaw = ((car_yaw + 2*math.pi) % (2*math.pi)) # normalize to [0, 2pi]

        return CarData(car_x, car_y, car_z, car_yaw, car_boost)
    
    @staticmethod
    def mirror_car_data(car_data: CarData):
        return CarData(-car_data.car_x, -car_data.car_y, car_data.car_z, -car_data.car_yaw, car_data.car_boost)

    @staticmethod
    def mirror_car_data_rel_yaw(car_data: CarData, relative_yaw):
        #mirror_yaw = ((2*relative_yaw - car_data.car_yaw) + 2*math.pi) % (2*math.pi) # normalize to [0, 2pi]
        mirror_yaw = (car_data.car_yaw + math.pi) % (2*math.pi) # normalize to [0, 2pi]
        return CarData(-car_data.car_x, -car_data.car_y, car_data.car_z, mirror_yaw, car_data.car_boost)

    @staticmethod
    def apply_car_data(car: CarWrapper, car_data: CarData):
        car.set_pos(car_data.car_x, car_data.car_y, car_data.car_z)
        car.set_rot(yaw = car_data.car_yaw)
        car.boost = car_data.car_boost

    def reset(self, state_wrapper: StateWrapper):
        blue_cars = state_wrapper.blue_cars()
        orange_cars = state_wrapper.orange_cars()

        num_blue= len(blue_cars)
        num_orange = len(orange_cars)

        state_wrapper.ball.position = np.array([0.0, 0.0, 92.75])

        assert num_blue == num_orange or num_blue == 1 , "Number of blue and orange cars must be equal, or there must be only one blue car"
        cars = state_wrapper.cars
        
        # rel_car_yaw = rnd.uniform(*RandomMirrorSetterFlexible.YAW_RANGE)

        # generate a random yaw using a normal distribution centered around 180 degrees (facing opposite the ball)
        rel_car_yaw = get_truncated_normal(mean = math.pi, sd = math.pi / 6, low = 0.0, upp = 2 * math.pi).rvs()

        if len(cars) == 1:
            car_data = self.gen_cal_loc_fixed_distance_and_yaw(0, 1, rel_car_yaw)
            self.apply_car_data(cars[0], car_data)
        else:

            for i in range(num_blue):
                car_data = self.gen_cal_loc_fixed_distance_and_yaw(i, num_blue, rel_car_yaw)
                
                self.apply_car_data(blue_cars[i], car_data)
                self.apply_car_data(orange_cars[i], self.mirror_car_data_rel_yaw(car_data, rel_car_yaw))