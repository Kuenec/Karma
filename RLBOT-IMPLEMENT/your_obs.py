import math
import numpy as np
from typing import Any, List
from rlgym_compat import common_values
from rlgym_compat.game_state import GameState
from rlgym_compat.player_data import PlayerData

"""
The observation builder.
"""

from abc import ABC, abstractmethod
import gym
import numpy as np
from typing import Any


class ObsBuilder(ABC):

    def __init__(self):
        pass

    def get_obs_space(self) -> gym.spaces.Space:
        """
        Function that returns the observation space type. It will be called during the initialization of the environment.

        :return: The type of the observation space
        """
        pass

    @abstractmethod
    def reset(self, initial_state: GameState):
        """
        Function to be called each time the environment is reset. Note that this does not need to return anything,
        the environment will call `build_obs` automatically after reset, so the initial observation for a policy will be
        constructed in the same way as every other observation.

        :param initial_state: The initial game state of the reset environment.
        """
        raise NotImplementedError

    def pre_step(self, state: GameState):
        """
        Function to pre-compute values each step. This function is called only once each step, before build_obs is
        called for each player.

        :param state: The current state of the game.
        """
        pass

    @abstractmethod
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        """
        Function to build observations for a policy. This is where all observations will be constructed every step and
        every reset. This function is given a player argument, and it is expected that the observation returned by this
        function will contain information from the perspective of that player. This function is called once for each
        agent automatically at every step.

        :param player: The player to build an observation for. The observation returned should be from the perspective of
        this player.

        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.

        :return: An observation for the player provided.
        """
        raise NotImplementedError


class DefaultObs(ObsBuilder):
    def __init__(self, pos_coef=1/2300, ang_coef=1/math.pi, lin_vel_coef=1/2300, ang_vel_coef=1/math.pi):
        """
        :param pos_coef: Position normalization coefficient
        :param ang_coef: Rotation angle normalization coefficient
        :param lin_vel_coef: Linear velocity normalization coefficient
        :param ang_vel_coef: Angular velocity normalization coefficient
        """
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position * self.POS_COEF,
               ball.linear_velocity * self.LIN_VEL_COEF,
               ball.angular_velocity * self.ANG_VEL_COEF,
               previous_action,
               pads]

        self._add_player_to_obs(obs, player, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            self._add_player_to_obs(team_obs, other, inverted)

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        obs.extend([
            player_car.position * self.POS_COEF,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity * self.LIN_VEL_COEF,
            player_car.angular_velocity * self.ANG_VEL_COEF,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])

        return player_car