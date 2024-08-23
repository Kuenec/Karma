import numpy as np # Import numpy, the python math library
from rlgym_sim.utils import RewardFunction # Import the base RewardFunction class
from rlgym_sim.utils.gamestates import GameState, PlayerData # Import game state stuff
import math
from typing import Optional, Tuple, Union
from rlgym_sim.utils.common_values import CEILING_Z, BACK_WALL_Y, CAR_MAX_SPEED, BLUE_TEAM, ORANGE_TEAM, BLUE_GOAL_BACK

from bubo_misc_utils import distance, distance2D

#Important Values for rewards:

KPH_TO_VEL = 250/9
RAMP_HEIGHT = 256
BALL_RADIUS = 92.75


class TouchBallRewardScaledByHitForce(RewardFunction): #rarely used this, i reccomend using this or either touchvelchange at the start when making a bot
    def __init__(self):
        super().__init__()
        self.max_hit_speed = 130 * KPH_TO_VEL
        self.last_ball_vel = None
        self.cur_ball_vel = None

    # game reset, after terminal condition
    def reset(self, initial_state: GameState):
        self.last_ball_vel = initial_state.ball.linear_velocity
        self.cur_ball_vel = initial_state.ball.linear_velocity

    # happens 
    def pre_step(self, state: GameState):
        self.last_ball_vel = self.cur_ball_vel
        self.cur_ball_vel = state.ball.linear_velocity

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            reward = np.linalg.norm(self.cur_ball_vel - self.last_ball_vel) / self.max_hit_speed
            return reward
        return 0
    
class GoodVelocityPlayerToBallReward(RewardFunction):
    def __init__(self, use_scalar_projection=False):
        super().__init__()
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 92.75 = 24.8
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / CAR_MAX_SPEED
            reward = float(np.dot(norm_pos_diff, norm_vel))
            if reward > 0:
                return reward
            else:
                return 0
    

class TouchVelChange(RewardFunction): #this is an altered version of TouchVelChange, I have added a threshold so that if the difference is less than 500uu then it gets no reward
    def __init__(self, threshold=500):
        self.last_vel = np.zeros(3)
        self.prev_vel = np.zeros(3)
        self.vel_difference = 0
        self.threshold = threshold

    def reset(self, initial_state: GameState):
        self.last_vel = np.zeros(3)
        self.prev_vel = np.zeros(3)
        self.vel_difference = 0

    def pre_step(self, state: GameState):
        self.prev_vel = self.last_vel
        self.last_vel = state.ball.linear_velocity
        self.vel_difference = np.linalg.norm(self.prev_vel - self.last_vel)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched and self.vel_difference > self.threshold:
            reward = self.vel_difference / 4600.0
        return reward
    

    
def distance2D(x: np.array, y: np.array) -> float:
    return distance(np.array([x[0], x[1], 0]), np.array([y[0], y[1], 0]))

class CradleReward(RewardFunction): #dont reccomend this either
    def __init__(self, minimum_barrier: float = 200):
        super().__init__()
        self.min_distance = minimum_barrier

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            player.car_data.position[2] < state.ball.position[2]
            and (BALL_RADIUS + 20 < state.ball.position[2] < BALL_RADIUS + 200)
            and distance2D(player.car_data.position, state.ball.position) <= 170
        ):
            if (
                abs(state.ball.position[0]) < 3946
                and abs(state.ball.position[1]) < 4970
            ):  # side and back wall values - 150
                if self.min_distance > 0:
                    for _player in state.players:
                        if (
                            _player.team_num != player.team_num
                            and distance(_player.car_data.position, state.ball.position)
                            < self.min_distance
                        ):
                            return 0

                return 1

        return 0
    
class GroundedReward(RewardFunction):
    def __init__(
        self,
    ):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return player.on_ground is True

    

class CradleFlickReward(RewardFunction): #do not reccomend using this
    def __init__(
        self,
        minimum_barrier: float = 400,
        max_vel_diff: float = 400,
        training: bool = True,
    ):
        super().__init__()
        self.min_distance = minimum_barrier
        self.max_vel_diff = max_vel_diff
        self.training = training
        self.cradle_reward = CradleReward(minimum_barrier=0)

    def reset(self, initial_state: GameState):
        self.cradle_reward.reset(initial_state)

    def stable_carry(self, player: PlayerData, state: GameState) -> bool:
        if BALL_RADIUS + 20 < state.ball.position[2] < BALL_RADIUS + 80:
            if (
                abs(
                    np.linalg.norm(
                        player.car_data.linear_velocity - state.ball.linear_velocity
                    )
                )
                <= self.max_vel_diff
            ):
                return True
        return False

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = self.cradle_reward.get_reward(player, state, previous_action) * 0.5
        if reward > 0:
            if not self.training:
                reward = 0
            stable = self.stable_carry(player, state)
            challenged = False
            for _player in state.players:
                if (
                    _player.team_num != player.team_num
                    and distance(_player.car_data.position, state.ball.position)
                    < self.min_distance
                ):
                    challenged = True
                    break
            if challenged:
                if stable:
                    if player.on_ground:
                        return reward - 0.5
                    else:
                        if player.has_flip:
                            # small reward for jumping
                            return reward + 2
                        else:
                            # print("PLAYER FLICKED!!!")
                            # big reward for flicking
                            return reward + 5
            else:
                if stable:
                    return reward + 1

        return reward
    
class GroundDribbleReward(RewardFunction):
    def __init__(self, distance_threshold: float = 162.5, ball_height_lower_threshold: float = 110.0):
        super().__init__()
        self.DISTANCE_THRESHOLD = distance_threshold
        self.BALL_HEIGHT_LOWER_THRESHOLD = ball_height_lower_threshold

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action):
        reward = 0

        player_pos = player.car_data.position
        ball_pos = state.ball.position

        dist_to_ball = np.linalg.norm(ball_pos - player_pos)

        ball_past_lower_threshold = ball_pos[2] >= self.BALL_HEIGHT_LOWER_THRESHOLD
        player_close_to_ground = player.on_ground

        if dist_to_ball > self.DISTANCE_THRESHOLD or not ball_past_lower_threshold or not player_close_to_ground:
            return reward

        reward = 1.0

        return reward



    
class LemTouchBallReward(RewardFunction):
    def init(self, aerial_weight=0):
        self.aerial_weight = aerial_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            if not player.on_ground and player.car_data.position[2] >= 256:
                height_reward = np.log1p(player.car_data.position[2] - 256)
                return height_reward
        return 0


class SwiftGroundDribbleReward(RewardFunction): 
    def __init__(self):
        super().__init__()

        self.MIN_BALL_HEIGHT = 109.0
        self.MAX_BALL_HEIGHT = 180.0
        self.MAX_DISTANCE = 197.0
        self.COEFF = 2.0

    def reset(self, initial_state : GameState):
        pass

    def get_reward(self, player : PlayerData, state : GameState, previous_action):
        if player.on_ground and state.ball.position[2] >= self.MIN_BALL_HEIGHT \
            and state.ball.position[2] <= self.MAX_BALL_HEIGHT and np.linalg.norm(player.car_data.position - state.ball.position) < self.MAX_DISTANCE:
            
            player_speed = np.linalg.norm(player.car_data.linear_velocity)
            ball_speed = np.linalg.norm(state.ball.linear_velocity)
            player_speed_normalized = player_speed / CAR_MAX_SPEED
            inverse_difference = 1.0 - abs(player_speed - ball_speed)
            twosum = player_speed + ball_speed
            speed_reward = player_speed_normalized + self.COEFF * (inverse_difference / twosum)

            return speed_reward

        return 0.0 #originally was 0

class JumpTouchReward(RewardFunction):
    def __init__(self, min_height=92.75):
        self.min_height = min_height
        self.max_height = 2044-92.75
        self.range = self.max_height - self.min_height

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return (state.ball.position[2] - self.min_height) / self.range

        return 0
    

class RetreatReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.defense_target = np.array(BLUE_GOAL_BACK)

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            ball = state.ball.position
            pos = player.car_data.position
            vel = player.car_data.linear_velocity
        else:
            ball = state.inverted_ball.position
            pos = player.inverted_car_data.position
            vel = player.inverted_car_data.linear_velocity

        reward = 0.0
        if ball[1]+200 < pos[1]:
            pos_diff = self.defense_target - pos
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / CAR_MAX_SPEED
            reward = np.dot(norm_pos_diff, norm_vel)
        return reward
    
class DistanceReward(RewardFunction):
    def __init__(self, dist_max=1000, max_reward=2):
        super().__init__()
        self.dist_max = dist_max

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        difference = state.ball.position - player.car_data.position
        distance = (
            math.sqrt(
                difference[0] * difference[0]
                + difference[1] * difference[1]
                + difference[2] * difference[2]
            )
            - 110
        )

        if distance > self.dist_max:
            return 0

        return 1 - (distance / self.dist_max)
    
    RAMP_HEIGHT = 256


class AerialDistanceReward(RewardFunction):
    def __init__(self, height_scale: float, distance_scale: float):
        super().__init__()
        self.height_scale = height_scale
        self.distance_scale = distance_scale

        self.current_car: Optional[PlayerData] = None
        self.prev_state: Optional[GameState] = None
        self.ball_distance: float = 0
        self.car_distance: float = 0

    def reset(self, initial_state: GameState):
        self.current_car = None
        self.prev_state = initial_state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = 0
        is_current = self.current_car is not None and self.current_car.car_id == player.car_id
        # Test if player is on the ground
        if player.car_data.position[2] < RAMP_HEIGHT:
            if is_current:
                is_current = False
                self.current_car = None
        # First non ground touch detection
        elif player.ball_touched and not is_current:
            is_current = True
            self.ball_distance = 0
            self.car_distance = 0
            rew = self.height_scale * max(player.car_data.position[2] + state.ball.position[2] - 2 * RAMP_HEIGHT, 0)
        # Still off the ground after a touch, add distance and reward for more touches
        elif is_current:
            self.car_distance += np.linalg.norm(player.car_data.position - self.current_car.car_data.position)
            self.ball_distance += np.linalg.norm(state.ball.position - self.prev_state.ball.position)
            if player.ball_touched:
                rew = self.distance_scale * (self.car_distance + self.ball_distance)
                self.car_distance = 0
                self.ball_distance = 0
        if is_current:
            self.current_car = player  # Update to get latest physics info
        self.prev_state = state
        return rew / (2 * BACK_WALL_Y)


class AirTouchReward(RewardFunction): #I've never used this reward ever throughout my entire time training this bot I have no clue if this works or not
    def __init__(self, max_time_in_air=1.75) -> None:
        super().__init__()
        self.max_time_in_air = max_time_in_air
        self.air_time = 0
        self.was_in_air = False

    def reset(self, initial_state: GameState) -> None:
        self.air_time = 0
        self.was_in_air = False

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.on_ground:
            self.was_in_air = False
        else:
            if not self.was_in_air:
                self.was_in_air = True
                self.air_time += 0.1
        
        height_frac = state.ball.position[2] / CEILING_Z
        air_time_frac = min(self.air_time, self.max_time_in_air) / self.max_time_in_air
        reward = min(air_time_frac, height_frac)
        return reward
    
    

class InAirReward(RewardFunction): # We extend the class "RewardFunction
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array
        
        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0
        
class LogCombinedReward(RewardFunction):
    """
    A reward composed of multiple rewards.
    """

    def __init__(
            self,
            reward_functions: Tuple[RewardFunction, ...],
            reward_weights: Optional[Tuple[float, ...]] = None
    ):
        """
        Creates the combined reward using multiple rewards, and a potential set
        of weights for each reward.

        :param reward_functions: Each individual reward function.
        :param reward_weights: The weights for each reward.
        """
        super().__init__()
        self.prev_rewards = None
        self.reward_functions = reward_functions
        self.reward_weights = reward_weights or np.ones_like(reward_functions)

        if len(self.reward_functions) != len(self.reward_weights):
            raise ValueError(
                ("Reward functions list length ({0}) and reward weights " \
                 "length ({1}) must be equal").format(
                    len(self.reward_functions), len(self.reward_weights)
                )
            )

    @classmethod
    def from_zipped(cls, *rewards_and_weights: Union[RewardFunction, Tuple[RewardFunction, float]]) -> "LogCombinedReward":
        """
        Alternate constructor which takes any number of either rewards, or (reward, weight) tuples.

        :param rewards_and_weights: a sequence of RewardFunction or (RewardFunction, weight) tuples
        """
        rewards = []
        weights = []
        for value in rewards_and_weights:
            if isinstance(value, tuple):
                r, w = value
            else:
                r, w = value, 1.
            rewards.append(r)
            weights.append(w)
        return cls(tuple(rewards), tuple(weights))

    def reset(self, initial_state: GameState) -> None:
        """
        Resets underlying reward functions.

        :param initial_state: The initial state of the reset environment.
        """
        for func in self.reward_functions:
            func.reset(initial_state)

    def pre_step(self, state: GameState) -> None:
        for func in self.reward_functions:
            if hasattr(func, 'pre_step'):
                func.pre_step(state)

    def get_reward(
            self,
            player: PlayerData,
            state: GameState, 
            previous_action: np.ndarray
    ) -> float:
        """
        Returns the reward for a player on the terminal state.

        :param player: Player to compute the reward for.
        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.

        :return: The combined rewards for the player on the state.
        """
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_functions
        ]
        self.prev_rewards = rewards

        return float(np.dot(self.reward_weights, rewards))

    def get_final_reward(
            self,
            player: PlayerData,
            state: GameState, 
            previous_action: np.ndarray
    ) -> float:
        """
        Returns the reward for a player on the terminal state.

        :param player: Player to compute the reward for.
        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.

        :return: The combined rewards for the player on the state.
        """
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        return float(np.dot(self.reward_weights, rewards))


        
    
    # reward.py
class KickoffProximityReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Check if the ball is at the kickoff position
        if state.ball.position[0] == 0 and state.ball.position[1] == 0:
            player_pos = np.array(player.car_data.position)
            ball_pos = np.array(state.ball.position)
            player_dist_to_ball = np.linalg.norm(player_pos - ball_pos)

            opponent_distances = []
            for p in state.players:
                if p.team_num != player.team_num:
                    opponent_pos = np.array(p.car_data.position)
                    opponent_dist_to_ball = np.linalg.norm(opponent_pos - ball_pos)
                    opponent_distances.append(opponent_dist_to_ball)

            # Reward if the player is closest to the ball, otherwise penalize
            if opponent_distances and player_dist_to_ball < min(opponent_distances):
                return 1
            else:
                return -1
        return 0
    
class CradleFlickReward(RewardFunction): #really do not reccomend using this as your bot will learn to flick on its own if your dribble reward is zerosum
    def __init__(
        self,
        minimum_barrier: float = 400,
        max_vel_diff: float = 400,
        training: bool = True,
    ):
        super().__init__()
        self.min_distance = minimum_barrier
        self.max_vel_diff = max_vel_diff
        self.training = training
        self.cradle_reward = CradleReward(minimum_barrier=0)

    def reset(self, initial_state: GameState):
        self.cradle_reward.reset(initial_state)

    def stable_carry(self, player: PlayerData, state: GameState) -> bool:
        if BALL_RADIUS + 20 < state.ball.position[2] < BALL_RADIUS + 80:
            if (
                abs(
                    np.linalg.norm(
                        player.car_data.linear_velocity - state.ball.linear_velocity
                    )
                )
                <= self.max_vel_diff
            ):
                return True
        return False

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = self.cradle_reward.get_reward(player, state, previous_action) * 0.5
        if reward > 0:
            if not self.training:
                reward = 0
            stable = self.stable_carry(player, state)
            challenged = False
            for _player in state.players:
                if (
                    _player.team_num != player.team_num
                    and distance(_player.car_data.position, state.ball.position)
                    < self.min_distance
                ):
                    challenged = True
                    break
            if challenged:
                if stable:
                    if player.on_ground:
                        return reward - 0.5
                    else:
                        if player.has_flip:
                            # small reward for jumping
                            return reward + 2
                        else:
                            # print("PLAYER FLICKED!!!")
                            # big reward for flicking
                            return reward + 5
            else:
                if stable:
                    return reward + 1

        return reward

class ZeroSumReward(RewardFunction):
    '''
    child_reward: The underlying reward function
    team_spirit: How much to share this reward with teammates (0-1)
    opp_scale: How to scale the penalty we get for the opponents getting this reward (usually 1)
    '''
    def __init__(self, child_reward: RewardFunction, team_spirit, opp_scale = 1.0):
        self.child_reward = child_reward # type: RewardFunction
        self.team_spirit = team_spirit
        self.opp_scale = opp_scale

        self._update_next = True
        self._rewards_cache = {}

    def reset(self, initial_state: GameState):
        self.child_reward.reset(initial_state)

    def pre_step(self, state: GameState):
        self.child_reward.pre_step(state)

        # Mark the next get_reward call as being the first reward call of the step
        self._update_next = True

    def update(self, state: GameState, is_final):
        self._rewards_cache = {}

        '''
        Each player's reward is calculated using this equation:
        reward = individual_rewards * (1-team_spirit) + avg_team_reward * team_spirit - avg_opp_reward * opp_scale
        '''

        # Get the individual rewards from each player while also adding them to that team's reward list
        individual_rewards = {}
        team_reward_lists = [ [], [] ]
        for player in state.players:
            if is_final:
                reward = self.child_reward.get_final_reward(player, state, None)
            else:
                reward = self.child_reward.get_reward(player, state, None)
            individual_rewards[player.car_id] = reward
            team_reward_lists[int(player.team_num)].append(reward)

        # If a team has no players, add a single 0 to their team rewards so the average doesn't break
        for i in range(2):
            if len(team_reward_lists[i]) == 0:
                team_reward_lists[i].append(0)

        # Turn the team-sorted reward lists into averages for each time
        # Example:
        #    Before: team_rewards = [ [1, 3], [4, 8] ]
        #    After:  team_rewards = [ 2, 6 ]
        team_rewards = np.average(team_reward_lists, 1)

        # Compute and cache:
        # reward = individual_rewards * (1-team_spirit)
        #          + avg_team_reward * team_spirit
        #          - avg_opp_reward * opp_scale
        for player in state.players:
            self._rewards_cache[player.car_id] = (
                    individual_rewards[player.car_id] * (1 - self.team_spirit)
                    + team_rewards[int(player.team_num)] * self.team_spirit
                    - team_rewards[1 - int(player.team_num)] * self.opp_scale
            )

    '''
    I made get_reward and get_final_reward both call get_reward_multi, using the "is_final" argument to distinguish
    Otherwise I would have to rewrite this function for final rewards, which is lame
    '''
    def get_reward_multi(self, player: PlayerData, state: GameState, previous_action: np.ndarray, is_final) -> float:
        # If this is the first get_reward call this step, we need to update the rewards for all players
        if self._update_next:
            self.update(state, is_final)
            self._update_next = False
        return self._rewards_cache[player.car_id]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward_multi(player, state, previous_action, False)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward_multi(player, state, previous_action, True)
