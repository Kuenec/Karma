#There is Probably some duplicates of some rewards in this file some may be exactly the same reward or some may have been ones i altered such as touchvell change with the threshold
import math
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import FaceBallReward
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils import math as rl_math
from bubo_misc_utils import *
import numpy as np
from rlgym_sim.utils.reward_functions import CombinedReward
from typing import Optional, Tuple, Union
from rlgym_sim.utils.common_values import CEILING_Z, BACK_WALL_Y, CAR_MAX_SPEED, BLUE_TEAM, ORANGE_TEAM, BLUE_GOAL_BACK, SUPERSONIC_THRESHOLD

SIDE_WALL_X = 4096  # +/-
BACK_WALL_Y = 5120  # +/-
CEILING_Z = 2044
BACK_NET_Y = 6000  # +/-

GOAL_HEIGHT = 642.775

ORANGE_GOAL_CENTER = (0, BACK_WALL_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_CENTER = (0, -BACK_WALL_Y, GOAL_HEIGHT / 2)

# Often more useful than center
ORANGE_GOAL_BACK = (0, BACK_NET_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_BACK = (0, -BACK_NET_Y, GOAL_HEIGHT / 2)

# ORANGE_GOAL_WAY_BACK = (0, 8000, GOAL_HEIGHT / 2)
# BLUE_GOAL_WAY_BACK = (0, -8000, GOAL_HEIGHT / 2)
# [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
BALL_RADIUS = 92.75

BALL_MAX_SPEED = 6000
CAR_MAX_SPEED = 2300
SUPERSONIC_THRESHOLD = 2200
CAR_MAX_ANG_VEL = 5.5
KPH_TO_VEL = 250/9
RAMP_HEIGHT = 256
BLUE_TEAM = 0
ORANGE_TEAM = 1
NUM_ACTIONS = 8

BOOST_LOCATIONS = (
    (0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0),
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (-940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0),
    (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0),
    (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0),
    (-940.0, 3310.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0),
)

def scalar_speed_towards_ball(player :PlayerData, state: GameState) -> float:
    vel = player.car_data.linear_velocity
    pos_diff = state.ball.position - player.car_data.position
    
    norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
    scalar_vel_towards_ball = float(np.dot(norm_pos_diff, vel))
    return scalar_vel_towards_ball

class TouchBallRewardScaledByHitForce(RewardFunction):
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
    

class TouchVelChange(RewardFunction):
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

class CradleReward(RewardFunction):
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

    

class CradleFlickReward(RewardFunction):
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


class GoalSpeedAndPlacementReward(RewardFunction):
    def __init__(self):
        self.prev_score_blue = 0
        self.prev_score_orange = 0
        self.prev_state_blue = None
        self.prev_state_orange = None
        self.min_height = BALL_RADIUS + 10
        self.height_reward = 1.75

    def reset(self, initial_state: GameState):
        self.prev_score_blue = initial_state.blue_score
        self.prev_score_orange = initial_state.orange_score
        self.prev_state_blue = initial_state
        self.prev_state_orange = initial_state

    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            score = state.blue_score
            # check to see if goal scored
            if score > self.prev_score_blue:
                reward = np.linalg.norm(self.prev_state_blue.ball.linear_velocity) / BALL_MAX_SPEED
                if self.prev_state_blue.ball.position[2] > self.min_height:
                    reward = self.height_reward * reward
                self.prev_state_blue = state
                self.prev_score_blue = score
                return reward
            else:
                self.prev_state_blue = state
                return 0.0
        else:
            score = state.orange_score
            # check to see if goal scored
            if score > self.prev_score_orange:
                reward = np.linalg.norm(self.prev_state_orange.ball.linear_velocity) / BALL_MAX_SPEED
                if self.prev_state_orange.ball.position[2] > self.min_height:
                    reward = self.height_reward * reward
                self.prev_state_orange = state
                self.prev_score_orange = score
                return reward
            else:
                self.prev_state_orange = state
                return 0.0
            
class SpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        linear_velocity = player.car_data.linear_velocity
        reward = float(math.vecmag(linear_velocity))
        
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


class QuickestTouchReward(RewardFunction):
    def __init__(self, timeout=0, tick_skip=8, num_agents=1):
        self.timeout = timeout * 120 # convert to ticks
        self.tick_skip = tick_skip
        self.num_agents = num_agents

    def reset(self, initial_state: GameState):
        self.timer = 0
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.timer += self.tick_skip
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.timer = self.timer / self.num_agents
        if player.ball_touched:
            #print("Touched ball after steps: ", self.timer)
            reward = ((self.timeout - self.timer) / self.timeout) * 100 + 30 
        else:
            norm_elapsed_time = self.timer / self.timeout
            player_ball_distance = float(np.linalg.norm(state.ball.position - player.car_data.position))
            norm_distance = player_ball_distance / 3500.0 # starting distance is 3500
            reward = -((norm_elapsed_time * 100) + (norm_distance * 100)) - 30

        # print("QuickestTouchReward.FinalReward(): ", reward)
        return reward

class SustainedVelocityPlayerToBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.comulative_reward = 0

    def reset(self, initial_state: GameState):
        self.comulative_reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        scalar_vel_towards_ball = float(np.dot(norm_pos_diff, vel))

        norm_scalar_vel_towards_ball = scalar_vel_towards_ball / SUPERSONIC_THRESHOLD

        if norm_scalar_vel_towards_ball > 0 and norm_scalar_vel_towards_ball < 1:
            reward = (scalar_vel_towards_ball / CAR_MAX_SPEED) / 50
        else:
            if norm_scalar_vel_towards_ball < 0:
                reward = (scalar_vel_towards_ball / CAR_MAX_SPEED) * 4
            else:
                reward = (scalar_vel_towards_ball / CAR_MAX_SPEED)

        self.comulative_reward += reward
        return reward
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # print("VelocityPlayerToBall: Cumulative: ", self.comulative_reward)
        return 0
class AccelerationPlayerToBallReward(RewardFunction):
    def __init__(self, tick_skip=8):
        super().__init__()
        self.tick_skip = tick_skip
        self.CAR_MAX_ACC = (CAR_MAX_SPEED ) / self.tick_skip
        self.cumulative_reward = 0

    def reset(self, initial_state: GameState):
        self.previous_velocity = 0.0
        self.cumulative_reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        scalar_vel_towards_ball = float(np.dot(norm_pos_diff, vel))

        acc_towards_ball = (scalar_vel_towards_ball - self.previous_velocity) / self.tick_skip

        self.previous_velocity = scalar_vel_towards_ball

        if acc_towards_ball >= 0:
            reward = (acc_towards_ball / self.CAR_MAX_ACC) * 0
        else:
            norm_acc_towards_ball = (acc_towards_ball / self.CAR_MAX_ACC) * 1
            reward = norm_acc_towards_ball
        
        self.cumulative_reward += reward
        return reward
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # print("AccelerationPlayerToBall: Cumulative: ", self.cumulative_reward)
        return 0

class SpeedOnBallTouchReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.cumulative_reward = 0
    
    def reset(self, initial_state: GameState):
        self.cumulative_reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched==True:
            speed_to_ball = scalar_speed_towards_ball(player, state)
            reward = ((speed_to_ball / SUPERSONIC_THRESHOLD) - 1) * 20
        else: 
            reward = 0

        self.cumulative_reward += reward
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # print("SpeedOnBallTouchReward: Cumulative: ", self.cumulative_reward)
        return 0

class FaceBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.cumulative_reward = 0

    def reset(self, initial_state: GameState):
        self.cumulative_reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        reward = float(np.dot(player.car_data.forward(), norm_pos_diff)) / 100

        self.cumulative_reward += reward
        return reward
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # print("FaceBallReward: Cumulative: ", self.cumulative_reward)
        return 0

class OnGroundReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.cumulative_reward = 0

    def reset(self, initial_state: GameState):
        self.cumulative_reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.on_ground:
            reward = 0.7
        else:
            reward = -1
        reward = reward / 100
        self.cumulative_reward += reward
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # print("OnGroundReward: Cumulative: ", self.cumulative_reward)
        return 0
    
    

class AirTouchReward(RewardFunction):
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
    
    

class InAirReward(RewardFunction): # We extend the class "RewardFunction"
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

class VelocityPlayerToNearestPlayerReward(RewardFunction):
    def __init__(self, use_scalar_projection=False):
        super().__init__()
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        team = player.team_num
        
        players = state.players
        playerList = []
        playerListDistance = []
        for i in players:
            if i.team_num != team:
                x = i.car_data.position
                y = player.car_data.position
                playerList.append(i.car_data.position)
                playerListDistance.append((np.sum((x*x - y*y)**2, axis=1)))
        nearestPlayer = playerList[playerListDistance.index(min(playerListDistance))]
        pos_diff = nearestPlayer - player.car_data.position
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
            return float(np.dot(norm_pos_diff, norm_vel))
        
    
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
    

class StarterReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.goal_reward = 10
        self.boost_weight = 0.1
        self.rew = CombinedReward(
            (
                EventReward(team_goal=self.goal_reward, concede=-self.goal_reward, demo=self.goal_reward/3),
                TouchVelChange(),
                VelocityBallToGoalReward(),
                VelocityPlayerToBallReward(),
                JumpTouchReward(min_height=120),
                KickoffReward(boost_punish=False)
            ),
            (1.0, 1.5, 0.075, 0.075, 2.0, 0.1))
            #(1.0, 1.0, 0.1, 2.0, 0.3334, 0.05))

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.rew.get_reward(player, state, previous_action)


class LavaFloorReward(RewardFunction):
    @staticmethod
    def get_reward(player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        if player.on_ground and player.car_data.position[2] < 50:
            return -1
        return 0

    @staticmethod
    def reset(initial_state: GameState):
        pass

class EventReward(RewardFunction):
    def __init__(self, goal=0., team_goal=0., concede=-0., touch=0., shot=0., save=0., demo=0., boost_pickup=0.):
        """
        :param goal: reward for goal scored by player.
        :param team_goal: reward for goal scored by player's team.
        :param concede: reward for goal scored by opponents. Should be negative if used as punishment.
        :param touch: reward for touching the ball.
        :param shot: reward for shooting the ball (as detected by Rocket League).
        :param save: reward for saving the ball (as detected by Rocket League).
        :param demo: reward for demolishing a player.
        :param boost_pickup: reward for picking up boost. big pad = +1.0 boost, small pad = +0.12 boost.
        """
        super().__init__()
        self.weights = np.array([goal, team_goal, concede, touch, shot, save, demo, boost_pickup])

        # Need to keep track of last registered value to detect changes
        self.last_registered_values = {}

    @staticmethod
    def _extract_values(player: PlayerData, state: GameState):
        if player.team_num == BLUE_TEAM:
            team, opponent = state.blue_score, state.orange_score
        else:
            team, opponent = state.orange_score, state.blue_score

        return np.array([player.match_goals, team, opponent, player.ball_touched, player.match_shots,
                         player.match_saves, player.match_demolishes, player.boost_amount])

    def reset(self, initial_state: GameState, optional_data=None):
        # Update every reset since rocket league may crash and be restarted with clean values
        self.last_registered_values = {}
        for player in initial_state.players:
            self.last_registered_values[player.car_id] = self._extract_values(player, initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        old_values = self.last_registered_values[player.car_id]
        new_values = self._extract_values(player, state)

        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # We only care about increasing values

        reward = np.dot(self.weights, diff_values)

        self.last_registered_values[player.car_id] = new_values
        return reward


class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / BALL_MAX_SPEED
        return float(np.dot(norm_pos_diff, norm_vel))


class VelocityPlayerToBallReward(RewardFunction):
    def __init__(self, use_scalar_projection=False):
        super().__init__()
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / CAR_MAX_SPEED
        return float(np.dot(norm_pos_diff, norm_vel))


class RuleOnePunishment(RewardFunction):
    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.on_ground and np.linalg.norm(player.car_data.linear_velocity) < 300:
            for p in state.players:
                if (
                    p.car_id != player.car_id
                    and p.on_ground
                    and distance(player.car_data.position, p.car_data.position) < 300
                    and relative_velocity_mag(
                        player.car_data.linear_velocity, p.car_data.linear_velocity
                    )
                    < 200
                ):
                    return -1

        return 0


class DemoPunish(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.demo_statuses = [True for _ in range(64)]

    def reset(self, initial_state: GameState) -> None:
        self.demo_statuses = [True for _ in range(64)]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        if player.is_demoed and not self.demo_statuses[player.car_id]:
            reward = -1

        self.demo_statuses[player.car_id] = player.is_demoed
        return reward


class BoostAcquisitions(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.boost_reserves = 1

    def reset(self, initial_state: GameState) -> None:
        self.boost_reserves = 1

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        boost_gain = player.boost_amount - self.boost_reserves
        self.boost_reserves = player.boost_amount
        return 0 if boost_gain <= 0 else boost_gain


class LandingRecoveryReward(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.up = np.array([0, 0, 1])

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if (
            not player.on_ground
            and player.car_data.linear_velocity[2] < 0
            and player.car_data.position[2] > 250
        ):
            flattened_vel = normalize(
                np.array(
                    [
                        player.car_data.linear_velocity[0],
                        player.car_data.linear_velocity[1],
                        0,
                    ]
                )
            )
            forward = player.car_data.forward()
            flattened_forward = normalize(np.array([forward[0], forward[1], 0]))
            reward += flattened_vel.dot(flattened_forward)
            reward += self.up.dot(player.car_data.up())
            reward /= 2

        return reward


class AerialNavigation(RewardFunction):
    def __init__(
        self, ball_height_min=400, player_height_min=200, beginner=True
    ) -> None:
        super().__init__()
        self.ball_height_min = ball_height_min
        self.player_height_min = player_height_min
        self.face_reward = FaceBallReward()
        self.beginner = beginner

    def reset(self, initial_state: GameState) -> None:
        self.face_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if (
            not player.on_ground
            and state.ball.position[2]
            > self.ball_height_min
            > player.car_data.position[2]
            and player.car_data.linear_velocity[2] > 0
            and distance2D(player.car_data.position, state.ball.position)
            < state.ball.position[2] * 3
        ):
            # vel check
            ball_direction = normalize(state.ball.position - player.car_data.position)
            alignment = ball_direction.dot(normalize(player.car_data.linear_velocity))
            if self.beginner:
                reward += max(
                    0, alignment * 0.5
                )  # * (np.linalg.norm(player.car_data.linear_velocity)/2300)
                # old
                # #face check
                # face_reward = self.face_reward.get_reward(player, state, previous_action)
                # if face_reward > 0:
                #     reward += face_reward * 0.25
                # #boost check
                #     if previous_action[6] == 1 and player.boost_amount > 0:
                #         reward += face_reward

            reward += alignment * (
                np.linalg.norm(player.car_data.linear_velocity) / 2300.0
            )

        return max(reward, 0)


class AerialTraining(RewardFunction):
    def __init__(self, ball_height_min=400, player_min_height=300) -> None:
        super().__init__()
        self.vel_reward = VelocityPlayerToBallReward()
        self.ball_height_min = ball_height_min
        self.player_min_height = player_min_height

    def reset(self, initial_state: GameState) -> None:
        self.vel_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
                not player.on_ground
                and state.ball.position[2] > self.ball_height_min
                and self.player_min_height < player.car_data.position[2] < state.ball.position[2]
        ):
            divisor = max(1, distance(player.car_data.position, state.ball.position)/1000)
            return self.vel_reward.get_reward(player, state, previous_action)/divisor

        return 0


class SaveBoostReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return player.boost_amount * 0.01


class SequentialRewards(RewardFunction):
    def __init__(self, rewards: list, steps: list):
        super().__init__()
        self.rewards_list = rewards
        self.step_counts = steps
        self.step_count = 0
        self.step_index = 0
        assert len(self.rewards_list) == len(self.step_counts)

    def reset(self, initial_state: GameState):
        for rew in self.rewards_list:
            rew.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            self.step_index < len(self.step_counts)
            and self.step_count > self.step_counts[self.step_index]
        ):
            self.step_index += 1
            print(f"Switching to Reward index {self.step_index}")

        self.step_count += 1
        return self.rewards_list[self.step_index].get_reward(
            player, state, previous_action
        )


class SelectiveChaseReward(RewardFunction):
    def __init__(self, distance_req: float = 500):
        super().__init__()
        self.vel_dir_reward = VelocityPlayerToBallReward()
        self.distance_requirement = distance_req

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            distance2D(player.car_data.position, state.ball.position)
            > self.distance_requirement
        ):
            return self.vel_dir_reward.get_reward(player, state, previous_action)
        return 0


class BoostDiscipline(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return float(-previous_action[6])


class BoostTrainer(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return previous_action[6] == 0


class OmniBoostDiscipline(RewardFunction):
    def __init__(self, aerial_forgiveness=False):
        super().__init__()
        self.values = [0 for _ in range(64)]
        self.af = aerial_forgiveness

    def reset(self, initial_state: GameState):
        self.values = [0 for _ in range(64)]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        old, self.values[player.car_id] = self.values[player.car_id], player.boost_amount
        if player.on_ground or not self.af:
            return -int(self.values[player.car_id] < old)
        return 0


class ControllerModerator(RewardFunction):
    def __init__(self, index: int = 0, val: int = 0, reward: float = -1):
        super().__init__()
        self.index = index
        self.val = val
        self.reward = reward

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if previous_action[self.index] == self.val:
            return self.reward
        return 0


class MillennialKickoffReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def closest_to_ball(self, player: PlayerData, state: GameState) -> bool:
        player_dist = np.linalg.norm(player.car_data.position - state.ball.position)
        for p in state.players:
            if p.team_num == player.team_num and p.car_id != p.car_id:
                dist = np.linalg.norm(p.car_data.position - state.ball.position)
                if dist < player_dist:
                    return False

        return True

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[0] == 0 and state.ball.position[1] == 0 and self.closest_to_ball(player, state):
            return -1
        return 0


class KickoffReward(RewardFunction):
    def __init__(self, boost_punish: bool = True):
        super().__init__()
        self.vel_dir_reward = VelocityPlayerToBallReward()
        self.vel_reward = NaiveSpeedReward()
        self.boost_punish = boost_punish
        self.primed = False
        self.ticks = 0

    def reset(self, initial_state: GameState):
        self.primed = False
        self.ticks = 0
        self.vel_dir_reward.reset(initial_state)

    def closest_to_ball(self, player: PlayerData, state: GameState) -> bool:
        player_dist = distance(player.car_data.position, state.ball.position)
        for p in state.players:
            if p.team_num == player.team_num and p.car_id != p.car_id:
                dist = distance(p.car_data.position, state.ball.position)
                if dist < player_dist:
                    return False

        return True


    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if state.ball.position[0] == 0 and state.ball.position[1] == 0 and self.closest_to_ball(player, state):
            if self.ticks > 0 and self.boost_punish:

                if (
                    previous_action[6] < 1
                    and np.linalg.norm(player.car_data.linear_velocity) < 2200
                ):
                    reward -= (1 - previous_action[6]) * 0.334

                if previous_action[0] < 1:
                    reward -= (1 - previous_action[0]) * 0.334

                if previous_action[7] > 0:
                    reward -= previous_action[7] * 0.334

            reward += self.vel_reward.get_reward(player, state, previous_action)
            reward += self.vel_dir_reward.get_reward(player, state, previous_action)
        self.ticks += 1
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


class TeamSpacingReward(RewardFunction):
    def __init__(self, min_spacing: float = 1000) -> None:
        super().__init__()
        self.min_spacing = clamp(math.inf, 0.0000001, min_spacing)

    def reset(self, initial_state: GameState):
        pass

    def spacing_reward(self, player: PlayerData, state: GameState) -> float:
        reward = 0
        for p in state.players:
            if p.team_num == player.team_num and p.car_id != player.car_id and not player.is_demoed and not p.is_demoed:
                separation = distance(player.car_data.position, p.car_data.position)
                if separation < self.min_spacing:
                    reward -= 1-(separation / self.min_spacing)

        return reward

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return self.spacing_reward(player, state)


class ThreeManRewards(RewardFunction):
    def __init__(self, min_spacing: float = 1500):
        super().__init__()
        self.min_spacing = min_spacing
        self.vel_reward = VelocityBallToGoalReward()
        self.KOR = KickoffReward()

    def spacing_reward(self, player: PlayerData, state: GameState, role: int):
        reward = 0
        if role != 0:
            for p in state.players:
                if p.team_num == player.team_num and p.car_id != player.car_id:
                    separation = distance(player.car_data.position, p.car_data.position)
                    if separation < self.min_spacing:
                        reward -= 1 - (separation / self.min_spacing)
        return reward

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        player_distances = []
        for p in state.players:
            if p.team_num == player.team_num:
                player_distances.append(
                    (distance(p.car_data.position, state.ball.position), p.car_id)
                )

        role = 0
        player_distances.sort(key=lambda x: x[0])
        for count, pd in enumerate(player_distances):
            if pd[1] == player.car_id:
                role = count
                break

        reward = self.spacing_reward(player, state, role)
        if role == 1:
            reward += self.vel_reward.get_reward(player, state, previous_action)
            reward += self.KOR.get_reward(player, state, previous_action)

        return reward


class SpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        car_speed = np.linalg.norm(player.car_data.linear_velocity)
        car_dir = sign(player.car_data.forward().dot(player.car_data.linear_velocity))
        if car_dir < 0:
            car_speed /= -2300

        else:
            car_speed /= 2300
        return min(car_speed, 1)


class ForwardBiasReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return player.car_data.forward().dot(normalize(player.car_data.linear_velocity))


class FlatSpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return abs(np.linalg.norm(player.car_data.linear_velocity[:2])) / 2300


class NaiveSpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return abs(np.linalg.norm(player.car_data.linear_velocity)) / 2300

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


class CenterReward(RewardFunction):
    def __init__(self, centered_distance=1200, punish_area_exit=False, non_participation_reward=0.0):
        self.centered_distance = centered_distance
        self.punish_area_exit = punish_area_exit
        self.non_participation_reward = non_participation_reward
        self.centered = False
        self.goal_spot = np.array([0, 5120, 0])

    def reset(self, initial_state: GameState):
        self.centered = False

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball_loc = state.ball.position
        if player.team_num != 0:
            ball_loc = state.inverted_ball.position

        coord_diff = self.goal_spot - ball_loc
        ball_dist_2d = np.linalg.norm(coord_diff[:2])
        #ball_dist_2d = math.sqrt(coord_diff[0] * coord_diff[0] + coord_diff[1] * coord_diff[1])
        reward = 0

        if self.centered:
            if ball_dist_2d > self.centered_distance:
                self.centered = False
                if self.punish_area_exit:
                    reward -= 1
        else:
            if ball_dist_2d < self.centered_distance:
                self.centered = True
                if state.last_touch == player.car_id:
                    reward += 1
                else:
                    reward += self.non_participation_reward
        return reward


class ClearReward(RewardFunction):
    def __init__(self, protected_distance=1200, punish_area_entry=False, non_participation_reward=0.0):
        self.protected_distance = protected_distance
        self.punish_area_entry=punish_area_entry
        self.non_participation_reward = non_participation_reward
        self.needs_clear = False
        self.goal_spot = np.array([0, -5120, 0])

    def reset(self, initial_state: GameState):
        self.needs_clear = False

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball_loc = state.ball.position
        if player.team_num != 0:
            ball_loc = state.inverted_ball.position

        coord_diff = self.goal_spot - ball_loc
        ball_dist_2d = np.linalg.norm(coord_diff[:2])
        #ball_dist_2d = math.sqrt(coord_diff[0]*coord_diff[0] + coord_diff[1]*coord_diff[1])
        reward = 0

        if self.needs_clear:
            if ball_dist_2d > self.protected_distance:
                self.needs_clear = False
                if state.last_touch == player.car_id:
                    reward += 1
                else:
                    reward += self.non_participation_reward
        else:
            if ball_dist_2d < self.protected_distance:
                self.needs_clear = True
                if self.punish_area_entry:
                    reward -= 1
        return reward

class WallTouchReward(RewardFunction):
    def __init__(self, min_height=92, exp=0.2):
        self.min_height = min_height
        self.exp = exp
        self.max = math.inf

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and player.on_ground and state.ball.position[2] >= self.min_height:
            return (clamp(self.max, 0.0001, state.ball.position[2] - 92) ** self.exp)-1

        return 0

class TouchVelChange(RewardFunction): #this is regular touchvel change without my threshold added
    def __init__(self):
        self.last_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.last_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            vel_difference = abs(np.linalg.norm(self.last_vel - state.ball.linear_velocity))
            reward = vel_difference / 4600.0

        self.last_vel = state.ball.linear_velocity

        return reward

class HeightTouchReward(RewardFunction):
    def __init__(self, min_height=92, exp=0.2, coop_dist=0):
        super().__init__()
        self.min_height = min_height
        self.exp = exp
        self.cooperation_dist = coop_dist

    def reset(self, initial_state: GameState):
        pass

    def cooperation_detector(self, player: PlayerData, state: GameState):
        for p in state.players:
            if p.car_id != player.car_id and \
                    distance(player.car_data.position, p.car_data.position) < self.cooperation_dist:
                return True

        return False


    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            if state.ball.position[2] >= self.min_height:
                if not player.on_ground or self.cooperation_dist < 90 or not self.cooperation_detector(player, state):
                    if player.on_ground:
                        reward += clamp(5000, 0.0001, (state.ball.position[2]-92)) ** self.exp
                    else:
                        reward += clamp(500, 1, (state.ball.position[2] ** (self.exp*2)))

            elif not player.on_ground:
                reward += 1

        return reward

class ModifiedTouchReward(RewardFunction):
    def __init__(self, min_change: float = 300, min_height: float = 200, vel_scale: float = 1, touch_scale: float = 1, jump_reward: bool = False, jump_scale: float = 0.1, tick_min: int = 0):
        super().__init__()
        self.psr = PowerShotReward(min_change)
        self.min_height = min_height
        self.height_cap = 2044-92.75
        self.vel_scale = vel_scale
        self.touch_scale = touch_scale
        self.jump_reward = jump_reward
        self.jump_scale = jump_scale
        self.tick_count = 0
        self.tick_min = tick_min

    def reset(self, initial_state: GameState):
        self.psr.reset(initial_state)
        self.tick_count = 0

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        psr = self.psr.get_reward(player, state, previous_action)
        if player.ball_touched:
            if self.tick_count <= 0:
                self.tick_count = self.tick_min
                reward += abs(psr * self.vel_scale)

                if not player.on_ground:
                    if self.jump_reward:
                        reward += self.jump_scale
                        if not player.has_flip:
                            reward += self.jump_scale
                    if state.ball.position[2] > self.min_height:
                        reward += abs((state.ball.position[2]/self.height_cap) * self.touch_scale)
            else:
                self.tick_count -= 1
        else:
            self.tick_count -= 1

        return reward


class PowerShotReward(RewardFunction):
    def __init__(self, min_change: float = 300):
        super().__init__()
        self.min_change = min_change
        self.last_velocity = np.array([0, 0])

    def reset(self, initial_state: GameState):
        self.last_velocity = np.array([0, 0])

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        cur_vel = np.array(
            [state.ball.linear_velocity[0], state.ball.linear_velocity[1]]
        )
        if player.ball_touched:
            vel_change = rl_math.vecmag(self.last_velocity - cur_vel)
            if vel_change > self.min_change:
                reward = vel_change / (2300*2)

        self.last_velocity = cur_vel
        return reward


class FlipCorrecter(RewardFunction):
    def __init__(self) -> None:
        self.last_velocity = np.zeros(3)
        self.armed = False

    def reset(self, initial_state: GameState) -> None:
        self.last_velocity = np.zeros(3)
        self.armed = False

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if self.armed and player.on_ground:
            self.armed = False

        elif self.armed and not player.has_flip:
            self.armed = False
            if distance(player.car_data.position, state.ball.position) <= 500:
                vel_diff = player.car_data.linear_velocity - self.last_velocity
                if np.linalg.norm(vel_diff) > 100 and previous_action[5] == 1:
                    ball_dir = normalize(state.ball.position - player.car_data.position)
                    reward = ball_dir.dot(normalize(vel_diff))
                # if distance(player.car_data.position, state.ball.position) >= 1200:
                #     rew2 = normalize(self.last_velocity).dot(normalize(vel_diff))
                #     if rew2 > reward:
                #         reward = rew2

        elif not self.armed and not player.on_ground and player.has_flip:
            self.armed = True

        self.last_velocity = player.car_data.linear_velocity
        return reward


class TouchBallTweakedReward(RewardFunction):
    def __init__(
        self,
        min_touch: float = 0.05,
        min_height: float = 170,
        min_distance: float = 300,
        aerial_weight: float = 0.15,
        air_reward: bool = True,
        first_touch: bool = False,
    ):
        self.min_touch = min_touch
        self.min_height = min_height
        self.aerial_weight = aerial_weight
        self.air_reward = air_reward
        self.first_touch = first_touch
        self.min_distance = min_distance
        self.min_change = 500
        self.last_velocity = np.array([0, 0, 0])

    def reset(self, initial_state: GameState):
        self.last_velocity = np.array([0, 0, 0])

    def get_closest_enemy_distance(self, player: PlayerData, state: GameState) -> float:
        closest_dist = 50000
        for car in state.players:
            if car.team_num != player.team_num:
                dist = distance2D(state.ball.position, car.car_data.position)
                if dist < closest_dist:
                    closest_dist = dist
        return closest_dist

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        current_vel = state.ball.linear_velocity
        if player.ball_touched:
            if state.ball.position[2] >= self.min_height or (
                state.ball.position[2] >= BALL_RADIUS + 20
                and (
                    self.min_distance == 0
                    or self.get_closest_enemy_distance(player, state)
                    > self.min_distance
                )
            ):
                reward += max(
                    [
                        self.min_touch,
                        (
                            abs(state.ball.position[2] - BALL_RADIUS)
                            ** self.aerial_weight
                        )
                        - 1,
                    ]
                )
                reward += np.linalg.norm(self.last_velocity - current_vel) / 2300

            if self.air_reward and not player.on_ground:
                reward += 0.5
                if not player.has_flip:
                    reward += 0.5

        self.last_velocity = current_vel
        # if abs(state.ball.position[0]) > 3896 or abs(state.ball.position[1]) > 4920:
        #     reward *= 0.75
        return reward


class TouchBallReward(RewardFunction):
    def __init__(
        self,
        min_touch: float = 0.05,
        min_height: float = 170,
        min_distance: float = 300,
        aerial_weight: float = 0.15,
        air_reward: bool = True,
        first_touch: bool = False,
    ):
        self.min_touch = min_touch
        self.min_height = min_height
        self.aerial_weight = aerial_weight
        self.air_reward = air_reward
        self.first_touch = first_touch
        self.min_distance = min_distance

    def reset(self, initial_state: GameState):
        pass

    def get_closest_enemy_distance(self, player: PlayerData, state: GameState) -> float:
        closest_dist = 50000
        for car in state.players:
            if car.team_num != player.team_num:
                dist = distance2D(player.car_data.position, car.car_data.position)
                if dist < closest_dist:
                    closest_dist = dist
        return closest_dist

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            if (
                self.first_touch
                and state.ball.position[0] == 0
                and state.ball.position[1] == 0
            ):
                reward += 5
            if state.ball.position[2] >= self.min_height:

                if (
                    self.min_distance == 0
                    or self.get_closest_enemy_distance(player, state)
                    > self.min_distance
                ):
                    reward += max(
                        [
                            self.min_touch,
                            (
                                abs(state.ball.position[2] - BALL_RADIUS)
                                ** self.aerial_weight
                            )
                            - 1,
                        ]
                    )
            if self.air_reward and not player.on_ground:
                if not player.has_flip:
                    reward += 1
                else:
                    reward += 0.5

        return reward


class PushReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        pos = state.ball.position
        if player.team_num != BLUE_TEAM:
            pos = state.inverted_ball.position

        if pos[1] > 0:
            y_scale = pos[1] / 5213
            if abs(pos[0]) > 800:
                x_scale = (abs(pos[0]) / 4096) * y_scale
                scale = y_scale - x_scale
                return scale
            return y_scale

        elif pos[1] < 0:
            y_scale = pos[1] / 5213
            if abs(pos[0]) > 800:
                x_scale = (abs(pos[0]) / 4096) * abs(y_scale)
                scale = y_scale + x_scale
                return scale
            return y_scale

        return 0


class BallYCoordinateReward(RewardFunction):
    def __init__(self, exponent=1):
        # Exponent should be odd so that negative y -> negative reward
        self.exponent = exponent

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            return (
                state.ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)
            ) ** self.exponent
        else:
            return (
                state.inverted_ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)
            ) ** self.exponent


class VersatileBallVelocityReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.offensive_reward = VelocityBallToGoalReward()
        self.defensive_reward = VelocityBallDefense()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (player.team_num == BLUE_TEAM and state.ball.position[1] < 0) or (
            player.team_num == ORANGE_TEAM and state.ball.position[1] > 0
        ):
            return self.defensive_reward.get_reward(player, state, previous_action)
        else:
            return self.offensive_reward.get_reward(player, state, previous_action)


class DefenderReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.enemy_goals = 0


    def reset(self, initial_state: GameState):
        pass

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.team_num == BLUE_TEAM:
            e_score = state.orange_score
            defend_loc = BLUE_GOAL_CENTER
        else:
            e_score = state.blue_score
            defend_loc = ORANGE_GOAL_CENTER

        if e_score > self.enemy_goals:
            self.enemy_goals = e_score
            dist = distance2D(np.asarray(defend_loc), player.car_data.position)
            if dist > 900:
                reward -= clamp(1, 0, dist/10000)
        return reward

class PositiveWrapperReward(RewardFunction):
    """A simple wrapper to ensure a reward only returns positive values"""
    def __init__(self, base_reward):
        super().__init__()
        #pass in instantiated reward object
        self.rew = base_reward

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        rew = self.rew.get_reward(player, state, previous_action)
        return 0 if rew < 0 else rew


class PositiveBallVelToGoalReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.rew = VelocityBallToGoalReward()

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return clamp(1, 0, self.rew.get_reward(player, state, previous_action))

class PositivePlayerVelToBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.rew = VelocityPlayerToBallReward()

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return clamp(1, 0, self.rew.get_reward(player, state, previous_action))


class DefenseTrainer(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            defense_objective = np.array(BLUE_GOAL_BACK)
        else:
            defense_objective = np.array(ORANGE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = defense_objective - state.ball.position
        norm_pos_diff = normalize(pos_diff)
        vel = vel/BALL_MAX_SPEED
        scale = clamp(1, 0, 1 - (distance2D(state.ball.position, defense_objective)/10000))
        return -clamp(1, 0, float(norm_pos_diff.dot(vel)*scale))


class VelocityBallDefense(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            defense_objective = np.array(BLUE_GOAL_BACK)
        else:
            defense_objective = np.array(ORANGE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = state.ball.position - defense_objective
        norm_pos_diff = normalize(pos_diff)
        vel = vel/BALL_MAX_SPEED
        return float(norm_pos_diff.dot(vel))


class CradleReward(RewardFunction):
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


class CradleFlickReward(RewardFunction):
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


class TweakedVelocityPlayerToGoalReward(RewardFunction):
    def __init__(self, max_leeway=100, default_power=0.0) -> None:
        super().__init__()
        self.max_leeway = max_leeway
        self.default_power = default_power

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball = state.ball
        player_pos = player.car_data.position
        player_goal = BLUE_GOAL_BACK
        if player.team_num == ORANGE_TEAM:
            ball = state.inverted_ball
            player_pos = player.inverted_car_data.position
            player_goal = ORANGE_GOAL_BACK

        diff = player_pos - ball.position
        if diff[1] < self.max_leeway:
            return 0

        direction = normalize(np.array(player_goal) - player_pos)
        vel = player.car_data.linear_velocity
        norm_pos_diff = direction / np.linalg.norm(direction)
        vel = vel/CAR_MAX_SPEED
        return float(np.dot(norm_pos_diff, vel))


class ChallengeReward(RewardFunction):
    def __init__(self, challenge_distance=300):
        super().__init__()
        self.challenge_distance = challenge_distance

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if (
            not player.on_ground
            and distance(player.car_data.position, state.ball.position)
            < self.challenge_distance
        ):
            for _player in state.players:
                if (
                    _player.team_num != player.team_num
                    and distance(_player.car_data.position, state.ball.position)
                    < self.challenge_distance
                ):
                    reward += 0.1
                    if not player.has_flip:
                        # ball_dir_norm = normalize(state.ball.position-player.car_data.position)
                        # direction = ball_dir_norm.dot(normalize(player.car_data.linear_velocity))
                        # return direction + reward
                        reward += 0.9
                    break

        return reward


class OncePerStepRewardWrapper(RewardFunction):
    def __init__(self, reward):
        super().__init__()
        self.reward = reward
        self.gs = None
        self.rv = 0

    def reset(self, initial_state: GameState):
        self.reward.reset(initial_state)
        self.gs = None
        self.rv = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state == self.gs:
            return self.rv

        self.gs = state
        reward = self.reward.get_reward(player, state, previous_action)
        self.rv = reward
        return self.rv


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


class PositioningReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        if player.team_num != BLUE_TEAM:
            ball = state.inverted_ball.position
            pos = player.inverted_car_data.position

        reward = 0.0
        if ball[1] < pos[1]:
            diff = ball[1] - pos[1]
            reward = -clamp(1, 0, abs(diff) / 5000)
        return reward


class GoalboxPenalty(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if abs(player.car_data.position[1]) >= 5120:
            return -1
        return 0


class PlayerAlignment(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        defending = ball[1] < 0
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc
            defending = ball[1] > 0

        if defending:
            reward = rl_math.cosine_similarity(ball - pos, pos - protecc)
        else:
            reward = rl_math.cosine_similarity(ball - pos, attacc - pos)

        return reward


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


class AirReward(RewardFunction):
    def __init__(
        self,
    ):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if not player.on_ground:
            if player.has_flip:
                return 0.5
            else:
                return 1
        return 0


if __name__ == "__main__":
    pass
