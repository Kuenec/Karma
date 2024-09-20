import numpy as np
from rlgym_sim.utils.gamestates import GameState
from lookup_act import LookupAction
from rlgym_ppo.util import MetricsLogger
from state_setters import ProbabilisticStateSetter, DribblingStateSetter
from customreward import Flick45DegreeReward, Flick45DegreeRewardV2,Car45DegreeFlickReward,Car45DegreeFlickRewardV6,Car45DegreeFlickRewardV7, QuickestTouchReward, GoalSpeedAndPlacementReward, KickoffProximityReward, ZeroSumReward, SwiftGroundDribbleReward, AirTouchReward, CradleFlickReward, LemTouchBallReward, RetreatReward, DistanceReward, AerialDistanceReward, InAirReward, TouchVelChange, CradleReward, GroundedReward, GroundDribbleReward, JumpTouchReward
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    GoodVelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward, FaceBallReward, SaveBoostReward, TouchBallReward, LiuDistanceBallToGoalReward, 
    AlignBallGoal
)
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_sim.utils import common_values
from rlgym_sim.utils.state_setters import RandomState, DefaultState
# Add custom LogCombinedReward
from customreward import LogCombinedReward

g_combined_reward = None  # type: LogCombinedReward

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        # Collect metrics including previous rewards
        metrics = [game_state.players[0].car_data.linear_velocity,
                   game_state.players[0].car_data.rotation_mtx(),
                   game_state.orange_score]
        if g_combined_reward and g_combined_reward.prev_rewards:
            metrics.append(g_combined_reward.prev_rewards)
        return metrics

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        avg_rewards = np.zeros(len(g_combined_reward.reward_functions)) if g_combined_reward else np.zeros(0)
        
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
            if g_combined_reward and len(metric_array) > 3:  # Ensure prev_rewards is available
                avg_rewards += metric_array[-1]
        
        avg_linvel /= len(collected_metrics)
        if g_combined_reward:
            avg_rewards /= len(collected_metrics)
        
        report = {"x_vel": avg_linvel[0],
                  "y_vel": avg_linvel[1],
                  "z_vel": avg_linvel[2],
                  "Cumulative Timesteps": cumulative_timesteps}
        
        # Add reward metrics
        if g_combined_reward:
            for i in range(len(g_combined_reward.reward_functions)):
                report["RW " + g_combined_reward.reward_functions[i].__class__.__name__] = avg_rewards[i]
        
        wandb_run.log(report)

def build_rocketsim_env():
    import rlgym_sim
    from customreward import LogCombinedReward


    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    reward_fn = LogCombinedReward.from_zipped(

                            #EVENT REWARDS:

        (EventReward(team_goal=1.35, concede=-1, boost_pickup=0.13), 150),
        #(TouchVelChange(threshold=500), 50),
        (JumpTouchReward(min_height=120), 40), 
        (GoalSpeedAndPlacementReward(), 15),
        
                            #CONTINOUS REWARDS:
        (VelocityBallToGoalReward(), 12),
        (TouchBallReward(), 1.25),
        (KickoffProximityReward(), 15),
        (GoodVelocityPlayerToBallReward(), .3),
        (SaveBoostReward(), 1.5),
        (FaceBallReward(), .001),
        (AerialDistanceReward(5, 10), 4.5),
        (LemTouchBallReward(), 2.5),
        (InAirReward(), .027),
        (ZeroSumReward(SwiftGroundDribbleReward(), 0, 1.0), 6)
    )
    global g_combined_reward
    g_combined_reward = reward_fn

    state_config = [[DefaultState(), RandomState(True, True, False)],
                    [5, 1]]

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=ProbabilisticStateSetter(state_config[0], state_config[1]),
                         )

    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)
    
    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    # 32 processes
    n_proc = 45

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      ppo_batch_size=50_000,
                      ts_per_iteration=50_000,
                      exp_buffer_size=150_000,
                      ppo_minibatch_size=25_000,
                      ppo_ent_coef=0.01,
                      ppo_epochs=2,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=50_000_000,
                      timestep_limit=100_000_000_000,
                      log_to_wandb=True,
                      wandb_run_name="Karma v1",
                      policy_layer_sizes=(2048, 1024, 1024, 1024),
                      critic_layer_sizes=(2048, 1024, 1024, 1024),
                      device="cuda",
                      render=True,
                      render_delay=0.04,
                      policy_lr=1e-4,
                      critic_lr=1e-4)
    
    build_rocketsim_env()  # Ensure the environment is built before learning

    learner.learn()
