import numpy as np
from rlgym_sim.utils.gamestates import GameState
from lookup_act import LookupAction
from rlgym_ppo.util import MetricsLogger
from state_setters import ProbabilisticStateSetter


class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward, FaceBallReward, InAirReward, SpeedTowardBallReward, TouchBallRewardScaledByHitForce, JumpTouchReward, SaveBoostReward,LavaFloorReward,TouchBallReward,LiuDistanceBallToGoalReward,SwiftProximityToBallReward, AlignBallGoal,CenterReward, GroundedReward, TouchVelChange
    from customreward import KickoffProximityReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from lookup_act import LookupAction
    from rlgym_sim.utils.state_setters import RandomState, DefaultState

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    reward_fn = CombinedReward.from_zipped(
    # Format is (func, weight)
    (EventReward(team_goal=1, concede=-1), 100),
    (VelocityBallToGoalReward(), 2),
    (TouchVelChange(), 35),
    (KickoffProximityReward(), 5),
    (VelocityPlayerToBallReward(), 1),
    (FaceBallReward(), .01),
    (InAirReward(), 0.005),
    )


    state_config = [[DefaultState(), RandomState(True, True, False)],
                    [15, 5]]


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
                         #state_setter=RandomState(True, True, False)
                         state_setter=ProbabilisticStateSetter(state_config[0], state_config[1]),
                         )

    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)
    
    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    # 32 processes
    n_proc = 40

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
                      save_every_ts=1_000_000,
                      timestep_limit=100_000_000_000,
                      log_to_wandb=True,
                      wandb_run_name = "Karma v1",
                      policy_layer_sizes=(2048, 1024, 1024, 1024),
                      critic_layer_sizes=(2048, 1024, 1024, 1024),
                      device="cuda",
                      render=True,
                      render_delay=0.05,
                      policy_lr=2e-4,
                      critic_lr=2e-4)
    learner.learn()