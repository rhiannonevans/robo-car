import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------

# cart pole
register(
    id='GazeboCartPole-v0',
    entry_point='gym_gazebo.envs.gazebo_cartpole:GazeboCartPolev0Env',
)
# line follower
register(
	id='Gazebo_linefollow-v0',
	entry_point='gym_gazebo.envs.gazebo_dqfollow:GazeboDeepQLineEnv',
)


