"""Reinforcement learning connect 4: playing script"""

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent

FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string("checkpoint_dir", "./checkpoint",
                    "Directory to save/load the agent models.", short_name='d')
flags.DEFINE_integer(
    "save_every", int(1e4),
    "Episode frequency at which the DQN agent models are saved.", short_name='s')
flags.DEFINE_integer("num_train_episodes", int(1e6),
                     "Number of training episodes.", short_name='t')
flags.DEFINE_integer(
    "eval_every", 1000,
    "Episode frequency at which the DQN agents are evaluated.", short_name='e')

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64],
                  "Number of hidden units in the Q-Network MLP.", short_name='h')
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.", short_name='r')
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.", short_name='b')


def eval_against_opponent(env, agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  sum_episode_rewards = np.zeros(2)
  for player_pos in range(2):
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        agent_id = time_step.observations["current_player"]
        agent_output = agents[agent_id].step(
            time_step, is_evaluation=True)
        action_list = [agent_output.action]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  return sum_episode_rewards / num_episodes


def play(env, agents):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""

  time_step = env.reset()
  episode_rewards = 0
  while not time_step.last():
    agent_id = time_step.observations["current_player"]
    if env.is_turn_based:
      agent_output = agents[agent_id].step(
          time_step, is_evaluation=True)
      action_list = [agent_output.action]

    time_step = env.step(action_list)
    print(str(env._state))


def main(_):
  game = "connect_four"
  num_players = 2

#   env_configs = {"columns": 5, "rows": 5}
  env = rl_environment.Environment(game)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  with tf.Session() as sess:
    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    # pylint: disable=g-complex-comprehension

    learned = dqn.DQN(
            session=sess,
            player_id=0,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
            batch_size=FLAGS.batch_size)

    opponent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)

    learned.restore("/tmp/dqn_test")

    agents = [
        learned,
        opponent,
    ]

    sess.run(tf.global_variables_initializer())

    r_mean = eval_against_opponent(env, agents, 1000)
    logging.info("Mean episode rewards %s", r_mean)

    play(env, agents)

    agents = [
        random_agent.RandomAgent(player_id=0, num_actions=num_actions),
        learned,
    ]

    learned.player_id = 1

    r_mean = eval_against_opponent(env, agents, 1000)
    logging.info("Mean episode rewards %s", r_mean)


if __name__ == "__main__":
  app.run(main)