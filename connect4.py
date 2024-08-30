"""DQN agents trained on Breakthrough by independent Q-learning."""

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
flags.DEFINE_string("checkpoint_dir", "/tmp/dqn_test",
                    "Directory to save/load the agent models.")
flags.DEFINE_integer(
    "save_every", int(1e4),
    "Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(1e6),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 1000,
    "Episode frequency at which the DQN agents are evaluated.")

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")


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


def main(_):
  game = "connect_four"
  num_players = 2

#   env_configs = {"columns": 5, "rows": 5}
  env = rl_environment.Environment("connect_four")
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  with tf.Session() as sess:
    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    # pylint: disable=g-complex-comprehension
    sess.run(tf.global_variables_initializer())

    temp = None
    import random

    learned = dqn.DQN(
            session=sess,
            player_id=0,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
            batch_size=FLAGS.batch_size)

    opponent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)
    rand_agent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)

    for lesson in range(2):
      for ep in range(10001):
        if (ep + 1) % FLAGS.eval_every == 0:
          r_mean = eval_against_opponent(env, [learned, rand_agent], 1000)
          logging.info("[%s] Mean episode rewards %s", ep + 1, r_mean)
        if (ep + 1) % FLAGS.save_every == 0:
          learned.save(FLAGS.checkpoint_dir)

        time_step = env.reset()
        while not time_step.last():
          agent_id = time_step.observations["current_player"]
          agent_output = None
          if agent_id == 0:
            agent_output = learned.step(time_step)
          else:
            agent_output = opponent.step(time_step)
          action_list = [agent_output.action]
          time_step = env.step(action_list)

        for agent in [learned, opponent]:
          agent.step(time_step)

      opponent = learned.copy_with_noise()
      opponent.player_id = 1


if __name__ == "__main__":
  app.run(main)