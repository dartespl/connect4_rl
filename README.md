# connect4_rl: a game of connect 4 with reinforcement learning

> Authors: [Paweł Fornalik](https://github.com/dartespl) and [Marcin Dąbrowski](https://github.com/mardab)

## Prerequisites
* AVX2-capable CPU or CUDA-capable GPU
* Linux operating system (WSL2 tested as working with Debian image)
* Python 3.9 (dependencies have not been updated for 3.10+)
* Python environment set up with pip (`requirements.txt`) or Conda (`environment.yml`)

## Usage

Run `connect4_train.py` to train reinforcement model, then `connect4_train.py` to run a set of games of connect 4.

### Both scripts have a common set of options, preset with common sense default values used for debugging, which can be overridden with flags when invoked in shell:

`-d, --checkpoint_dir` (string, `./checkpoint`)

> Directory to save/load the agent models.

`-s, --save_every` (integer, `1e4`)

> Episode frequency at which the DQN agent models are saved.

`-t, --num_train_episodes` (integer, `1e6`)

> Number of training episodes.

`-e, --eval_every` (integer, `1000`)

> Episode frequency at which the DQN agents are evaluated.

`-h, --hidden_layers_sizes` (list of 2 integers, `[64, 64]`)

> Number of hidden units in the Q-Network MLP.

`-r, replay_buffer_capacity` (integer, `1e5`)

> Size of the replay buffer.

`-b --batch_size` (integer, `32`)

> Number of transitions to sample at each learning step.

### Example:
`python connect4_train.py -d PATH`
> Training results will be saved to chosen PATH
> 
> (remember to invoke `python connect4_train.py` with the same `-d PATH`)

### Results
![alt text](https://github.com/dartespl/connect4_rl/blob/main/evaluations.png?raw=true)
> Training has 30000 iterations.
> Evaluation "eval_won_against_bots" is % games won against bots which play randomly. There are 1000 games played on each evaluation.
> CON-9 hidden_layers_sizes = [64]
> CON-10 hidden_layers_sizes = [64, 64]
> CON-11 hidden_layers_sizes = [64, 64, 64]
