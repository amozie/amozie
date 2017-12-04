import numpy as np
import time
import matplotlib.pyplot as plt

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

env = OpenAIGym('MountainCar-v0', visualize=False)

network_spec = [
    dict(type='dense', size=16, activation='relu'),
    dict(type='dense', size=16, activation='relu'),
    dict(type='dense', size=16, activation='relu')
]

agent = PPOAgent(
    states_spec=env.states,
    actions_spec=env.actions,
    network_spec=network_spec,
    batch_size=1024,
    # Agent
    # preprocessing=None,
    # exploration=None,
    # reward_preprocessing=None,
    # BatchAgent
    keep_last_timestep=True,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    optimization_steps=10,
    # Model
    scope='ppo',
    discount=0.99,
    # DistributionModel
    distributions_spec=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    summary_spec=None,
    distributed_spec=None
)

reward_list = []

args_episodes = 1000
args_episode_max_steps = 200
episode = 0
agent.reset()
while True:
    agent.reset()
    state = env.reset()
    episode += 1
    episode_step = 0
    episode_reward = 0
    while True:
        action = agent.act(state)
        state, terminal, reward = env.execute(action)
        reward = np.abs(state[1]) - 0.05
        episode_reward += reward
        episode_step += 1
        if args_episode_max_steps is not None and episode_step >= args_episode_max_steps:
            terminal = True
        agent.observe(terminal, reward)

        if terminal:
            break
    print('episode {0} steps {1} reward {2}'.format(episode, episode_step, episode_reward))
    reward_list.append(episode_reward)
    if episode >= args_episodes:
        break
    # if len(reward_list) > 100 and np.mean(reward_list[-100:]) > 199:
    #     print('good enough!!!')
    #     break

plt.plot(reward_list)
