import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import Monitor
import MyQLearning


def discretize(observation, bins):
    return tuple([int(np.digitize(observation[i], bins[i])) for i in range(len(observation))])


def myTest(policy_filename, env, episodes, step, discrete_util):
    bins = discrete_util()
    history_reward = []
    policy = np.load(policy_filename).item()
    for i_episode in range(episodes):
        sum_reward = 0
        observation = env.reset()
        state = discretize(observation, bins)
        for t in range(step):
            if state in policy.keys():
                action = policy[state]
            else:
                action = 0
            observation, reward, done, info = env.step(action)
            sum_reward += reward
            next_state = discretize(observation, bins)
            state = next_state
            if done or t == step - 1:
                print("episode: {}/{}, score: {}".format(i_episode, episodes, t))
                break
        history_reward.append(sum_reward)

    plot_reward(history_reward)


def plot_reward(history_reward):
    num = len(history_reward) // 100
    history_reward = history_reward[:num * 100]
    splited_reward = [history_reward[i:i + 100] for i in range(0, len(history_reward), 100)]
    mean = np.mean(splited_reward, axis=1)
    std = np.std(splited_reward, axis=1)
    x = list(range(num))

    print("mean: ", mean, "\nstd: ", std)

    plt.errorbar(x, mean, std, linestyle='None', marker='o')
    plt.legend(['Testing Reward (mean & std)'])
    plt.xlabel('Set Index')
    plt.ylabel('(mean & std)')
    plt.show()


def test_cartpole():
    cartpole = MyQLearning.CartPole()
    policy_filename = "policy/policy_9600.npy"
    env = gym.make('CartPole-v0')

    env = env.unwrapped
    env = Monitor(env, "video", force=True)
    episodes = 1000
    step = 20000
    myTest(policy_filename, env, episodes, step, cartpole.discrete_util)


def test_mountaincar():
    mountaincar = MyQLearning.MountainCar()
    policy_filename = "policy/policy_5300.npy"
    env = gym.make('MountainCar-v0')

    env = env.unwrapped
    #env = Monitor(env, "video", force=True)
    episodes = 1000
    step = 2000
    myTest(policy_filename, env, episodes, step, mountaincar.discrete_util)


def test_acrobot():
    acrobot = MyQLearning.Acrobot()
    policy_filename = "policy/policy_9900.npy"
    env = gym.make('Acrobot-v1')

    env = env.unwrapped
    #env = Monitor(env, "video", force=True)
    episodes = 1000
    step = 2000
    myTest(policy_filename, env, episodes, step, acrobot.discrete_util)


def main():
    test_cartpole()
    #test_mountaincar()
    #test_acrobot()


if __name__ == '__main__':
    main()

