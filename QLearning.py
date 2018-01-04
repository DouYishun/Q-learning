import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class QLearning:
    def __init__(self, alpha, gamma, epsilon, epsilon_decay, epsilon_min, penalty, legal_actions, discrete_util):
        self.alpha = alpha
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.legal_actions = legal_actions
        self.discrete_util = discrete_util
        self.penalty = penalty

        self.Q = defaultdict(float)  # {(state, action): value}
        self.Policy = {}  # {state: action}

    def q_learn_step(self, s, a, r, next_s, done=False):
        acts = self.legal_actions(next_s)
        if done:  # for terminal next state
            tmp = r
        else:  # for non-terminal next state
            tmp = r + self.gamma * max([self.Q[(next_s, a_next)] for a_next in acts])
        self.Q[(s, a)] += self.alpha * (tmp - self.Q[(s, a)])
        self.Policy[s] = acts[np.argmax([self.Q[(s, t)] for t in acts])]

    def epsilon_greedy_action(self, state):
        acts = self.legal_actions(state)
        if np.random.random() < self.epsilon:
            return acts[np.random.randint(0, len(acts))]
        if state not in self.Policy:
            self.Policy[state] = acts[0]
        return self.Policy[state]

    def train(self, env, episodes, step):
        bins = self.discrete_util()
        history_reward = []
        lr = self.alpha
        for i_episode in range(episodes):
            sum_reward = 0
            if i_episode % 10 == 0 and self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.alpha = lr
            observation = env.reset()
            state = self.discretize(observation, bins)
            for t in range(step):
                action = self.epsilon_greedy_action(state)
                observation, reward, done, info = env.step(action)
                sum_reward += reward
                next_state = self.discretize(observation, bins)
                reward = reward if not done else self.penalty
                self.q_learn_step(state, action, reward, next_state)
                state = next_state
                if done or t == step - 1:
                    print("episode: {}/{}, score: {}, epsilon: {:.2}".format(i_episode, episodes, t, self.epsilon))
                    break
            history_reward.append(sum_reward)
            if i_episode % 100 == 0:
                np.save("policy/policy_" + str(i_episode), self.Policy)
        return history_reward

    @staticmethod
    def discretize(observation, bins):
        return tuple([int(np.digitize(observation[i], bins[i])) for i in range(len(observation))])


def plot_reward(history_reward):
    num = len(history_reward) // 100
    history_reward = history_reward[:num * 100]
    splited_reward = [history_reward[i:i + 100] for i in range(0, len(history_reward), 100)]
    mean_reward = np.mean(splited_reward, axis=1)
    x = [i * 100 for i in list(range(num))]
    plt.plot(x, mean_reward)
    plt.legend(['Train Reward'])
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.show()


class CartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env = self.env.unwrapped

    @staticmethod
    def legal_actions(state):
        return [0, 1]

    @staticmethod
    def discrete_util():
        bins_num = [4, 8, 4, 8]
        state_feature_dim = 4
        min_values = [-0.5, -2.0, -0.2, -1.0]
        max_values = [0.5, 2.0, 0.2, 1.0]
        bins = [np.linspace(min_values[i], max_values[i], bins_num[i]) for i in range(state_feature_dim)]
        print(bins)
        return bins

    def run(self):
        alpha, gamma, epsilon, epsilon_decay, epsilon_min, penalty = 0.06, 0.9, 1.0, 0.99, 0.01, -2000
        episodes, step = 10000, 20000

        agent = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon,
                          epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                          penalty=penalty,
                          legal_actions=self.legal_actions,
                          discrete_util=self.discrete_util)
        history_reward = agent.train(env=self.env, episodes=episodes, step=step)
        plot_reward(history_reward)
        self.env.close()


class MountainCar:
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.env = self.env.unwrapped

    @staticmethod
    def legal_actions(state):
        return [0, 1, 2]

    @staticmethod
    def discrete_util():
        bins_num = 8
        state_feature_dim = 2
        min_values = [-1.0, -0.06]
        max_values = [0.5, 0.06]
        bins = [np.linspace(min_values[i], max_values[i], bins_num) for i in range(state_feature_dim)]
        return bins

    def run(self):
        alpha, gamma, epsilon, epsilon_decay, epsilon_min, penalty = 0.06, 0.9, 0.5, 0.99, 0.01, 2000
        episodes, step = 10000, 2000

        agent = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon,
                          epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                          penalty=penalty,
                          legal_actions=self.legal_actions,
                          discrete_util=self.discrete_util)
        history_reward = agent.train(env=self.env, episodes=episodes, step=step)
        plot_reward(history_reward)
        self.env.close()


class Acrobot:
    def __init__(self):
        self.env = gym.make('Acrobot-v1')
        self.env = self.env.unwrapped

    @staticmethod
    def legal_actions(state):
        return [0, 1, 2]

    @staticmethod
    def discrete_util():
        bins_num = [8, 8, 8, 8, 32, 64]
        state_feature_dim = 6
        min_values = [-0.9, -0.9, -0.9, -0.9, -10, -20]
        max_values = [0.9, 0.9, 0.9, 0.9, 10, 20]
        bins = [np.linspace(min_values[i], max_values[i], bins_num[i]) for i in range(state_feature_dim)]
        return bins

    def run(self):
        alpha, gamma, epsilon, epsilon_decay, epsilon_min, penalty = 0.06, 0.9, 1.0, 0.99, 0.01, -20000
        episodes, step = 10000, 2000

        agent = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon,
                          epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                          penalty=penalty,
                          legal_actions=self.legal_actions,
                          discrete_util=self.discrete_util)
        history_reward = agent.train(env=self.env, episodes=episodes, step=step)
        plot_reward(history_reward)
        self.env.close()


def main():
    cartpole = CartPole()
    cartpole.run()

    #mountaincar = MountainCar()
    #mountaincar.run()

    #acrobot = Acrobot()
    #acrobot.run()


if __name__ == '__main__':
    main()


