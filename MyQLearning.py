import gym
from gym import wrappers
import numpy as np
from collections import defaultdict


class QLearning:
    def __init__(self, alpha, gamma, epsilon, alpha_decay, epsilon_decay, penalty, legal_actions, discrete_util):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha_decay = alpha_decay
        self.epsilon_decay = epsilon_decay
        self.legal_actions = legal_actions  # function of
        self.discrete_util = discrete_util
        self.penalty = penalty

        self.Q = defaultdict(float)  # {(state, action): value}
        self.Policy = {}  # {state: action}

    def q_learn_step(self, x, a, x_next, r, done=False):
        """

        :param x: current state
        :param a: action
        :param x_next: next state
        :param r: reward
        :param done:
        :return:
        """
        acts = self.legal_actions(x_next)
        if done:
            tmp = r
        else:
            tmp = r + self.gamma * max([self.Q[(x_next, a_next)] for a_next in acts])
        self.Q[(x, a)] += self.alpha * (tmp - self.Q[(x, a)])
        self.alpha *= self.alpha_decay
        self.Policy[x] = acts[np.argmax([self.Q[(x, t)] for t in acts])]

    def epsilon_greedy_action(self, state):
        acts = self.legal_actions(state)
        if np.random.random() < self.epsilon:
            self.epsilon *= self.epsilon_decay
            return acts[np.random.randint(0, len(acts))]
        if state not in self.Policy:
            self.Policy[state] = acts[0]
        return self.Policy[state]

    def train(self, env, episodes, step):
        bins = self.discrete_util()
        lr = self.alpha
        for i_episode in range(episodes):
            self.alpha = lr
            observation = env.reset()
            state = self.discretize(observation, bins)
            for t in range(step):
                action = self.epsilon_greedy_action(state)
                observation, reward, done, info = env.step(action)
                state_next = self.discretize(observation, bins)
                if done:
                    reward = self.penalty
                    self.q_learn_step(state, action, state_next, reward, done)
                    print(i_episode, " episode [done] at ", t)
                    break
                self.q_learn_step(state, action, state_next, reward)
                state = state_next
                if t == step - 1:
                    print(i_episode, " episode [finish]")

    @staticmethod
    def discretize(observation, bins):
        return tuple([int(np.digitize(observation[i], bins[i])) for i in range(len(observation))])


class CartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        #self.env = wrappers.Monitor(self.env, '/tmp/cartpole', force=True)
        self.env = self.env.unwrapped

    @staticmethod
    def legal_actions(state):
        return [0, 1]

    @staticmethod
    def discrete_util():
        bins_num = 8
        state_feature_dim = 4
        min_values = [-2.0, -2.0, -0.18, -3.0]
        max_values = [2.0, 2.0, 0.18, 3.0]
        bins = [np.linspace(min_values[i], max_values[i], bins_num) for i in range(state_feature_dim)]
        return bins

    def run(self):
        """
        best paras:
        alpha, gamma, epsilon, alpha_decay, epsilon_decay = 0.06, 0.9, 0.5, 0.999, 0.99999
        PENALTY = -20000貌似penalty需要调高！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

        5000
        :return:
        """
        alpha, gamma, epsilon, alpha_decay, epsilon_decay, penalty = 0.06, 0.9, 0.5, 1, 0.99, -20000
        episodes, step = 10000, 20000

        agent = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon,
                          alpha_decay=alpha_decay, epsilon_decay=epsilon_decay,
                          penalty=penalty,
                          legal_actions=self.legal_actions,
                          discrete_util=self.discrete_util)

        print("episodes, step: ", episodes, step)
        agent.train(env=self.env, episodes=episodes, step=step)
        self.env.close()


class MountainCar:
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        #self.env = wrappers.Monitor(self.env, '/tmp/mountaincar', force=True)
        self.env = self.env.unwrapped

    @staticmethod
    def legal_actions(state):
        return [0, 1, 2]

    @staticmethod
    def discrete_util():
        bins_num = 8
        state_feature_dim = 2
        min_values = [-1.0, -0.05]
        max_values = [0.5, 0.05]
        bins = [np.linspace(min_values[i], max_values[i], bins_num) for i in range(state_feature_dim)]
        return bins

    def run(self):
        """
        best paras:
        alpha, gamma, epsilon, alpha_decay, epsilon_decay, penalty = 0.06, 0.9, 0.5, 1, 0.99, -2000
        PENALTY = -2000

        5000
        :return:
        """
        alpha, gamma, epsilon, alpha_decay, epsilon_decay, penalty = 0.06, 0.9, 0.5, 1, 0.99, -2000
        episodes, step = 10000, 2000

        agent = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon,
                          alpha_decay=alpha_decay, epsilon_decay=epsilon_decay,
                          penalty=penalty,
                          legal_actions=self.legal_actions,
                          discrete_util=self.discrete_util)

        print("episodes, step: ", episodes, step)
        agent.train(env=self.env, episodes=episodes, step=step)
        self.env.close()


class Acrobot:
    def __init__(self):
        self.env = gym.make('Acrobot-v1')
        #self.env = wrappers.Monitor(self.env, '/tmp/acrobot', force=True)
        self.env = self.env.unwrapped

    @staticmethod
    def legal_actions(state):
        return [0, 1, 2]

    @staticmethod
    def discrete_util():
        bins_num = 8
        state_feature_dim = 6
        min_values = [-0.9, -0.9, -0.9, -0.9, -10, -20]
        max_values = [0.9, 0.9, 0.9, 0.9, 10, 20]
        bins = [np.linspace(min_values[i], max_values[i], bins_num) for i in range(state_feature_dim)]
        return bins

    def run(self):
        """
        best paras:
        alpha, gamma, epsilon, alpha_decay, epsilon_decay, penalty = 0.06, 0.9, 0.5, 1, 0.99, -2000
        PENALTY = -2000

        5000
        :return:
        """
        alpha, gamma, epsilon, alpha_decay, epsilon_decay, penalty = 0.06, 0.9, 0.5, 1, 0.999, -2000
        episodes, step = 10000, 2000

        agent = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon,
                          alpha_decay=alpha_decay, epsilon_decay=epsilon_decay,
                          penalty=penalty,
                          legal_actions=self.legal_actions,
                          discrete_util=self.discrete_util)

        print("episodes, step: ", episodes, step)
        agent.train(env=self.env, episodes=episodes, step=step)
        self.env.close()


def main():
    #cartpole = CartPole()
    #cartpole.run()

    #mountaincar = MountainCar()
    #mountaincar.run()

    acrobot = Acrobot()
    acrobot.run()

if __name__ == '__main__':
    main()


