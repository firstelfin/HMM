#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : 
# @Time     : 2021/6/25 上午10:52
# @File     : HMM.py
# @Software : PyCharm

from copy import deepcopy
import numpy as np

A = np.array([
    [0, 1, 0],
    [0.2, 0.35, 0.45],
    [0.4, 0.14, 0.46]
])
B = np.array([
    [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
    [0.23, 0.2, 0.175, 0.14, 0.135, 0.12],
    [0.24, 0.2, 0.175, 0.13, 0.135, 0.12]
])


# A = np.array([
#     [0.5, 0.2, 0.3],
#     [0.3, 0.5, 0.2],
#     [0.2, 0.3, 0.5]
# ])
#
# B = np.array([
#     [0.5, 0.5],
#     [0.4, 0.6],
#     [0.7, 0.3]
# ])


class HMM(object):
    """
    隐马尔可夫模型
    :param transformer: 状态转移概率矩阵
    :param observation: 观测概率矩阵
    :param pi: 初始状态分布
    """

    def __init__(self, transformer, observation, pi):
        self.transformer = transformer
        self.observation = observation
        self.pi = pi
        self.old_transformer = deepcopy(transformer)
        self.old_observation = deepcopy(observation)
        self.old_pi = deepcopy(pi)
        self.forward_alpha = 0
        self.backward_beta = 0
        self.gamma = 0
        self.xi = 0
        self.test = 0

    def forward_prob(self, last_forward, observation):
        """
        计算时刻t的前向概率分布
        :param last_forward: 上一时刻的概率分布
        :param observation: 观测概率向量
        :return: 当前的前向概率分布
        """
        res = np.dot(last_forward, self.transformer) * observation
        return res

    def forward_prob_distribution(self, observe_seq):
        """
        计算观测序列出现的概率, 分布是指在当前节点各个状态的概率分布
        :param observe_seq: 观测序列
        :return: 前向概率分布
        """
        last_prob = self.pi * self.observation[:, observe_seq[0] - 1]
        if len(observe_seq) == 1:
            return last_prob
        for v in observe_seq[1:]:
            last_prob = self.forward_prob(last_prob, self.observation[:, v - 1])
        return last_prob

    def get_forward_prob(self, observe_seq):
        """
        计算观测序列出现的概率
        :param observe_seq: 观测序列
        :return: 观测序列的概率值
        """
        last_prob = self.forward_prob_distribution(observe_seq)
        return np.sum(last_prob)

    def backward_prob(self, last_backward, observation):
        """
        计算时刻t的后向概率分布
        :param last_backward: 下一时刻的概率分布
        :param observation: 观测概率向量
        :return: 当前的后向概率分布
        """
        return np.dot(self.transformer, observation * last_backward)

    def backward_prob_distribution(self, observe_seq):
        """
        计算观测序列出现的概率
        :param observe_seq: 观测序列
        :return: 后向概率分布
        """
        last_backward = np.array([1] * self.transformer.shape[0])
        observe_seq.reverse()
        for v in observe_seq[:-1]:
            last_backward = self.backward_prob(last_backward, self.observation[:, v - 1])
        return last_backward

    def get_backward_prob(self, observe_seq):
        """
        计算观测序列出现的概率
        :param observe_seq: 观测序列
        :return: 观测序列的概率值
        """
        last_backward = self.backward_prob_distribution(observe_seq)
        return np.sum(self.pi * self.observation[:, observe_seq[-1] - 1] * last_backward)

    def get_forward_backward_prob(self, observe_seq, split_index=1):
        """
        同时使用前向--后向概率计算观测的条件概率
        :param observe_seq: 观测序列
        :param split_index: 切割的时刻, 这个时刻默认分配给前向算法
        :return: 观测序列的条件概率
        """
        assert split_index < len(observe_seq), \
            "Expect split_index to be less than the length of observe_seq."
        forward_distribution = self.forward_prob_distribution(observe_seq[:split_index])
        backward_distribution = self.backward_prob_distribution(observe_seq[split_index:])
        res = np.dot(
            forward_distribution,
            np.dot(self.transformer,
                   self.observation[:, observe_seq[split_index] - 1] * backward_distribution)
        )
        return res

    def get_qi2t_prob(self, observe_seq, t, qi):
        """
        对于给定隐马尔可夫模型与观测序列，计算在时刻t处于状态qi的概率
        :param observe_seq: 观测序列
        :param t: 时刻的编码, [默认从1开始编码]
        :param qi: 状态的编码, [默认从1开始编码]
        :return: 概率值
        """
        forward_distribution = self.forward_prob_distribution(observe_seq[:t])
        backward_distribution = self.backward_prob_distribution(observe_seq[t - 1:])
        res = forward_distribution[qi - 1] * backward_distribution[qi - 1] \
              / np.dot(forward_distribution, backward_distribution)
        return res

    def get_qi2t_qj2next_prob(self, observe_seq, t, qi, qj):
        """
        对于给定隐马尔可夫模型与观测序列，计算在时刻t处于状态qi,且下一时刻处于qj状态的概率
        :param observe_seq: 观测序列
        :param t: 时刻的编码, [默认从1开始编码]
        :param qi: t时刻状态的编码, [默认从1开始编码]
        :param qj: t+1时刻状态的编码, [默认从1开始编码]
        :return: 概率值
        """
        forward_distribution = self.forward_prob_distribution(observe_seq[:t])
        backward_distribution = self.backward_prob_distribution(observe_seq[t:])
        denominator = np.dot(
            forward_distribution,
            np.dot(self.transformer,
                   self.observation[:, observe_seq[t] - 1] * backward_distribution)
        )
        numerator = forward_distribution[qi - 1] * self.transformer[qi - 1, qj - 1] * self.observation[
            qj - 1, observe_seq[t] - 1] * backward_distribution[qj - 1]
        return numerator / denominator

    def expect_i_appear_trans(self, observe_seq, qi, transfer=False):
        """
        在观测O下, 计算状态qi出现的概率
        :param observe_seq: 观测序列
        :param qi: 状态的编码, [默认从1开始编码]
        :param transfer: 是否由状态qi转移, 默认False
        :return: 概率值
        """
        res = 1
        for t in range(1, len(observe_seq)):
            res *= 1 - self.get_qi2t_prob(observe_seq, t, qi)
        if not transfer:
            res *= 1 - self.get_qi2t_prob(observe_seq, len(observe_seq), qi)
        return 1 - res

    def expect_i2j(self, observe_seq, qi, qj):
        """
        在观测O下, 计算状态转移的概率
        :param observe_seq: 观测序列
        :param qi: 状态的编码, [默认从1开始编码]
        :param qj: 状态的编码, [默认从1开始编码]
        :return: 概率值
        """
        res = 1
        for t in range(1, len(observe_seq)):
            res *= 1 - self.get_qi2t_qj2next_prob(observe_seq, t, qi, qj)
        return 1 - res

    def get_alpha_beta_distribute(self, observe_seq):
        """
        获取在每个时刻t, alpha_{t}(i)、beta_{t}(i)的分布
        :param observe_seq: 观测序列
        :return: self.forward_alpha, self.backward_beta
        """
        length_seq = len(observe_seq)
        last_prob = np.array([0, 0, 0])
        last_backward = np.array([1] * self.transformer.shape[0])
        self.forward_alpha = np.zeros((self.transformer.shape[0], length_seq))
        for i in range(length_seq):
            if i == 0:
                last_prob = self.pi * self.observation[:, observe_seq[0] - 1]
            else:
                last_prob = self.forward_prob(last_prob, self.observation[:, observe_seq[i] - 1])
            self.forward_alpha[:, i] = last_prob

        observe_seq.reverse()
        self.backward_beta = np.zeros((self.transformer.shape[0], length_seq))

        del last_prob
        for i in range(length_seq - 1):
            last_backward = self.backward_prob(last_backward, self.observation[:, observe_seq[i] - 1])
            self.backward_beta[:, length_seq - 2 - i] = last_backward
        self.backward_beta[:, length_seq - 1] = np.array([1] * self.transformer.shape[0])

        del last_backward
        observe_seq.reverse()
        return self.forward_alpha, self.backward_beta

    def get_gamma_distribute(self, observe_seq=None):
        """
        获取在每个时刻t, gamma_{t}(i)的分布
        :param observe_seq: 观测序列
        :return: self.gamma
        """
        if type(self.backward_beta) == int or type(self.forward_alpha) == int:
            self.get_alpha_beta_distribute(observe_seq)
        self.gamma = self.forward_alpha * self.backward_beta
        for i in range(self.gamma.shape[1]):
            self.gamma[:, i] = self.gamma[:, i] / sum(self.gamma[:, i])
        return self.gamma

    def get_xi_distribute(self, observe_seq):
        """
        对于给定隐马尔可夫模型与观测序列，计算所有时刻t处于状态qi,且下一时刻处于qj状态的概率
        :param observe_seq: 观测序列
        :return: self.xi
        """
        self.xi = np.zeros((len(observe_seq) - 1, *self.transformer.shape))
        if type(self.backward_beta) == int or type(self.forward_alpha) == int:
            self.get_alpha_beta_distribute(observe_seq)

        prob_observe_on_lambda = sum(self.backward_beta[:, 1] * self.forward_alpha[:, 1])

        for t in range(len(observe_seq) - 1):
            for i in range(self.transformer.shape[0]):
                # for j in range(self.transformer.shape[0]):
                #     self.test[t, i, j] = self.forward_alpha[i, t] * self.transformer[i, j] * \
                #                          self.observation[j, observe_seq[t+1] - 1] * self.backward_beta[j, t+1]

                self.xi[t, i, :] = self.forward_alpha[i, t] * self.transformer[i, :] * \
                                   self.observation[:, observe_seq[t + 1] - 1] * self.backward_beta[:, t + 1]
            self.xi[t, :, :] = self.xi[t, :, :] / prob_observe_on_lambda
        return self.xi

    def update_param(self, observe_seq):
        self.pi = self.gamma[:, 0]
        for i in range(self.transformer.shape[0]):
            self.transformer[i, :] = sum(self.xi[:, i, :]) / sum(self.gamma[i, :-1])
            if self.transformer[i].sum() != 1.0:
                assert abs(self.transformer[i].sum() - 1) < 1e-8, f"self.transformer[{i}, :].sum() Error"
                self.transformer[i] = self.transformer[i] / self.transformer[i].sum()
        for j in range(self.observation.shape[0]):
            total_gama_j = self.gamma[j, :].sum()
            for k in range(self.observation.shape[1]):
                self.observation[j, k] = np.array([self.gamma[j, t] for t in range(len(observe_seq))
                                                   if observe_seq[t] - 1 == k]).sum() / total_gama_j
            if self.observation[j, :].sum() != 1.0:
                assert abs(self.observation[j, :].sum() - 1) < 1e-8, f"self.observation[{j}, :].sum() Error"
                self.observation[j, :] = self.observation[j, :] / self.observation[j, :].sum()
        return self.transformer, self.observation, self.pi

    def expect_and_max(self, observe_seq, threshold=1e-5):
        while True:
            self.get_alpha_beta_distribute(observe_seq)
            self.get_gamma_distribute(observe_seq)
            self.get_xi_distribute(observe_seq)
            self.update_param(observe_seq)
            total_error = abs(self.transformer - self.old_transformer).sum() / \
                          (self.transformer.shape[0] * self.transformer.shape[1]) + \
                          abs(self.observation - self.old_observation).sum() / \
                          (self.observation.shape[0] * self.observation.shape[1]) + \
                          abs(self.pi - self.old_pi).sum() / len(self.pi)
            if total_error < threshold:
                print("self.transformer:\n", self.transformer)
                print("self.observation:\n", self.observation)
                print("self.pi:\n", self.pi)
                break
            else:
                self.old_transformer = deepcopy(self.transformer)
                self.old_observation = deepcopy(self.observation)
                self.old_pi = deepcopy(self.pi)
        return True


if __name__ == '__main__':
    hmm = HMM(A, B, [1 / 3, 1 / 3, 1 / 3])
    # hmm.get_gamma_distribute([6, 3, 1, 2, 4, 2])
    # hmm.get_xi_distribute([6, 3, 1, 2, 4, 2])
    # hmm.update_param([6, 3, 1, 2, 4, 2])
    hmm.expect_and_max([6, 3, 1, 2, 4, 2])
    # hmm.get_gamma_distribute([6, 3, 1, 2, 4, 2])
    # result = hmm.forward_prob_distribution([6, 3, 1, 2, 4, 2])
    # print(result)
    # result = hmm.get_forward_prob([6, 3, 1, 2, 4, 2])
    # print(result)
    # result = hmm.get_backward_prob([6, 3, 1, 2, 4, 2])
    # print(result)
    # result = hmm.get_forward_backward_prob([6, 3, 1, 2, 4, 2], 3)
    # print(result)
    # result = hmm.get_qi2t_prob([6, 3, 1, 2, 4, 2], 3, 3)
    # print(result)
    # result = hmm.get_qi2t_qj2next_prob([6, 3, 1, 2, 4, 2], 3, 1, 2)
    # print(result)
    # result = hmm.expect_i_appear_trans([6, 3, 1, 2, 4, 2], 3)
    # print(result)
    # result = hmm.expect_i2j([6, 3, 1, 2, 4, 2], 2, 3)
    # print(result)
    # hmm = HMM(A, B, [0.2, 0.4, 0.4])
    # result = hmm.get_forward_prob([1, 2, 1])
    # print(result)
    # result = hmm.get_backward_prob([1, 2, 1])
    # print(result)
