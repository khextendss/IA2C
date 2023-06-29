'''====================================================================================
Generic Actor-Critic Network classes with functions to build, train, and run the NNs.

Copyright (C) August, 2019  Bikramjit Banerjee

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

===================================================================================='''

import tensorflow as tf
import numpy as np
import sys
from gym.spaces import Discrete
import random

from baselines.common.distributions import make_pdtype

tf.compat.v1.disable_eager_execution()
BETA = 0.001 #Exploration
LR_A = 0.000001 # (slower) learning rate for actor
LR_C = 0.00001

class CriticNetwork:
    def __init__(self, scope, num_state_features, num_actions):
        self.scope = scope
        self.loss_vec = []
        with tf.compat.v1.variable_scope(self.scope + '/main_critic'):
            self.state_input = tf.compat.v1.placeholder(tf.float32, [None, num_state_features], name="state")
            self.Q_train = tf.compat.v1.placeholder(tf.float32, [None], name="target_Q_val") 
            hidden_out = tf.nn.tanh(tf.compat.v1.layers.dense(self.state_input, 200, name="CH1")) 
            hidden_out = tf.compat.v1.layers.dense(hidden_out, 200, name="CH2") 
            self.Q = tf.compat.v1.layers.dense(hidden_out, num_actions, name="Q_val") 
            self.act_t = tf.compat.v1.placeholder(tf.int32, [None], name="action")
            q_t_selected = tf.reduce_sum(self.Q * tf.one_hot(self.act_t, num_actions), 1)

            self.loss = tf.losses.mean_squared_error(self.Q_train, q_t_selected)
            self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=LR_C).minimize(self.loss)
            
        with tf.compat.v1.variable_scope(self.scope + '/target_critic'): #target network with delayed params (must have same arch as main)
            self.target_state_input = tf.compat.v1.placeholder(tf.float32, [None, num_state_features], name="state")
            hidden_out = tf.nn.tanh(tf.compat.v1.layers.dense(self.target_state_input, 200, name="CH1")) 
            hidden_out = tf.compat.v1.layers.dense(hidden_out, 200, name="CH2") 
            self.target_Q = tf.compat.v1.layers.dense(hidden_out, num_actions, name="Q_val")

        init_op = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        self.sess.run(init_op)

        self.main_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/main_critic')
        self.target_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/target_critic')
        self.sync_op = [t_p.assign(m_p) for t_p, m_p in zip(self.target_params, self.main_params)]

    def sync(self): #Copy the main critic's weights into the target critic
        self.sess.run(self.sync_op)
        #print("AFTER SYNC: ", self.sess.run([self.target_params]))

    def batch_update(self, x_vec, op_neuron_sel_vec, target_vec): #Updates only the main critic
        num_steps = 10
        for it in range(num_steps):
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.state_input: x_vec, self.act_t: op_neuron_sel_vec, self.Q_train: target_vec})
            self.loss_vec.append(loss)
        self.value_loss = np.mean(self.loss_vec[-100:])

    def run_main(self, s):
        s=[s] #Assumes being run on a single state
        return self.sess.run(self.Q, feed_dict={self.state_input: s})[0]
        
    def run_target(self, s):
        s=[s]
        return self.sess.run(self.target_Q, feed_dict={self.target_state_input: s})[0]



class ActorNetwork:
    def __init__(self, scope, num_state_features, num_actions): 
        self.scope = scope
        self.optimizer = tf.compat.v1.train.AdamOptimizer(LR_A)
        with tf.compat.v1.variable_scope(self.scope + '/actor'):
            self.state_input = tf.compat.v1.placeholder(tf.float32, [None, num_state_features])
            self.act_train = tf.compat.v1.placeholder(tf.uint8, [None])
            self.adv = tf.compat.v1.placeholder(tf.float32, [None])
            hidden_out = tf.nn.tanh(tf.compat.v1.layers.dense(self.state_input, 200, name="AH1"))
            hidden_out = tf.nn.relu(tf.compat.v1.layers.dense(hidden_out, 100, name="AH2")) 
            logits = tf.compat.v1.layers.dense(hidden_out, num_actions, name="AH3")
            self.pdtype = make_pdtype(Discrete(num_actions))
            self.pd, self.pi = self.pdtype.pdfromlatent(logits, init_scale=0.01)
            with tf.name_scope('sample_a'):
                self.A_sample = self.pd.sample()
            with tf.name_scope('best_a'):
                self.A_best = self.pd.mode()
            with tf.name_scope('distribution_a'):
                self.A_dist = self.pd.mean
            neglogpac = self.pd.neglogp(self.act_train)
            pg_loss = self.adv * neglogpac
            entropy = self.pd.entropy()
            self.loss = tf.reduce_mean(pg_loss - entropy * BETA)
            
        self.params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        self.grads = tf.gradients(self.loss, self.params)
        self.update_op = self.optimizer.apply_gradients(zip(self.grads, self.params))
        
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        self.sess.run(init_op)

    def batch_update(self, x_vec, op_neuron_sel_vec, adv_vec):
        self.sess.run(self.update_op, feed_dict={self.state_input: x_vec, self.act_train: op_neuron_sel_vec, self.adv: adv_vec})
        
    def sample_action(self, s):
        s=[s]  #Assumes being run on a single state
        return self.sess.run(self.A_sample, feed_dict={self.state_input: s})[0]

    def best_action(self, s):
        s=[s]  #Assumes being run on a single state
        return self.sess.run(self.A_best, feed_dict={self.state_input: s})[0]
        
    def action_distribution(self, s):
        s=[s]  #Assumes being run on a single state
        return self.sess.run(self.A_dist, feed_dict={self.state_input: s})[0]
