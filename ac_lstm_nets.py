'''====================================================================================
Generic Actor-Critic Recurrent Network classes with functions to build, train, and
run the NNs.

Copyright (C) November, 2019  Bikramjit Banerjee

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

#BETA = 0.001 #Exploration
#LR_A = 0.000001 # (slower) learning rate for actor
#LR_C = 0.00001

class CriticNetwork:
    def __init__(self, scope, num_state_features, num_actions, LR_C):
        self.scope = scope
        self.num_state_features = num_state_features
        self.optimizer = tf.train.AdamOptimizer(LR_C)
        self.loss_vec = []
        self.lstm_input_size = 200
        with tf.variable_scope(self.scope + '/main_critic'):
            self.state_input = tf.placeholder(tf.float32, [None, num_state_features], name="state")
            self.seq_length = tf.placeholder(tf.int32, [None], name="episode_length")
            self.initial_lstm_state = tf.placeholder(
                    tf.float32, [None, 2*self.lstm_input_size], name='initital_state')
            self.Q_train = tf.placeholder(tf.float32, [None], name="target_Q_val")
            hidden_out = tf.nn.tanh(tf.layers.dense(self.state_input, self.lstm_input_size, name="CH1"))
            with tf.variable_scope('lstm_layer') as vs:
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                            self.lstm_input_size, state_is_tuple=True, forget_bias=1.0)
                batch_size = tf.shape(self.seq_length)[0]
                hidden_reshaped = tf.reshape(hidden_out,
                        [batch_size, -1, hidden_out.get_shape().as_list()[-1]])
                state_tuple = tf.contrib.rnn.LSTMStateTuple(
                            *tf.split(self.initial_lstm_state, 2, 1))

                self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
                            lstm_cell,
                            hidden_reshaped,
                            initial_state=state_tuple,
                            sequence_length=self.seq_length,
                            time_major=False)
                self.lstm_state = tf.concat(self.lstm_state, 1)
                ox = tf.reshape(self.lstm_outputs, [-1,self.lstm_input_size], name='reshaped_lstm_outputs')
                
            self.Q = tf.layers.dense(ox, num_actions, name="CH2")
            self.act_t = tf.placeholder(tf.int32, [None], name="action")
            q_t_selected = tf.reduce_sum(self.Q * tf.one_hot(self.act_t, num_actions), 1)
            self.loss = tf.losses.mean_squared_error(self.Q_train, q_t_selected)
            
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/main_critic')
        self.grads = tf.gradients(self.loss, self.params)
        self.clipped_grads = [tf.clip_by_norm(g, 0.1/LR_C) for g in self.grads]
        self.update_op = self.optimizer.apply_gradients(zip(self.clipped_grads, self.params))

        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

        #self.main_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/main_critic')
        #self.target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/target_critic')
        #self.sync_op = [t_p.assign(m_p) for t_p, m_p in zip(self.target_params, self.main_params)]

    '''
    def sync(self): #Copy the main critic's weights into the target critic
        self.sess.run(self.sync_op)
        #print("AFTER SYNC: ", self.sess.run([self.target_params]))
    '''
    
    def batch_update(self, x_t_vec, op_neuron_sel_t_vec, target_t_vec, ep_lens): #Updates only the main critic
        #ep_lens = [ep1_len, ep2_len, ...]
        num_episodes = len(ep_lens)
        reset_lstm_start_state = np.zeros([1, 2*self.lstm_input_size])
        #batch contains multiple episodes, but the self.initial_lstm_state needs to be reset at the beginning of each

        feed_dict = []
        end_idx = -1
        for i in range(num_episodes):
            start_idx = end_idx + 1
            end_idx += ep_lens[i]
            feed_dict.append( {self.state_input: x_t_vec[start_idx : end_idx + 1],
                               self.act_t: op_neuron_sel_t_vec[start_idx : end_idx + 1],
                               self.Q_train: target_t_vec[start_idx : end_idx + 1],
                               self.seq_length: [len(x_t_vec[start_idx : end_idx + 1])],
                               self.initial_lstm_state: reset_lstm_start_state})
        num_steps = 20
        for it in range(num_steps):
            net_loss = 0
            for i in range(num_episodes):
                _, loss = self.sess.run([self.update_op, self.loss], feed_dict[i])
                net_loss += loss
            self.loss_vec.append(net_loss)
        self.value_loss = np.mean(self.loss_vec[-100:])


    def run_main(self, s, lstm_state):
        #Input: env. state (s), lstm_state (if None, assume reset)
        #Output: [Q(s,a1), Q(s,a2), ... ] and lstm_state
        if lstm_state is None:
            lstm_state_feed = np.zeros([1, 2*self.lstm_input_size])
        else:
            lstm_state_feed = lstm_state
        feed_dict = {self.state_input: [s],
                     self.seq_length: [1],
                     self.initial_lstm_state: lstm_state_feed}
        Q_vals, lstm_state_out =  self.sess.run([self.Q, self.lstm_state], feed_dict)
        return Q_vals[0], lstm_state_out
        

class ActorNetwork:
    def __init__(self, scope, num_state_features, num_actions, LR_A, BETA): 
        self.scope = scope
        self.optimizer = tf.train.AdamOptimizer(LR_A)
        self.num_state_features = num_state_features
        self.lstm_input_size = 200
        with tf.variable_scope(self.scope + '/actor'):
            self.state_input = tf.placeholder(tf.float32, [None, num_state_features], name="state")
            self.seq_length = tf.placeholder(tf.int32, [None], name="num_steps")
            self.initial_lstm_state = tf.placeholder(
                    tf.float32, [None, 2*self.lstm_input_size], name='initital_state')
            self.act_train = tf.placeholder(tf.uint8, [None])
            self.adv_seq = tf.placeholder(tf.float32, [None])
            hidden_out = tf.nn.tanh(tf.layers.dense(self.state_input, self.lstm_input_size, name="AH1"))
            with tf.variable_scope('lstm_layer') as vs:
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                            self.lstm_input_size, state_is_tuple=True, forget_bias=1.0)
                batch_size = tf.shape(self.seq_length)[0]
                hidden_reshaped = tf.reshape(hidden_out,
                        [batch_size, -1, hidden_out.get_shape().as_list()[-1]])
                state_tuple = tf.contrib.rnn.LSTMStateTuple(
                            *tf.split(self.initial_lstm_state, 2, 1))

                self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
                            lstm_cell,
                            hidden_reshaped,
                            initial_state=state_tuple,
                            sequence_length=self.seq_length,
                            time_major=False)
                self.lstm_state = tf.concat(self.lstm_state, 1)
                logits = tf.reshape(self.lstm_outputs, [-1,self.lstm_input_size], name='reshaped_lstm_outputs')

            self.pdtype = make_pdtype(Discrete(num_actions))
            self.pd, self.pi = self.pdtype.pdfromlatent(logits, init_scale=0.01)
            with tf.name_scope('sample_a'):
                self.A_sample = self.pd.sample()
            with tf.name_scope('best_a'):
                self.A_best = self.pd.mode()
            with tf.name_scope('distribution_a'):
                self.A_dist = self.pd.mean
            neglogpac = self.pd.neglogp(self.act_train)
            pg_loss = self.adv_seq * neglogpac
            entropy = self.pd.entropy()
            self.loss = tf.reduce_mean(pg_loss - entropy * BETA)
            
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        self.grads = tf.gradients(self.loss, self.params)
        self.clipped_grads = [tf.clip_by_norm(g, 0.1/LR_A) for g in self.grads]
        self.update_op = self.optimizer.apply_gradients(zip(self.clipped_grads, self.params))
        
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

    def batch_update(self, x_t_vec, op_neuron_sel_t_vec, adv_t_vec, ep_lens):
        num_episodes = len(ep_lens)
        reset_lstm_start_state = np.zeros([1, 2*self.lstm_input_size])
        feed_dict = []
        end_idx = -1
        for i in range(num_episodes):
            start_idx = end_idx + 1
            end_idx += ep_lens[i]
            feed_dict.append( {self.state_input: x_t_vec[start_idx : end_idx + 1],
                               self.act_train: op_neuron_sel_t_vec[start_idx : end_idx + 1],
                               self.adv_seq: adv_t_vec[start_idx : end_idx + 1],
                               self.seq_length: [len(x_t_vec[start_idx : end_idx + 1])],
                               self.initial_lstm_state: reset_lstm_start_state})
        num_steps = 20
        for it in range(num_steps):
            for i in range(num_episodes):
                self.sess.run(self.update_op, feed_dict[i])
        
        
    def run(self, s, lstm_state):
        if lstm_state is None:
            lstm_state_feed = np.zeros([1, 2*self.lstm_input_size])
        else:
            lstm_state_feed = lstm_state
        feed_dict = {self.state_input: [s],
                     self.seq_length: [1],
                     self.initial_lstm_state: lstm_state_feed}
        [a_sample, a_best, a_distribution, lstm_state_out] = self.sess.run([self.A_sample, self.A_best, self.A_dist, self.lstm_state], feed_dict)
        return a_sample[0], a_best[0], a_distribution[0], lstm_state_out
    
'''
if __name__ == '__main__':
    cn = CriticNetwork("critic", 1, 1)
    s = [[1],[2],[3],[4],[5],[3],[7],[2]]
    #a = [0, 1, 0, 1, 0]
    #t = [s[i][0]**2/3 if a[i]==0 else  s[i][0]/5. for i in range(len(s))]
    a=[0,0,0,0,0,0,0,0]
    t=[s[i][0]**2/3 for i in range(len(s))]
    el = [5,3]
    cn.batch_update(s,a,t,el)
    print(t)
    hs = None
    for i in range(len(s)):
        k = s[i]
        if i==el[0]:
            hs = None
        q,hs = cn.run_main(k, hs)
        print(k, q)
'''
