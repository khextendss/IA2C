'''====================================================================================
Implementation of A2C to test Actor-Critic Network classes on RL envs where
episodes have a fixed length.

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

import numpy as np
import scipy as sp
import gym
from ac_nets import *
from itertools import combinations_with_replacement
from operator import add
from random import randrange

GAMMA = 0.9
NUM_AGENTS = 2
NUM_EPISODES = 50000
STEPS_PER_EPISODE = 30
NOISE = 20
DEBUG = False

env = gym.make("Org-v1")

n_features = 1
actor_actions = 3 
critic_actions = 6
critic, actor, D = [], [], []
for i in range(NUM_AGENTS):
    critic.append([])
    actor.append([])
    D.append([1,1,1])

exp_buff, reward_lst = [], []
RT = [4,1,0]
D = (0,0,0)

joints = list(combinations_with_replacement("012", NUM_AGENTS))
all_cf = []

for i in range(len(joints)):
    all_cf.append([0,0,0])

for i in range(len(joints)):
    for j in range(len(joints[i])):
        all_cf[i][int(joints[i][j])]+= 1

#return configuration index        
def cf_to_index(cf):
    index = all_cf.index(cf)
    return index

#map joint action to corresponding configuration
def ja_to_cf(ja):
    cf = [0,0,0]
    for i in range(len(ja)):
        cf[ja[i]]+= 1

    return cf

#add noise to private observation    
def add_noise(action, noise):
    obs = [0,0,0]
    
    for i in range(3):
        count = 0
        while count < action[i]:
            count += 1
            random = randrange(100)
            if random < noise/2:
                obs[(i+1)%3] += 1
            elif random >= noise/2 and random < noise:
                obs[(i+2)%3] += 1
            else:
                obs[i] += 1
    return obs

#get configuration distribution from current Dirichlet distribution    
def get_dist(dist):
    count = 0
    action_dist = []
    for i in range(len(joints)):
        action_dist.append(1)
    for i in joints:    
        for j in i:
            action_dist[count] = action_dist[count] * dist[int(j)]
        count += 1

    s = sum(action_dist)

    for i in range(len(action_dist)):
        action_dist[i] = action_dist[i]/s

    return action_dist

for i in range(NUM_AGENTS):
    critic[i] = CriticNetwork("main"+str(i), n_features, critic_actions)
    
for i in range(NUM_AGENTS):
    actor[i] = ActorNetwork("act"+str(i), n_features, actor_actions)

for ep in range(NUM_EPISODES):
    s, _, _, _ = env.reset()
    done = False
    ep_r = 0
    s = [s]
    ja = []
    for i in range(NUM_AGENTS):
        ja.append(actor[i].sample_action( s ))
        
    cf = ja_to_cf(ja)

    for step in range(STEPS_PER_EPISODE):
        s_, gr, done, info = env.step(cf) # make step in environment
        s_ = [s_] 
        ja_ = []
        r = []
        for i in range(NUM_AGENTS):
            ja_.append(actor[i].sample_action( s_ ))
            r.append(RT[ja[i]]+gr)
        if DEBUG:
            print ("ja = ", cf, "s = ", s_)
        
        cf_ = ja_to_cf(ja_)
        obs = add_noise(cf, NOISE)
                
        d = list(map(add, list(D), obs))
        D = (d[0], d[1], d[2])
        alpha = [0,0,0]
        
        for i in range(3):
            alpha[i] = round(D[i]*1.28571 - D[(i+1)%3]*0.142857 - D[(i+2)%3]*0.142857)
            if alpha[i] <= 0:
                alpha[i] = 0.1
                
        pd = np.random.dirichlet(alpha, 1)
        pd = pd[0]        
        pa = np.random.multinomial(NUM_AGENTS, pd, 1)
        pa = pa[0].tolist()       
        
        exp_buff.append([s,ja,cf,r,s_,ja_,pa])
        ep_r += sum(r)
        s, ja, cf = s_, ja_, cf_

        if done:
            break

    dist = get_dist(pd)
    x_vec, neuron_cf_vec, actor_adv_vec, critic_target_vec, neuron_sel_vec, Q_next = [], [], [], [], [], []
    for i in range(NUM_AGENTS):
        actor_adv_vec.append([])
        critic_target_vec.append([])
        neuron_sel_vec.append([])
        Q_next.append([])

    if ep % 10 == 0:
        #Critic Update
        for (state, action, configuration, rewards, next_state, next_action, next_configuration) in exp_buff:
            x_vec.append( state )
            for i in range(NUM_AGENTS):
                neuron_sel_vec[i].append(action[i])
                Q_next[i] = critic[i].run_main( next_state )[ next_action[i] ]
                critic_target_vec[i].append( rewards[i] + GAMMA * Q_next[i] )
            neuron_cf_vec.append( cf_to_index(configuration) )
                
        for i in range(NUM_AGENTS):
            critic[i].batch_update( x_vec, neuron_cf_vec, critic_target_vec[i] )

        #Actor Update
        for (state, action, configuration, rewards, next_state, next_action, next_configuration) in exp_buff:
            Q = [None] * NUM_AGENTS
            Q_cur = [None] * NUM_AGENTS
            V = [None] * NUM_AGENTS
            for i in range(NUM_AGENTS):
                Q[i] = critic[i].run_main( state )
                Q_cur[i] = Q[i][ action[i] ]
                V[i] = np.dot( Q[i], dist )
                actor_adv_vec[i].append( Q_cur[i] - V[i] )

        for i in range(NUM_AGENTS):
            actor[i].batch_update( x_vec, neuron_sel_vec[i], actor_adv_vec[i] )
        exp_buff = []
        
    reward_lst.append(ep_r)
    
    if (ep %1000 == 0):
        for i in range(NUM_AGENTS):
            #print(ep, np.mean(reward_lst[-500:]), critic[i].value_loss)
        

    #if ep % 50 == 0:
    #    critic.sync()

