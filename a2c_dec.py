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

import sys
import numpy as np
import gym
from ac_nets import *

LR_C = float(sys.argv[1])
LR_A = float(sys.argv[2])
BETA = float(sys.argv[3])
#print ("LR_C = ", LR_C, "LR_A = ", LR_A, "BETA = ", BETA)

GAMMA = 0.9
NUM_EPISODES = 50000
STEPS_PER_EPISODE = 30
DEBUG = False
MEM = True 
STATE = False
PREDICT = True
env = gym.make("Org-v0")

n_features = 2 #env.observation_space.shape[0]
actor_actions = 3 #env.action_space.n
critic_actions = 9
critic1 = CriticNetwork("main1", n_features, critic_actions, LR_C)
critic2 = CriticNetwork("main2", n_features, critic_actions, LR_C)
actor1 = ActorNetwork("act1", n_features, actor_actions, LR_A, BETA)
actor2 = ActorNetwork("act2", n_features, actor_actions, LR_A, BETA)

experience_buffer = []
reward_lst = []

even_dist = True
filterAction1, filterAction2 = [[0.8,0.1,0.1],[0.6,0.2,0.2],[0.4,0.3,0.3]],[[0.8,0.1,0.1],[0.6,0.2,0.2],[0.4,0.3,0.3]]
prior1, prior2 = [], []
filters1, filters2 = [[0.8,0.6,0.4],[0.1,0.2,0.3],[0.1,0.2,0.3]], [[0.8,0.6,0.4],[0.1,0.2,0.3],[0.1,0.2,0.3]]
fil1, fil2 = [], []
results1, results2 = [], []
num_model1, num_model2 = 3, 3
p_obs1, p_obs2 = [0,0,0], [0,0,0]

for ep in range(NUM_EPISODES):
    #print (ep, "/", NUM_EPISODES)
    s = env.reset()
    done = False
    ep_r = 0

    if (MEM):
        temp_s = []
        for i in range(len(s)):
            if (s[i] == 1):
                temp_s.append(i)
        o1 = temp_s[0]
        o2 = temp_s[1]

    a1 = actor1.sample_action( s )
    a2 = actor2.sample_action( s )
    a = a1*3 + a2%3 

    if (a1 == 0):
        p_obs1 = [0.8, 0.1, 0.1]
    if (a1 == 1):
        p_obs1 = [0.1, 0.8, 0.1]
    if (a1 == 2):
        p_obs1 = [0.1, 0.1, 0.8]
    if (a2 == 0):
        p_obs2 = [0.8, 0.1, 0.1]
    if (a2 == 1):
        p_obs2 = [0.1, 0.8, 0.1]
    if (a2 == 2):
        p_obs2 = [0.1, 0.1, 0.8]
    
    for step in range(STEPS_PER_EPISODE):
        s_, r, done, info = env.step(a) # make step in environment
        r1 = r
        r2 = r 

        if (MEM):
            temp_s_ = []
            for i in range(len(s_)):
                if (s_[i] == 1):
                    temp_s_.append(i)
            o1_ = temp_s_[0]
            o2_ = temp_s_[1]
        else:    
            s_ = s_

        a1_ = actor1.sample_action( s_ )
        a2_ = actor2.sample_action( s_ )
        a_ = a1_*3 + a2_%3
        
        if PREDICT:
            for n in range(num_model1):
                if even_dist:
                    prior1.append(round(1.0/num_model1, 2))
                    prior2.append(round(1.0/num_model2, 2))
                else:
                    prior1 = bprime1
                    prior2 = bprime2

                temp1,temp2 = [], []

            belief1 = np.zeros((len(prior1), len(prior1)))
            belief2 = np.zeros((len(prior2), len(prior2)))

            for i in range(len(belief1)):
                belief1[i][i] = prior1[i]
                belief2[i][i] = prior2[i]

            result1, result2 = [], []

            for fil1 in filters1:
                result1.append(np.matmul(belief1, fil1))
            for fil2 in filters2:
                result2.append(np.matmul(belief2, fil2))

            bp1 = np.zeros(len(filterAction1))
            bp2 = np.zeros(len(filterAction2))

            for i in range(len(p_obs1)):
                bp1 = np.add(bp1, [(p_obs1[i] * j) for j in result1[i]])
                bp2 = np.add(bp2, [(p_obs2[i] * j) for j in result2[i]])

            bprime1 = np.zeros(len(prior1))
            bprime2 = np.zeros(len(prior2))

            for i in range(len(bprime1)):
                bprime1[i] = bp1[i] / np.sum(bp1)
                bprime2[i] = bp2[i] / np.sum(bp2)
            prediction1 = np.zeros(len(filterAction1))
            prediction2 = np.zeros(len(filterAction2))
            for i in range(len(prediction1)):
                for j in range(len(bprime1)):
                    prediction1[i] = prediction1[i] + bprime1[j] * filterAction1[j][i]
                    prediction2[i] = prediction2[i] + bprime2[j] * filterAction2[j][i]

            ap1 = np.random.choice(np.arange(len(filterAction1)), p=prediction1)
            ap2 = np.random.choice(np.arange(len(filterAction2)), p=prediction2)

            for i in range(len(bprime1)):
                bprime1[i] = round(bp1[i] / np.sum(bp1), 2)
                bprime2[i] = round(bp2[i] / np.sum(bp2), 2)

            even_dist = False

            ap = ap1*2 + ap2%2
            experience_buffer.append([o1,o2,a1,a2,a,r1,r2,o1_,o2_,ap1,ap2,ap])
        else:
            experience_buffer.append([o1,o2,a1,a2,a,r1,r2,s_,s_,o1_,o2_,a_])     
        o1, o2, a1, a2, a = o1_, o2_, a1_, a2_, a_
        ep_r += r
        
        if done:
            break

    x_vec, neuron_sel_vec1, neuron_sel_vec2, neuron_sel_vec, critic_target_vec1, critic_target_vec2, actor_adv_vec1, actor_adv_vec2 = [], [], [], [], [], [], [], []

    if ep % 10 == 0:
        for (state1, state2, action1, action2, action, reward1, reward2, next_state1, next_state2, next_action1, next_action2, next_action) in experience_buffer:
            if MEM:
                ob = np.array([1.0 if (i == state1 or i == state2) else 0.0 for i in range(6)])
                next_ob = np.array([1.0 if (i == next_state1 or i == next_state2) else 0.0 for i in range(6)])
            elif STATE:
                ob = np.array([1.0 if (i == state1 or i == state2) else 0.0 for i in range(13)])
                next_ob = np.array([1.0 if (i == next_state1 or i == next_state2) else 0.0 for i in range(13)])
            x_vec.append( ob )
            neuron_sel_vec1.append( action1 )
            neuron_sel_vec2.append( action2 )
            neuron_sel_vec.append( action )
            Q_next1 = critic1.run_main( next_ob )[ next_action ] #not using target critic
            Q_next2 = critic2.run_main( next_ob )[ next_action ] #not using target critic
            critic_target_vec1.append( reward1 + GAMMA * Q_next1 )
            critic_target_vec2.append( reward2 + GAMMA * Q_next2 )
            
        critic1.batch_update( x_vec, neuron_sel_vec, critic_target_vec1 )
        critic2.batch_update( x_vec, neuron_sel_vec, critic_target_vec2 )

        for (state1, state2, action1, action2, action, reward1, reward2, next_state1, next_state2, next_action1, next_action2, next_action) in experience_buffer:
            if MEM:
                ob = np.array([1.0 if (i == state1 or i == state2) else 0.0 for i in range(6)])
            elif STATE:
                ob = np.array([1.0 if (i == state1 or i == state2) else 0.0 for i in range(13)])
            Q1 = critic1.run_main( ob )
            Q2 = critic2.run_main( ob )
            Q_cur1 = Q1[ action ]
            Q_cur2 = Q2[ action ]
            d1 = actor1.action_distribution( ob )
            d2 = actor2.action_distribution( ob )

            dist = [d1[0]*d2[0], d1[0]*d2[1], d1[0]*d2[2], d1[1]*d2[0], d1[1]*d2[1], d1[1]*d2[2], d1[2]*d2[0], d1[2]*d2[1], d1[2]*d2[2]]

            V1 = np.dot( Q1, dist )
            V2 = np.dot( Q2, dist )
            actor_adv_vec1.append( Q_cur1 - V1 )
            actor_adv_vec2.append( Q_cur2 - V2 )

        actor1.batch_update( x_vec, neuron_sel_vec1, actor_adv_vec1 )
        actor2.batch_update( x_vec, neuron_sel_vec2, actor_adv_vec2 )
        experience_buffer = []

    reward_lst.append(ep_r)

    if (ep %1000 == 0):
        print(ep, np.mean(reward_lst[-500:]), critic1.value_loss, critic2.value_loss)
        
    #if ep % 50 == 0:
    #    critic.sync()
