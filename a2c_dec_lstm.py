'''====================================================================================
Implementation of A2C with LSTM to test Actor-Critic Recurrent Network classes on RL
envs where episodes have a fixed length.

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

import sys
import numpy as np
import gym
from ac_lstm_nets import *

LR_C = float(sys.argv[1])
LR_A = float(sys.argv[2])
BETA = float(sys.argv[3])
#print ("LR_C = ", LR_C, "LR_A = ", LR_A, "BETA = ", BETA)

GAMMA = 0.9
NUM_EPISODES = 35000
STEPS_PER_EPISODE = 30

env = gym.make("Org-v0")
PREDICT = True
n_features = 2
actor_actions = 3
critic_actions = 9
critic1 = CriticNetwork("main1", n_features, critic_actions, LR_C)
critic2 = CriticNetwork("main2", n_features, critic_actions, LR_C)
actor1 = ActorNetwork("act1", n_features, actor_actions, LR_A, BETA)
actor2 = ActorNetwork("act2", n_features, actor_actions, LR_A, BETA)

experience_batch = []
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
    s = env.reset()

    episode_vec = []
    done = False
    ep_r = 0

    lstm_actor_cur_state1 = None
    lstm_actor_cur_state2 = None

    a1, _, _, lstm_actor_cur_state1 = actor1.run( [s],  lstm_actor_cur_state1 )
    a2, _, _, lstm_actor_cur_state2 = actor2.run( [s],  lstm_actor_cur_state2 )
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

        if step < STEPS_PER_EPISODE-1:
            a1_, _, _, lstm_actor_cur_state1 = actor1.run( [s_], lstm_actor_cur_state1 )
            a2_, _, _, lstm_actor_cur_state2 = actor2.run( [s_], lstm_actor_cur_state2 )
            a_ = a1_*3 + a2_%3
            episode_vec.append([s,s,a1,a2,r1,r2,s_,s_,a1_,a2_])

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
            experience_batch.append([s,s,a1,a2,r1,r2,s_,s_,ap1,ap2])
        else:
            experience_batch.append(episode_vec)
        s, a, a1, a2 = s_, a_, a1_, a2_
        ep_r += r

        if done or (step == STEPS_PER_EPISODE - 1):
            #p = random.random()
            #if (p < (1 - len(experience_batch)/BATCH_SIZE)):
            #experience_batch.append(episode_vec)
            break

    if (ep+1) % 100 == 0: #Update networks every 100 episodes
        x_vec1, x_vec2, neuron_sel_vec, neuron_sel_vec1, neuron_sel_vec2, critic_target_vec1, critic_target_vec2, ep_len_vec = [], [], [], [], [], [], [], []
        for episode in experience_batch:
            lstm_critic_state1 = None
            lstm_critic_state2 = None
            s0_1 = episode[0]
            s0_2 = episode[0]
            _, lstm_critic_state1 = critic1.run_main( [s0_1], lstm_critic_state1 )
            _, lstm_critic_state2 = critic2.run_main( [s0_2], lstm_critic_state2 )
            ep_len_ctr = 0
            for (observation1, observation2, action1, action2, reward1, reward2, next_observation1, next_observation2, next_action1, next_action2) in episode:
                ep_len_ctr += 1
                action = action1*3 + action2%3
                next_action = next_action1*3 + next_action2%3
                x_vec1.append( [observation1] )
                x_vec2.append( [observation2] )
                neuron_sel_vec.append( action )
                neuron_sel_vec1.append( action1 )
                neuron_sel_vec2.append( action2 )
                Q_next_vals1, lstm_critic_state1 = critic1.run_main( [next_observation1], lstm_critic_state1 ) #not using target critic
                Q_next_vals2, lstm_critic_state2 = critic2.run_main( [next_observation2], lstm_critic_state2 ) #not using target critic
                Q_next1 = Q_next_vals1[ next_action ]
                Q_next2 = Q_next_vals2[ next_action ]
                critic_target_vec1.append( reward1 + GAMMA * Q_next1 )
                critic_target_vec2.append( reward2 + GAMMA * Q_next2 )
            ep_len_vec.append( ep_len_ctr )
        critic1.batch_update( x_vec1, neuron_sel_vec, critic_target_vec1, ep_len_vec )
        critic2.batch_update( x_vec2, neuron_sel_vec, critic_target_vec2, ep_len_vec )
        
        actor_adv_vec1, actor_adv_vec2 = [], []
        for episode in experience_batch:
            lstm_actor_state1, lstm_actor_state2, lstm_critic_state1, lstm_critic_state2 = None, None, None, None
            for (observation1, observation2, action1, action2, reward1, reward2, next_observation1, next_observation2, next_action1, next_action2) in episode:
                Q1, lstm_critic_state1 = critic1.run_main( [observation1], lstm_critic_state1 )
                Q2, lstm_critic_state2 = critic2.run_main( [observation2], lstm_critic_state2 )
                action = action1*3 + action2%3
                Q_cur1 = Q1[ action ]
                Q_cur2 = Q2[ action ]
                _, _, dist1, lstm_actor_state1 = actor1.run( [observation1], lstm_actor_state1 )
                _, _, dist2, lstm_actor_state2 = actor2.run( [observation2], lstm_actor_state2 )

                dist = [dist1[0]*dist2[0], dist1[0]*dist2[1], dist1[0]*dist2[2], dist1[1]*dist2[0], dist1[1]*dist2[1], dist1[1]*dist2[2], dist1[2]*dist2[0], dist1[2]*dist2[1], dist1[2]*dist2[2]]
                if (abs(sum(dist) - 1.) > 0.000001):
                    print("ERROR: ", ep, step, dist)
                V1 = np.dot( Q1, dist )
                V2 = np.dot( Q2, dist )
                actor_adv_vec1.append( Q_cur1 - V1 )
                actor_adv_vec2.append( Q_cur2 - V2 )
        actor1.batch_update( x_vec1, neuron_sel_vec1, actor_adv_vec1, ep_len_vec )
        actor2.batch_update( x_vec2, neuron_sel_vec2, actor_adv_vec2, ep_len_vec )
        experience_batch.clear()
        
    reward_lst.append(ep_r)
    if ((ep+1) %500 == 0):
        print(ep, np.mean(reward_lst[-300:]), critic1.value_loss, critic2.value_loss)
        
    #if ep % 100 == 0:
    #    critic.sync()

