import tensorflow as tf
import numpy as np
import sys

sess = tf.Session()
even_dist = True
action = []
prior = []
filters = []
fil = []
results = []
num_model = 0
with open("models.txt") as file:
    data = file.readlines()
    for line in data:
        action.append(map(float, line.split()))
        num_model = num_model + 1

b = tf.placeholder(tf.float32)
f = tf.placeholder(tf.float32)

input_belief = tf.reshape(b, [1, 3, 3, 1])
kernel = tf.reshape(f, [1, 3, 1, 1])

output = tf.squeeze(tf.nn.conv2d(input_belief, kernel, strides=[1,1,1,1], padding='VALID'))

for time in range(3):
    for n in range(num_model):
        if even_dist:
            prior.append(1.0/num_model)
        else:
            prior = bprime

        temp = []
        for m in range(num_model):
            temp.append(action[m][n])
        filters.append(temp)

    belief = np.zeros([len(prior),len(prior)])
    for i in range(len(belief)):
        belief[i][i] = prior[i]

    result = []

    for fil in filters:
        temp_result = sess.run(output, feed_dict = {b: belief, f: fil})
        result.append(temp_result)

    p_obs = [0.4,0.3,0.3]
    bp = np.zeros(len(p_obs))
    for i in range(len(p_obs)):
        bp = np.add(bp, [(p_obs[i] * j) for j in result[i]])

    bprime = np.zeros(len(prior))

    for i in range(len(bprime)):
        bprime[i] = bp[i] / (np.sum(bp))

    even_dist = False

    prediction = np.zeros(len(action))
    for i in range(len(prediction)):
        for j in range(len(bprime)):
            prediction[i] = prediction[i] + bprime[j]*action[j][i]

    p_action = np.random.choice(np.arange(len(action)), p=prediction)
 