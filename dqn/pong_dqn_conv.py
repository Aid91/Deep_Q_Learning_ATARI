
import time
import numpy as np
import tensorflow as tf
import numpy.random as rd
import gym
import cv2

import matplotlib.pyplot as plt
from tf_tools import variable_summaries
from replay_memory import *


env = gym.make("Pong-v0")

mini_batch_size = 32
replay_memory_size = 100000
replay_memory = ReplayMemory(replay_memory_size)

should_render = False
double_q_learning = False

# 0-> no movement, 2->UP, 3->DOWN
n_action = 3
action_list = [0, 2, 3]

print_per_episode = 1
n_train_trials = 4000
n_test_trials = 10
trial_duration = 10000
gamma = 0.99
learning_rate = 0.00025


state_holder = tf.placeholder("float", [None, 84, 84, 4])
target_placeholder = tf.placeholder(tf.float32, shape=(None,))
actions_placeholder = tf.placeholder(tf.float32, shape=(None, n_action))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

with tf.name_scope('conv_target_Q_network'):
    # network weights
    W_conv_tar1 = weight_variable([8, 8, 4, 32])
    b_conv_tar1 = bias_variable([32])

    W_conv_tar2 = weight_variable([4, 4, 32, 64])
    b_conv_tar2 = bias_variable([64])

    W_conv_tar3 = weight_variable([3, 3, 64, 64])
    b_conv_tar3 = bias_variable([64])

    W_fc_tar1 = weight_variable([7*7*64, 512])
    b_fc_tar1 = bias_variable([512])

    W_fc_tar2 = weight_variable([512, n_action])
    b_fc_tar2 = bias_variable([n_action])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(state_holder, W_conv_tar1, 4) + b_conv_tar1)

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv_tar2, 2) + b_conv_tar2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv_tar3, 1) + b_conv_tar3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 7*7*64])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc_tar1) + b_fc_tar1)

    # output layer
    target_Q= tf.matmul(h_fc1, W_fc_tar2) + b_fc_tar2

with tf.name_scope('conv_Q_network'):
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([7 * 7 * 64, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, n_action])
    b_fc2 = bias_variable([n_action])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(state_holder, W_conv1, 4) + b_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 64])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # output layer
    Q= tf.matmul(h_fc1, W_fc2) + b_fc2

    # defining the placeholders and ops for copying the parameters for Q to target-Q network
    W_update_placeholder = tf.placeholder(W_conv1.dtype, shape=W_conv1.get_shape())
    W_update_placeholder_op = W_conv_tar1.assign(W_update_placeholder)

    w_update_placeholder = tf.placeholder(W_conv3.dtype, shape=W_conv3.get_shape())
    w_update_placeholder_op = W_conv_tar3.assign(w_update_placeholder)

    b_update_placeholder = tf.placeholder(b_conv3.dtype, shape=b_conv3.get_shape())
    b_update_placeholder_op = b_conv_tar3.assign(b_update_placeholder)

    bhid_update_placeholder = tf.placeholder(b_conv1.dtype, shape=b_conv1.get_shape())
    bhid_update_placeholder_op = b_conv_tar1.assign(bhid_update_placeholder)

    W2_update_placeholder = tf.placeholder(W_conv2.dtype, shape=W_conv2.get_shape())
    W2_update_placeholder_op = W_conv_tar2.assign(W2_update_placeholder)

    b2_update_placeholder = tf.placeholder(b_conv2.dtype, shape=b_conv2.get_shape())
    b2_update_placeholder_op = b_conv_tar2.assign(b2_update_placeholder)

    W_fc1_update_placeholder =  tf.placeholder(W_fc1.dtype, shape=W_fc1.get_shape())
    W_fc1_update_placeholder_op = W_fc_tar1.assign(W_fc1_update_placeholder)

    b_fc1_update_placeholder = tf.placeholder(b_fc1.dtype, shape=b_fc1.get_shape())
    b_fc1_update_placeholder_op = b_fc_tar1.assign(b_fc1_update_placeholder)

    W_fc2_update_placeholder = tf.placeholder(W_fc2.dtype, shape=W_fc2.get_shape())
    W_fc2_update_placeholder_op = W_fc_tar2.assign(W_fc2_update_placeholder)

    b_fc2_update_placeholder = tf.placeholder(b_fc2.dtype, shape=b_fc2.get_shape())
    b_fc2_update_placeholder_op = b_fc_tar2.assign(b_fc2_update_placeholder)

    variable_summaries(Q, '/Q_values')

    max_Q = tf.reduce_max(Q, reduction_indices=1)
    variable_summaries(max_Q, '/max_Q')

    acc_episode_reward_placeholder = tf.placeholder(dtype=tf.float32)
    variable_summaries(acc_episode_reward_placeholder, '/acc_episode_reward')


with tf.name_scope('loss'):
    prediction = tf.reduce_sum(tf.mul(Q, actions_placeholder), 1)
    error = tf.reduce_mean((tf.square(target_placeholder - prediction)))
    variable_summaries(prediction, '/prediction')
    variable_summaries(target_placeholder, '/target')
    variable_summaries(error, '/error')

with tf.name_scope('train'):
    # We define the optimizer to use the RMSProp optimizer, and ask it to minimize our loss
    training_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, epsilon=1e-6).minimize(error)

# train step used in the test without experience replay
# defines sampling and generating actions, states and targets
def train_step_no_experience(state,action,reward,next_state,done):
    targets = []
    stats = []
    actns = []

    target = 0
    if done:
        target += reward
    else:
        Q_values = sess.run(target_Q, feed_dict={state_holder: next_state.reshape(1, 84, 84, 4)})
        val = np.max(Q_values[0, :])
        target += reward + gamma * val

    action_t = np.zeros(n_action)
    action_t[action] = 1

    actns.append(action_t)
    stats.append(state)
    targets.append(target)

    return actns, stats, targets

# train step used in the real test with experience replay
# defines sampling and generating actions, states and targets
def train_step():
        targets = []
        stats = []
        actns = []

        if replay_memory.length() > mini_batch_size:
            states, actions, rewards, next_states, terminals = replay_memory.sample(mini_batch_size)


            for i in range(len(terminals)):
                target = 0
                if terminals[i]:
                     target += rewards[i]
                else:
                    Q_values = sess.run(target_Q, feed_dict={state_holder: next_states[i].reshape(1,84,84,4)})
                    val = np.max(Q_values[0, :])
                    target += rewards[i] + gamma * val

                action_t = np.zeros(n_action)
                action_t[actions[i]] = 1

                actns.append(action_t)
                stats.append(states[i])
                targets.append(target)

        return actns, stats, targets

# definition of epsilon-greedy policy
def policy(state):
    '''
    This should implement an epsilon greedy policy:
    - with probability epsilon take a random action
    - otherwise take an action the action that maximizes Q(s,a) with s the current state

    :param Q_table:
    :param state:
    :return:
    '''
    if rd.rand() < epsilon:
        return rd.randint(0, n_action)

    Q_values = sess.run(Q,
                        feed_dict={state_holder: state.reshape(1,84,84,4)
                        })
    val = np.max(Q_values[0, :])
    max_indices = np.where(Q_values[0, :] == val)[0]
    return rd.choice(max_indices)

sess = tf.Session()  # FOR NOW everything is symbolic, this object has to be called to compute each value of Q

# SUMMARIES
merged = tf.merge_all_summaries()
suffix = time.strftime('%Y-%m-%d--%H-%M-%S')
train_writer = tf.train.SummaryWriter('tensorboard/pong/{}'.format(suffix) + '/train', sess.graph)
test_writer = tf.train.SummaryWriter('tensorboard/pong/{}'.format(suffix) + '/test')
# Start

sess.run(tf.initialize_all_variables())

single_episode_rewards = []
episode_rewards_list = []
episode_steps_list = []

step = 0
episode_no = 0

reward_list = []
err_list = []
val_list = []
err_sum_list = []

start_learning = 100000
epsilon_start = 1.
epsilon_end = .1

C = 10000
annealing_period = 500000
update_frequency = 4
acc_t = 0

for k in range(n_train_trials + n_test_trials):

    observation = env.reset()  # Init the state
    error_list = []

    # preproces the initial observation by scaling it to 84x84x1 and then make the binary image
    # stack last 4 frames of the same image at the beginning
    obs = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, obs = cv2.threshold(obs, 100, 255, cv2.THRESH_BINARY)
    processed_observation = np.stack((obs, obs, obs, obs), axis=2)

    for t in range(trial_duration):  # The number of time steps in this game is maximum 200

        acc_t += 1
        step += 1

        epsilon = (epsilon_end +
              max(0., (epsilon_start - epsilon_end)
                  * (annealing_period - max(0., acc_t - start_learning)) / annealing_period))

        if acc_t % 500 == 0:
            print ('T: %d' % acc_t)
            print ('Epsilon: {%.10f}' % epsilon)


        if should_render: env.render()

        action = policy(processed_observation)  # Init the first action

        observation, reward, done, info = env.step(action_list[action])  # Take the action

        # preproces the new observation by scaling it to 84x84x1 and then make the binary image
        obs = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
        _, obs = cv2.threshold(obs, 100, 255, cv2.THRESH_BINARY)
        obs = np.reshape(obs, (84, 84, 1))
        processed_new_observation = np.append(processed_observation[:, :, 1:],obs, axis=2)

        replay_memory.add(state=processed_observation, action=action, reward=reward, next_state=processed_new_observation, is_terminal=done)

        processed_observation = processed_new_observation

        err = 0
        if acc_t > start_learning:

            if acc_t % 500 == 0:
                print ('STARTED TRAINING')


            if t % update_frequency == 0:

                #actions, states, targets = train_step_no_experience(state=processed_new_observation, action=action, done=done, next_state=processed_new_observation, reward=reward)

                actions, states, targets = train_step()

                # Perform one step of gradient descent
                summary, _ = sess.run([merged, training_step], feed_dict={
                state_holder: np.array(states),
                target_placeholder: np.array(targets),
                actions_placeholder: np.array(actions),
                acc_episode_reward_placeholder: np.sum(single_episode_rewards)
                })

                train_writer.add_summary(summary, acc_t)

                # Compute the Error for monitoring
                err = sess.run(error, feed_dict={
                target_placeholder: np.array(targets),
                state_holder:  np.array(states),
                actions_placeholder: np.array(actions),
                })

            # copy the parameters of the Q network to target-Q network
            if acc_t % C == 0:
                sess.run(W_update_placeholder_op, {W_update_placeholder: W_conv1.eval(sess)})
                sess.run(w_update_placeholder_op, {w_update_placeholder: W_conv3.eval(sess)})
                sess.run(b_update_placeholder_op, {b_update_placeholder: b_conv3.eval(sess)})
                sess.run(bhid_update_placeholder_op, {bhid_update_placeholder: b_conv1.eval(sess)})
                sess.run(W2_update_placeholder_op, {W2_update_placeholder: W_conv2.eval(sess)})
                sess.run(b2_update_placeholder_op, {b2_update_placeholder: b_conv2.eval(sess)})

                sess.run(W_fc1_update_placeholder_op, {W_fc1_update_placeholder: W_fc1.eval(sess)})
                sess.run(b_fc1_update_placeholder_op, {b_fc1_update_placeholder: b_fc1.eval(sess)})
                sess.run(W_fc2_update_placeholder_op, {W_fc2_update_placeholder: W_fc2.eval(sess)})
                sess.run(b_fc2_update_placeholder_op, {b_fc2_update_placeholder: b_fc2.eval(sess)})

        err_list.append(err)
        single_episode_rewards.append(reward)


        if done:
            # Done with episode. Reset stuff.

            episode_no += 1
            step = 0

            err_sum_list.append(np.mean(err_list))
            episode_rewards_list.append(np.sum(single_episode_rewards))
            episode_steps_list.append(step)

            single_episode_rewards = []
            err_list = []

            if episode_no % print_per_episode == 0:
                print('ERROR {}'.format(np.mean(err_sum_list)))

                print("Average REWARDS in last {} episodes before episode {}".format(print_per_episode, episode_no),
                      np.mean(episode_rewards_list[(episode_no - print_per_episode):episode_no]), '+-',
                      np.std(episode_rewards_list[(episode_no - print_per_episode):episode_no])
                      )
                print("Average STEPS in last {} episodes before episode {}".format(print_per_episode, episode_no),
                      np.mean(episode_steps_list[(episode_no - print_per_episode):episode_no]), '+-',
                      np.std(episode_steps_list[(episode_no - print_per_episode):episode_no])
                      )
            break

plt.figure()
ax = plt.subplot(121)
ax.plot(range(len(episode_rewards_list)), episode_rewards_list)
ax.set_title("Training rewards")
ax.set_xlabel('Episode number')
ax.set_ylabel('Episde reward')
plt.show()