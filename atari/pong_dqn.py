import time
import tensorflow as tf
import numpy.random as rd
import gym

import numpy as np
import matplotlib.pyplot as plt
from pong_tools import prepro
from tf_tools import variable_summaries
from replay_memory import *


env = gym.make("Pong-v0")
n_hidden = 50

mini_batch_size = 32
replay_memory_size = 1000000
replay_memory = ReplayMemory(replay_memory_size)

should_render = False

# 0-> no movement, 2->UP, 3->DOWN
n_action = 3
action_list = [0, 2, 3]
# The *relative* y coordinate of the opponent and the x,y coordinates of the ball for *two* frames
dim_state = 6

annealing_period = 1000000
update_frequency = 4
print_per_episode = 10
n_train_trials = 5000
n_test_trials = 100
trial_duration = 10000
gamma = 0.99
learning_rate = 0.00025
C = 10000
gradient_momentum = 0.95
K = 300

epsilon_start = 1.
epsilon_end = 0.1
epsilon_decay = 0.9999999

state_holder = tf.placeholder(dtype=tf.float32, shape=(None, dim_state), name='symbolic_state')
target_placeholder = tf.placeholder(tf.float32, shape=(None,))
actions_placeholder = tf.placeholder(tf.float32, shape=(None, n_action))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

with tf.name_scope('target_Q_network'):

    W0 = np.array(rd.randn(dim_state, n_hidden) / np.sqrt(dim_state), dtype=np.float32)
    w0 = np.array(rd.randn(n_hidden, n_action) / np.sqrt(n_hidden), dtype=np.float32)
    bhid0 = np.zeros(n_hidden, dtype=np.float32)
    b0 = np.zeros(n_action, dtype=np.float32)

    # Create the parameters of the target-Q model
    W = tf.Variable(initial_value=W0.reshape(dim_state, n_hidden), trainable=True, name='weight_variable_matrix')
    w = tf.Variable(initial_value=w0.reshape(n_hidden, n_action), trainable=True, name='weight_variable')
    b = tf.Variable(initial_value=b0, trainable=True, name='bias')
    bhid = tf.Variable(initial_value=bhid0, trainable=True, name='bias')

    W20 = np.array(rd.randn(n_hidden, n_hidden) / np.sqrt(n_hidden), dtype=np.float32)
    b20 = np.zeros(n_hidden, dtype=np.float32)
    W2 = tf.Variable(initial_value=W20.reshape(n_hidden, n_hidden), trainable=True, name='weight_variable_matrix')
    b2 = tf.Variable(initial_value=b20, trainable=True, name='bias')

    a_y = tf.matmul(state_holder, W, name='hidden_layer_activation') + bhid
    y = tf.nn.relu(a_y, name='output_of_the_hidden_layer')

    a_y = tf.matmul(y, W2, name='hidden_layer_2_activation') + b2
    y = tf.nn.relu(a_y, name='output_of_the_second_hidden_layer')

    # Q function at the current step
    a_z = tf.matmul(y, w, name='output_activation') + b
    target_Q = tf.nn.softmax(a_z)

with tf.name_scope('Q_network'):

    # Create the parameters of the Q model
    W_q = tf.Variable(initial_value=W0.reshape(dim_state, n_hidden), trainable=True, name='weight_variable_matrix')
    w_q = tf.Variable(initial_value=w0.reshape(n_hidden, n_action), trainable=True, name='weight_variable')
    b_q = tf.Variable(initial_value=b0, trainable=True, name='bias')
    bhid_q = tf.Variable(initial_value=bhid0, trainable=True, name='bias')

    W2_q = tf.Variable(initial_value=W20.reshape(n_hidden, n_hidden), trainable=True, name='weight_variable_matrix')
    b2_q = tf.Variable(initial_value=b20, trainable=True, name='bias')

    a_y_q = tf.matmul(state_holder, W_q, name='hidden_layer_activation') + bhid_q
    y_q = tf.nn.relu(a_y_q, name='output_of_the_hidden_layer')

    a_y_q = tf.matmul(y_q, W2_q, name='hidden_layer_2_activation') + b2_q
    y_q = tf.nn.relu(a_y_q, name='output_of_the_second_hidden_layer')

    # Q function at the current step
    a_z_q = tf.matmul(y_q, w_q, name='output_activation') + b_q
    Q  = tf.nn.softmax(a_z_q, name = 'action_probabilities')

    # defining placeholders and op's for copying parameters from Q network to target Q network
    W_update_placeholder = tf.placeholder(W_q.dtype, shape=W_q.get_shape())
    W_update_placeholder_op = W.assign(W_update_placeholder)

    w_update_placeholder = tf.placeholder(w_q.dtype, shape=w_q.get_shape())
    w_update_placeholder_op = w.assign(w_update_placeholder)

    b_update_placeholder = tf.placeholder(b_q.dtype, shape=b_q.get_shape())
    b_update_placeholder_op = b.assign(b_update_placeholder)

    bhid_update_placeholder = tf.placeholder(bhid_q.dtype, shape=bhid_q.get_shape())
    bhid_update_placeholder_op = bhid.assign(bhid_update_placeholder)

    W2_update_placeholder = tf.placeholder(W2_q.dtype, shape=W2_q.get_shape())
    W2_update_placeholder_op = W2.assign(W2_update_placeholder)

    b2_update_placeholder = tf.placeholder(b2_q.dtype, shape=b2_q.get_shape())
    b2_update_placeholder_op = b2.assign(b2_update_placeholder)

    variable_summaries(Q, '/action_probabilities')

with tf.name_scope('loss'):
    prediction = tf.reduce_sum(tf.mul(Q, actions_placeholder), 1)

    delta = target_placeholder - prediction

    error = tf.reduce_sum(tf.square(delta))
    variable_summaries(prediction, '/Q_values')
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

# train step used in the test with experience replay
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
                    Q_values = sess.run(target_Q, feed_dict={state_holder: next_states[i].reshape(1, dim_state)})
                    val = np.max(Q_values[0, :])
                    target += rewards[i] + gamma * val

                action_t = np.zeros(n_action)
                action_t[actions[i]] = 1

                actns.append(action_t)
                stats.append(states[i].reshape(dim_state))
                targets.append(target)

        return actns, stats, targets

# implementation of epsilon greedy policy
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
                        feed_dict={state_holder: state.reshape(1,dim_state)
                        })
    val = np.max(Q_values[0, :])
    max_indices = np.where(Q_values[0, :] == val)[0]
    return rd.choice(max_indices)

sess = tf.Session()  # FOR NOW everything is symbolic, this object has to be called to compute each value of Q

# SUMMARIES
merged = tf.merge_all_summaries()
suffix = time.strftime('%Y-%m-%d--%H-%M-%S')
train_writer = tf.train.SummaryWriter('tensorboard/pong_manual_preprocessing/{}'.format(suffix) + '/train', sess.graph)
test_writer = tf.train.SummaryWriter('tensorboard/pong_manual_preprocessing/{}'.format(suffix) + '/test')

# Start
sess.run(tf.initialize_all_variables())

single_episode_rewards = []
episode_rewards_list = []
episode_steps_list = []

step = 0
episode_no = 0

time_list = []
reward_list = []
err_list = []
val_list = []
err_sum_list = []
acc_t = 0

start_learning = 50000

for k in range(n_train_trials + n_test_trials):

    # Init the accumulated reward, and state of action pair of previous state
    acc_reward = 0  # Init the accumulated reward
    observation = prev_observation = env.reset()  # Init the state

    trial_err_list = []

    for t in range(trial_duration):  # The number of time steps in this game is maximum 200

        acc_t += 1
        epsilon = (epsilon_end +
              max(0., (epsilon_start - epsilon_end)
                  * (annealing_period - max(0., acc_t - start_learning)) / annealing_period))


        if acc_t % 500 == 0:
            print ('T: %d' % acc_t)
            print ('Epsilon: {%.10f}' % epsilon)


        if should_render: env.render()

        # Preprocess the observation (which is an image) before using it to select an action.
        processed_observation = prepro(observation, prev_observation)

        action = policy(processed_observation)  # Init the first action

        prev_observation = observation.copy()
        observation, reward, done, info = env.step(action_list[action])  # Take the action

        processed_new_observation = prepro(observation, prev_observation)

        replay_memory.add(state=processed_observation, action=action, reward=reward, next_state=processed_new_observation, is_terminal=done)
        error_list = []

        err = 0
        if acc_t > start_learning:

            if acc_t % 500 == 0:
                print ('STARTED TRAINING')

            if acc_t % update_frequency == 0:
                actions, states, targets = train_step(t)

                # Perform one step of gradient descent
                sess.run(training_step, feed_dict={
                state_holder: np.array(states),
                target_placeholder: np.array(targets),
                actions_placeholder: np.array(actions),
                })

                # Compute the  Error for monitoring
                err = sess.run(error, feed_dict={
                target_placeholder: np.array(targets),
                state_holder:  np.array(states),
                actions_placeholder: np.array(actions),
                })

            # copy the Q network to Target-Q network
            if acc_t % C == 0:

                sess.run(W_update_placeholder_op, {W_update_placeholder: W_q.eval(sess)})
                sess.run(w_update_placeholder_op, {w_update_placeholder: w_q.eval(sess)})
                sess.run(b_update_placeholder_op, {b_update_placeholder: b_q.eval(sess)})
                sess.run(bhid_update_placeholder_op, {bhid_update_placeholder: bhid_q.eval(sess)})
                sess.run(W2_update_placeholder_op, {W2_update_placeholder: W2_q.eval(sess)})
                sess.run(b2_update_placeholder_op, {b2_update_placeholder: b2_q.eval(sess)})


        err_list.append(err)
        single_episode_rewards.append(reward)

        if done:
            # Done with episode. Reset stuff.

            episode_no += 1

            err_sum_list.append(np.sum(err_list))
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