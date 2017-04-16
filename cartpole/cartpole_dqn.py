import gym
import time
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from gym.envs.classic_control.cartpole import CartPoleEnv
from tf_tools import *

from cartpole_utils import plot_results,print_results
import tensorflow as tf
from replay_memory import ReplayMemory

mini_batch_size = 200
replay_memory_size = 1000
replay_memory = ReplayMemory(replay_memory_size)

# Algorithm parameters
learning_rate = 0.001
gamma = .99
epsilon = 0.05

# General parameters
render = False
N_print_every = 10
N_trial = 1000
N_trial_test = 10
trial_duration = 200
write_summary = 1

n_hidden = 50

# Generate the environment
env = CartPoleEnv()
dim_state = env.observation_space.high.__len__()
n_action = env.action_space.n

double_q_learning = False

# Generate the symbolic variables to hold the state values
state_holder = tf.placeholder(dtype=tf.float32, shape=(None, dim_state), name='symbolic_state')
target_placeholder = tf.placeholder(tf.float32, shape=(None,))
actions_placeholder = tf.placeholder(tf.float32, shape=(None, n_action))

# Initialize the parameters of the Target-Q model
W0 = np.array(rd.randn(dim_state, n_hidden) / np.sqrt(dim_state),dtype=np.float32)
w0 = np.array(rd.randn(n_hidden, n_action) / np.sqrt(n_hidden),dtype=np.float32)
bhid0 = np.zeros(n_hidden, dtype=np.float32)
b0 = np.zeros(n_action, dtype=np.float32)

# Create the parameters of the Target-Q model
W = tf.Variable(initial_value=W0.reshape(dim_state, n_hidden), trainable=True, name='weight_variable_matrix')
w = tf.Variable(initial_value=w0.reshape(n_hidden, n_action), trainable=True, name='weight_variable')
b = tf.Variable(initial_value=b0, trainable=True, name='bias')
bhid = tf.Variable(initial_value=bhid0, trainable=True, name='bias')

a_y = tf.matmul(state_holder, W, name='hidden_layer_activation') + bhid
y = tf.nn.relu(a_y, name='output_of_the_hidden_layer')

# Target-Q function at the current step
a_z = tf.matmul(y, w, name='output_activation') + b
target_Q = -tf.nn.sigmoid(a_z)

# Create the parameters of the Q model
W_q = tf.Variable(initial_value=W0.reshape(dim_state, n_hidden), trainable=True, name='weight_variable_matrix')
w_q = tf.Variable(initial_value=w0.reshape(n_hidden, n_action), trainable=True, name='weight_variable')
b_q = tf.Variable(initial_value=b0, trainable=True, name='bias')
bhid_q = tf.Variable(initial_value=bhid0, trainable=True, name='bias')

a_y_q = tf.matmul(state_holder, W_q, name='hidden_layer_activation') + bhid_q
y_q = tf.nn.relu(a_y_q, name='output_of_the_hidden_layer')

# Q function at the current step
a_z_q = tf.matmul(y_q, w_q, name='output_activation') + b_q
Q = -tf.nn.sigmoid(a_z_q)

# Define the operation that performs the optimization
q_values = tf.reduce_sum(tf.mul(Q, actions_placeholder), 1)
error = tf.reduce_mean(tf.square(target_placeholder - q_values))
training_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(error)

# used in double-Q learning
target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
target_q_with_idx = tf.gather_nd(target_Q, target_q_idx)

# placeholders and op's used when copying Q-network to Target-Q network
W_update_placeholder = tf.placeholder(W_q.dtype, shape=W_q.get_shape())
W_update_placeholder_op = W.assign(W_update_placeholder)

w_update_placeholder = tf.placeholder(w_q.dtype, shape=w_q.get_shape())
w_update_placeholder_op = w.assign(w_update_placeholder)

b_update_placeholder = tf.placeholder(b_q.dtype, shape=b_q.get_shape())
b_update_placeholder_op = b.assign(b_update_placeholder)

bhid_update_placeholder = tf.placeholder(bhid_q.dtype, shape=bhid_q.get_shape())
bhid_update_placeholder_op = bhid.assign(bhid_update_placeholder)


variable_summaries(Q, '/Q_values')
acc_episode_reward_placeholder = tf.placeholder(dtype=tf.float32)
variable_summaries(acc_episode_reward_placeholder, '/episode_reward')

err_placeholder = tf.placeholder(dtype=tf.float32)
variable_summaries(err_placeholder, '/error')

sess = tf.Session()  # FOR NOW everything is symbolic, this object has to be called to compute each value of Q

merged = tf.merge_all_summaries()
suffix = time.strftime('%Y-%m-%d--%H-%M-%S')
train_writer = tf.train.SummaryWriter('tensorboard/cart_pole/{}'.format(suffix) + '/train', sess.graph)
test_writer = tf.train.SummaryWriter('tensorboard/cart_pole/{}'.format(suffix) + '/test')

sess.run(tf.initialize_all_variables())

# train step -> sample, and generate targets, states and actions
def train_step():
    targets = []
    stats = []
    actns = []
    err = 0.0

    if replay_memory.length() >= mini_batch_size:
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

                # Perform one step of gradient descent
        sess.run(training_step, feed_dict={
                    state_holder: np.array(stats),
                    target_placeholder: np.array(targets),
                    actions_placeholder: np.array(actns),
                })

        # Compute the Bellman Error for monitoring
        err = sess.run(error, feed_dict={
                    target_placeholder: np.array(targets),
                    state_holder: np.array(stats),
                    actions_placeholder: np.array(actns),
                })



    return  actns, stats, targets, err
'''

def train_step(state, action,reward,next_state,done):
    targets = []
    stats = []
    actns = []
    err = 0.0

    target = 0
    if done:
            target += reward
    else:
        Q_values = sess.run(target_Q, feed_dict={state_holder: next_state.reshape(1, dim_state)})
        val = np.max(Q_values[0, :])
        target += reward + gamma * val

    action_t = np.zeros(n_action)
    action_t[action] = 1

    actns.append(action_t)
    stats.append(state.reshape(dim_state))
    targets.append(target)

                # Perform one step of gradient descent
    sess.run(training_step, feed_dict={
                    state_holder: np.array(stats),
                    target_placeholder: np.array(targets),
                    actions_placeholder: np.array(actns),
                })

        # Compute the Bellman Error for monitoring
    err = sess.run(error, feed_dict={
                    target_placeholder: np.array(targets),
                    state_holder: np.array(stats),
                    actions_placeholder: np.array(actns),
                })



    return  actns, stats, targets, err
'''

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


time_list = []
reward_list = []
err_list = []
val_list = []

acc_t = 0

start_learning = 1
epsilon_start = .1
epsilon_end = .1
episode_no = 0

C = 100
annealing_period = 1

for k in range(N_trial + N_trial_test):

    # Init the accumulated reward, and state of action pair of previous state
    acc_reward = 0  # Init the accumulated reward
    observation = env.reset()  # Init the state

    trial_err_list = []

    for t in range(trial_duration):  # The number of time steps in this game is maximum 200
        if render: env.render()

        acc_t += 1

        epsilon = (epsilon_end +
              max(0., (epsilon_start - epsilon_end)
                  * (annealing_period - max(0., acc_t - start_learning)) / annealing_period))

        #if acc_t % 500 == 0:
            #print ('T: %d' % acc_t)
            #print ('Epsilon: {%.10f}' % epsilon)

        action = policy(observation)  # Init the first action

        new_observation, reward, done, info = env.step(action)  # Take the action

        reward = 0
        if done and t < 199: reward = -1  # The reward is modified

        target = 0
        error_list = []

        replay_memory.add(state=observation, action=action, reward=reward, next_state=new_observation, is_terminal=done)

        err = 0
        if acc_t > start_learning:
            #if acc_t % 500 == 0:
            #$print ('STARTED TRAINING')

            #actions,states,targets, err = train_step(observation, action,reward,new_observation,done)
            actions, states, targets, err = train_step()


        if acc_t % C == 0:
            sess.run(W_update_placeholder_op, {W_update_placeholder: W_q.eval(sess)})
            sess.run(w_update_placeholder_op, {w_update_placeholder: w_q.eval(sess)})
            sess.run(b_update_placeholder_op, {b_update_placeholder: b_q.eval(sess)})
            sess.run(bhid_update_placeholder_op, {bhid_update_placeholder: bhid_q.eval(sess)})


        observation = new_observation  # Pass the new state to the next step
        acc_reward += reward  # Accumulate the reward

        # Add the error in a trial-specific list of errors
        trial_err_list.append(err)

        if done:
            break  # Stop the trial when the environment says it is done


    summary = sess.run(merged, feed_dict={
                state_holder: observation.reshape(1, dim_state),
                err_placeholder: np.mean(trial_err_list),
                acc_episode_reward_placeholder: acc_reward
            })

    train_writer.add_summary(summary, k)

    # Stack values for monitoring
    err_list.append(np.mean(trial_err_list))
    time_list.append(t + 1)
    reward_list.append(acc_reward)  # Store the result

    if len(reward_list) >= 10 and np.mean(reward_list[-10:])==0:
        break

    print_results(k, time_list, err_list, reward_list,N_print_every)

plot_results(N_trial,N_trial_test,reward_list,time_list, err_list)
plt.show()
