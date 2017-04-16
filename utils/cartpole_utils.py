import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def plot_results(N_trial,N_trial_test,reward_list,time_list, err_list):

    print('Testing over {} trials:'.format(N_trial_test))
    print('Average trial duration: {}'.format(np.mean(time_list[N_trial:])))
    print('Average acc. reward {}'.format(np.mean(reward_list[N_trial:])))
    fig, ax_list = plt.subplots(3, 2)
    ax_list[0,0].set_title('Training')
    ax_list[0,0].plot(reward_list[:N_trial], lw=2, color='green')
    ax_list[0,0].set_ylabel('Acc. Reward')
    ax_list[1,0].plot(time_list[:N_trial], lw=2, color='blue')
    ax_list[1,0].set_ylabel('Trial duration')
    ax_list[2,0].plot(err_list[:N_trial], lw=2, color='red')
    ax_list[2,0].set_ylabel('Bellman error')
    ax_list[2,0].set_xlabel('Trial number')
    ax_list[2,0].set_yscale('log')

    ax_list[0,1].set_title('Testing')
    ax_list[0,1].plot(reward_list[N_trial:], lw=2, color='green')
    ax_list[0,1].set_ylabel('Acc. Reward')
    ax_list[0,1].axhline(np.mean(reward_list[N_trial:]),color='black',linestyle='dashed',lw=4)
    ax_list[0,1].set_ylim(ax_list[0,0].get_ylim())

    ax_list[1,1].plot(time_list[N_trial:], lw=2, color='blue')
    ax_list[1,1].set_ylabel('Trial duration')
    ax_list[1,1].axhline(np.mean(time_list[N_trial:]),color='black',linestyle='dashed',lw=4)
    ax_list[1,1].set_ylim(ax_list[1,0].get_ylim())

    ax_list[2,1].plot(err_list[N_trial:], lw=2, color='red')
    ax_list[2,1].set_ylabel('Error')
    ax_list[2,1].axhline(np.mean(err_list[N_trial:]),color='black',linestyle='dashed',lw=4)
    ax_list[2,1].set_ylim(ax_list[2,0].get_ylim())
    ax_list[2,1].set_yscale('log')

    ax_list[2,1].set_xlabel('Trial number')

def print_results(k,time_list,err_list,reward_list,N_print_every=100):
    if np.mod(k,N_print_every) == 0:

        print(
            'Trial number: {} \t Duration: {:.3g} +- {:.3g} \t Error {:.3g} +- {:.3g} \t Acc reward: {:.3g} +- {:.3g}'.format(
                k,
                np.mean(time_list[-N_print_every:]),
                np.std(time_list[-N_print_every:]),
                np.mean(err_list[-N_print_every:]),
                np.std(err_list[-N_print_every:]),
                np.mean(reward_list[-N_print_every:]),
                np.std(reward_list[-N_print_every:]),
            ))