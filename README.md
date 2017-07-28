# Deep-Q Learning algorithm implementation using TensorFlow library

This project represents the implementation of the popular reinforcement learning algorithm called Deep-Q Learning introduced by Google Deep Mind. The reference to the paper is given on the following link:[Human-Level Control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
 
The implementation consists of:

1. Deep Q-Learning 
2. Using experience replay memory 
3. Using two separate deep convolutional neural networks: Q and Target-Q network

## Requirements


- Python 2.7 or Python 3.3+
- [gym](https://github.com/openai/gym)
- [OpenCV2](http://opencv.org/)
- [TensorFlow 0.11.0](https://github.com/tensorflow/tensorflow/tree/r0.11)


## Usage

To train and test a model for CartPole:

     $ python cartpole_dqn.py

To train and test a model for Pong:

    $ python pong_dqn_conv.py

To train and test a model for Breakout:

    $ python breakout_dqn_conv.py


## Detailed results

** [1] Accumulated reward, Q values and average error values obtained for CartPole environment: **

![acc_reward_cartpole](assets/acc_reward_cartpole)
![Q_cartpole](assets/Q_cartpole)
![error_cartpole](assets/error_cartpole)

** [2] Accumulated reward, Q values and average error values obtained for Pong environment: **

![acc_reward_pong](assets/acc_reward_pong)
![Q_pong_cnn](assets/Q_pong_cnn)
![error_pong](assets/error_pong)


** [3] Accumulated reward, Q values and average error values obtained for Breakout environment: **

![acc_reward_breakout](assets/acc_reward_breakout)
![Q_breakout](assets/Q_breakout)
![error_breakout](assets/error_breakout)

## License

MIT licence.
