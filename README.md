# HMPL_Project

## Introduction
This is a project using reinforcement learning to train an agent of Atari game Pong to compete with the game A.I. Insteal of using Q-table to this project uses (stochastic) Policy Gradients to aviod the huge amount of game states.  We explore and try to get a fast-learning or more intelligence agent. 
 
 

 # Part 1  
Adjust and bugfix the training script from Andrej Karpathy's gist. The training script bases on the [Open AI Gym environments](https://github.com/openai/gym) to run the Atari emulator and environments, and use only numpy without any external ML framework.

We tuned the parameters of model and tried to achieve a faster learning. 

### Neuron Numbers
We tried different numbers of neurons in the dense layer for 2000 episodes. 
exp | #1 | #2 | #3 | #4 |
--- | --- | --- | --- |--- 
Neurons # | 100 | 200 | 300 | 500

And we got following results. It shows that for the learning speed increase with the neuron numbers.

<img width="583" alt="image" src="https://user-images.githubusercontent.com/12693783/168729861-222e2169-b9fe-4330-9d18-e8e5b366cc0b.png">

### Batch Size and Learning Rate
We tried different combination of batch size & learning rates for 3000 episodes. 

exp | #1 | #2 | #3 | #4 |
--- | --- | --- | --- |--- 
Batch Size | 5 | 10 | 20 | 20
Neurons | 1e^-3 | 1e^-3 | 1e^-3 | 1e^-3 * 2

And we got following results. It shows that for simply adjust the batch size doesn't help with the learning. Increasing learning rate even have nagetive effects on the learning. However, when we doubled the batch size along with doubling the learning rate, we got an improved performance.

<img width="583" alt="image" src="https://user-images.githubusercontent.com/12693783/168732087-e5f1fd70-0289-426c-b5b2-c78df3fb59ba.png">


## folder structure
- pong-from-pixels

      - ：pong-from-pixels.py ----- raw python training script of the model, adjust the parameters to tune
      - : save.p ----- saved model (we have saved a model with 10 batch size, 1e-3 lr and 200 neurons, set variable 'resume' in pong-from-pixels.py to decide whether load the trained model.
      - : episode_reward_sums.txt ----- saved rewards sum during training
      - : episode_running_means.txt ----- saved running means during training


# Part 2
## folder structure

- PongGame

      - : main.py ----- main GUI file
      - : pong.kv ----- UI file 
      - : pong1.kv ------ revised UI file
      - : q_learn.py -------- Q Agent class

- HPML_project

       - : pong_t.py ------ raw python script for the policy network
       - : save.p   ---- saved model
       
- HPML-project/spinning-up-a-Pong-AI-with-deep-RL(reference: https://github.com/mtrazzi/spinning-up-a-Pong-AI-with-deep-RL)

      - : train.ipynb ------- tensroflow based code 
