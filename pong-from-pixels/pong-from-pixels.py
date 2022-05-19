""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle as pickle
import atari_py
import gym

from gym import wrappers

# hyperparameters to tune
H = 200 # number of hidden layer neurons
batch_size = 10 
learning_rate = 1e-3 

resume = False # resume training from previous checkpoint from save.p file
render = True # set true to render the video output

# model initialization
D = 75 * 80 
if resume:
  filename = 'save_H' + str(H) + '_bs' + str(batch_size) + 'lr_' + str(learning_rate) + '.p'
  model = pickle.load(open(filename, 'rb'))
  print('resume from ' + filename + '.p')
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization - Shape will be H x D
  model['W2'] = np.random.randn(H) / np.sqrt(H) # Shape will be H
  model['episodes'] = 0 # To save current epidodes saved in model

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
  """ remove redundant pixels, downsample by factor of 2, erase background, """
  I = I[35:185] 
  I = I[::2,::2,0]
  I[I == 144] = 0
  I[I == 109] = 0
  I[I != 0] = 1 
  return I.astype(float).ravel() # ravel flattens an array and collapses it into a column vector

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)): 
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * 0.99 + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  """forward prop"""
  h = np.dot(model['W1'], x) 
  h[h<0] = 0 # ReLU introduces nonlinearity
  logp = np.dot(model['W2'], h) 
  p = sigmoid(logp)  # sigmoid output to  between 0 & 1 range
  return p, h # return probability of taking action 2 (UP), and hidden state

def policy_backward(eph, epx, epdlogp):
  """ backward pass."""
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # eph is array of intermediate hidden states
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = model['episodes']

episode_counts = []
episode_results = []
game_results = []
episode_reward_sums = []
episode_running_means = []
batch_counts = []

while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  # take the difference between two frames as input
  x = cur_x - prev_x if prev_x is not None else np.zeros(D) # Take
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  # roll the dice to sample the action
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice! 

  # record various intermediates (needed later for backprop).
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"

  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
  # if take action UP, then the y - aprob would be nagetive, 

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward
  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    game_results = []
    print(episode_counts)
    print(episode_results)
    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (Policy Grad magic happens right here.)
    grad = policy_backward(eph, epx, epdlogp)
    for k in model: 
      if k != 'episodes':
        grad_buffer[k] += grad[k] # accumulate grad over batch
    # perform rmsprop parameter update every batch_size episodes
    
    if episode_number % batch_size == 0:
      for k,v in model.items():
        if k != 'batches':
          g = grad_buffer[k] # gradient
          rmsprop_cache[k] = 0.99 * rmsprop_cache[k] + (1 - 0.99) * g**2
          model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
          grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode reward total was ' + str(reward_sum) + 'running mean: ' + str(running_reward))

    model['episodes'] = episode_number

    textfile = open("episode_counts.txt", "a")
    textfile.write(str(episode_number) + ",")
    textfile.close()

    textfile = open("episode_reward_sums.txt", "a")
    textfile.write(str(reward_sum) + ",")
    textfile.close()

    textfile = open("episode_running_means.txt", "a")
    textfile.write(str(running_reward) + ",")
    textfile.close()

    print('H:' + str(H))
    print('batch_size:' + str(batch_size))

    print('saving')
    pickle.dump(model, open('save.p', 'wb'))

    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0: 
    # not 0 means we get a game reward
    game_results.append(reward)
    print ('ep' + str(episode_number) +': game finished, reward: ' + str(reward) + ('' if reward == -1 else ' !!!!!!!!'))
