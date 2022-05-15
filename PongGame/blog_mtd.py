import numpy as np
import pickle

# # hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# # model initialization
D = 160 * 120 # input dimensionality: 80x80 grid

if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory


class agent():
    def __init__(self,resume,name):
        self.name = name
        self.train_complete = False
        # hyperparameters
        self.H = 200 # number of hidden layer neurons
        self.batch_size = 10 # every how many episodes to do a param update?
        self.learning_rate = 1e-4
        self.gamma = 0.99 # discount factor for reward
        self.decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
        self.resume = resume # resume from previous checkpoint?
        self.render = False

        # model initialization
        self.D = 160 * 120 # input dimensionality: 80x80 grid

        if self.resume:
            self.model = pickle.load(open('save_%s.p'%(self.name), 'rb'))
        else:
            self.model = {}
            self. model['W1'] = np.random.randn(self.H,self.D) / np.sqrt(self.D) # "Xavier" initialization
            self.model['W2'] = np.random.randn(self.H) / np.sqrt(self.H)

        self.grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory
        self.prev = None
        self.epx,self.eph,self.epdlogps,self.epr = None,None,None,None
        self.xs,self.hs,self.dlogps,self.drs = [],[],[],[]
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0
        
    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

    def discount_rewards(self,r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_forward(self,x):
        h = np.dot(self.model['W1'], x)
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = self.sigmoid(logp)
        return p, h # return probability of taking action 2, and hidden state

    def policy_backward(self,eph,epx, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}

    def getObvs(self,coor):
        #coor[p1,p2,ball]
        p1 = coor[0]
        p2 = coor[1]
        ball = coor[2]
        mtx = np.array([[0 for _ in range(160)] for _ in range(120)])
        # print(mtx[3:5,-1])
        mtx[p1:p1+20,0] = 1
        mtx[p2:p2+20,-1] = 1
        mtx[ball[1],ball[0]]= 1
        return mtx.astype(np.float).ravel()
    def train(self,observe,reward,done,info):
        
        # preprocess the input image
        cur_x = self.getObvs(observe)
        x = cur_x - self.prev if self.prev is not None else np.zeros(self.D)
        self.prev = cur_x

        aprob, h = self.policy_forward(x)
        action = 2 if np.random.uniform() < aprob else -2 # roll the dice!
        
        self.xs.append(x) #observation
        self.hs.append(h) # hidden state
        y = 1 if action == 2 else 0
        self.dlogps.append(y-aprob) # grad that encourages the action that was taken to be taken

        self.reward_sum +=reward
        self.drs.append(reward)

        if reward != 0:
            print(('ep %d: game finished, reward: %f' % (self.episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
        
        if done:
            self.episode_number+=1

            self.epx = np.vstack(self.xs)
            self.eph = np.vstack(self.hs)
            self.epdlogp = np.vstack(self.dlogps)
            self.epr = np.vstack(self.drs)
            self.xs,self.hs,self.dlogps,self.drs = [],[],[],[] # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = self.discount_rewards(self.epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            self.epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
            grad = self.policy_backward(self.eph,self.epx,self.epdlogp)
            
            # for k in self.model:
            #     print(k, ": ",len(self.model[k]))
            #     print(len(grad[k][0]))
            #     # self.grad_buffer[k] += grad[k]
            for k in self.model:
                self.grad_buffer[k] += grad[k]
            # perform rmsprop parameter update every batch_size episodes
            if self.episode_number % batch_size == 0:
                for k,v in self.model.items():
                    g = self.grad_buffer[k] # gradient
                    self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                    self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                    self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            
            self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, self.running_reward) )
            if self.episode_number % 100 == 0: pickle.dump(self.model, open('save_%s.p'%(self.name), 'wb'))
            self.reward_sum = 0
            self.prev_x = None
            # self.train_complete = True
        # else:
        return action*20 if 0 < action*20 + observe[1] <= 1000 else 0