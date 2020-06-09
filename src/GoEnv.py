import gym
import numpy

class GoEnv:

    def __init__(self, size = 9):
        self.size = size
        self.reset()

    def reset(self):
        self.env = gym.make('gym_go:go-v0', size=self.size, reward_method='real')
        self.state =  self.env.reset()

        self.state_shape    = (6, self.size, self.size)
        self.actions_count  = 1 + self.size*self.size

        self.legal_moves     = numpy.ones(self.size*self.size + 1, dtype = int)
        self.legal_moves[self.size*self.size] = 1

        return self.state
        
    def step_e_greedy(self, agent_output, epsilon):
        action = self._e_greedy_action(agent_output, epsilon)
        self.state, reward, done, info = self.env.step(action)

        self._update_legal_moves(self.state[3])

        return self.state, reward, done, info

    def step_prob(self, agent_output):
        action = self._prob_action(agent_output)
        self.state, reward, done, info = self.env.step(action)

        self._update_legal_moves(self.state[3])

        return self.state, reward, done, info

    def render(self):
        self.env.render('terminal')

    def _e_greedy_action(self, agent_output, epsilon):
        probs = self._get_actions_probs(agent_output)
 
        if numpy.random.rand() < epsilon:
            actions_allowed = numpy.nonzero(probs)[0]
            i = numpy.random.randint(len(actions_allowed))
            action = actions_allowed[i]
        else:
            action = numpy.argmax(probs)


        return action

    def _prob_action(self, agent_output):
        probs = self._get_actions_probs(agent_output)

        action = numpy.random.choice(self.actions_count, 1, p=probs)[0]

        return action

    def _get_actions_probs(self, x):
        probs = x - numpy.max(x)
        probs = numpy.exp(probs)
        probs = probs*self.legal_moves
        probs = probs/(numpy.sum(probs))

        return probs

    def _update_legal_moves(self, mask):
        mask = 1 - mask.flatten().astype(int)
        self.legal_moves[0:self.size*self.size] = mask



if __name__ == "__main__":
    size = 19
    env = GoEnv(size)

    while True:
        black = numpy.random.rand(size*size + 1)
        state, reward, done, info = env.step_e_greedy(black, 0.5)
        env.render()
        if done:
            break

        white = numpy.random.rand(size*size + 1)
        state, reward, done, info = env.step_e_greedy(white, 0.5)
        env.render()
        if done:
            break

