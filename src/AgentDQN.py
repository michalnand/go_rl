import numpy


class AgentDQN:
    def __init__(self, state_shape, outputs_count):
        self.size = state_shape[1]

    def reset(self):
        pass

    def step(self, env):
        state = env.state

        policy = numpy.random.rand(self.size*self.size + 1)

        state, reward, done, info = env.step_prob(policy)
        return state, reward, done, info
    
    def set_done(self, reward):
       pass

