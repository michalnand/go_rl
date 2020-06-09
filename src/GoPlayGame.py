from GoEnv import *
from AgentRandom import *

class GoPlayGame:

    def __init__(self, env, agent_a, agent_b):
        self.env        = env
        self.agent_a    = agent_a
        self.agent_b    = agent_b

        self.reset()

    def reset(self):
        self.state = self.env.reset()
        self.agent_a.reset()
        self.agent_b.reset()

    def step(self):
        state, reward, done, info = self.agent_a.step(self.env)
        
        if done:
            self.agent_a.set_done(reward)
            self.agent_b.set_done(-reward)
            return 1

        state, reward, done, info = self.agent_a.step(self.env)
 
        if done:
            self.agent_b.set_done(reward)
            self.agent_a.set_done(-reward)
            return -1

        return 0


if __name__ == "__main__":
    size = 9
    env = GoEnv(size)
    agent_a = AgentRandom(env.state_shape, env.actions_count)
    agent_b = AgentRandom(env.state_shape, env.actions_count)

    play = GoPlayGame(env, agent_a, agent_b)
    
    result = 0
    while result == 0:
        result = play.step()
    