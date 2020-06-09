from GoEnv import *
from GoPlayGame import *
from AgentRandom import *

class GoTrain:

    def __init__(self, agent_a, agent_b, size, games_count):
        self.agent_a        = agent_a
        self.agent_b        = agent_b
        self.size           = size
        self.games_count    = games_count
  

    def run(self):
        epsilon = 0.5

        n = games_count//2

        a_wins_black = 0
        a_wins_white = 0
        b_wins_black = 0
        b_wins_white = 0

        moves = 0
       
        for game in range(n):
            result_ab, moves_ab = self._play_game(self.agent_a, self.agent_b)
            result_ba, moves_ba = self._play_game(self.agent_b, self.agent_a)

            done = round(100.0*game/n, 1)
            print("game done ", done, result_ab, result_ba)

            if result_ab > 0:
                a_wins_black+= 1
            else:
                b_wins_white+= 1

            if result_ba > 0:
                b_wins_black+= 1
            else:
                a_wins_white+= 1

            moves+= moves_ab + moves_ba
 

        print("agent A wins B/W ", a_wins_black, a_wins_white)
        print("agent B wins B/W ", b_wins_black, b_wins_white)    
        print("moves per game = ", moves/games_count)        


    def _play_game(self, agent_a, agent_b):
        env     = GoEnv(size)
        play    = GoPlayGame(env, agent_a, agent_b)

        result = 0
        moves = 0
        while result == 0:
            result = play.step()
            moves+= 1
        
        return result, moves
        

if __name__ == "__main__":
    size = 9

    state_shape     = (6, size, size)
    outputs_count   = size*size + 1

    games_count = 20

    agent_a = AgentRandom(state_shape, outputs_count)
    agent_b = AgentRandom(state_shape, outputs_count)

    train = GoTrain(agent_a, agent_b, size, games_count)
    train.run()
        

    