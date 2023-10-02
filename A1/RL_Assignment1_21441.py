5
import numpy as np
import math

class Maze:
    def __init__(self, gridHeight=6, gridWidth=6, terminalReward=10, lockPickProb=0.5):
        self.rewardsLeft = np.array([[-1, 0, 0, 0, 0, 0], 
                                    [-1, -1, 0, 0, 0,-10], 
                                    [-1, 0, 0, -1, -1, -1], 
                                    [0, 0, 0, -10, -1, -1],
                                    [-1, -1, 0, 0, -1, 0],
                                    [-1, 0, -1, 0, 0 ,-1]])

        self.rewardsRight =  np.array([[ 0, 0, 0, 0, 0, -1], 
                            [ -1, 0, 0 , 0, -10, -1],
                            [ 0, 0, -1, -1, -1, -1],
                            [ 0, 0, -10, -1, -1 ,-1],
                            [ -1, 0, 0, -1, 0, -1],
                            [ 0, -1, 0, 0, -1, -1]])

        self.rewardsUp  =  np.array([[ -1, -1, -1, -1, -1, -1], 
                            [ 0, -1, -1, -1, -1, 0],
                            [ 0, 0, -1, 0, 0, 0],
                            [ -1, 0,0, 0,0, 0],
                            [ 0, -10, -1, -1, -1, 0],
                            [ 0,  0, -1, -10, 0, 0]])


        self.rewardsDown =  np.array([[ 0, -1, -1, -1, -1, 0], 
                            [ 0, 0, -1, 0, 0, 0],
                            [ -1, 0, 0, 0, 0, 0],
                            [ 0, -10,-1,-1,-1, 0],
                            [  0,0,-1,-10,0, 0],
                            [ -1, -1, -1, 0, -1, -1]])

        self.gridHeight = gridHeight
        self.gridWidth = gridWidth
        self.lockPickProb = lockPickProb
        self.terminalReward = terminalReward


    def isStateTerminal(self, state):
        if state == (3, 0) :
            return True
        elif state == (5, 3):
            return True
        return False

    def takeAction(self, state, action):
        retVal = []
        if(self.isStateTerminal(state)):
            return [[state,1, self.terminalReward]] 

        if action=='left':
            reward = self.rewardsLeft[state]
            if(reward == -1):
                retVal.append([state,1,-1])
            elif(reward == -10):
                retVal.append([(state[0], state[1]-1),self.lockPickProb,-1])
                retVal.append([state,1-self.lockPickProb,-1])
            else:
                retVal.append([(state[0], state[1]-1),1,-1])

        if action=='right':
            reward = self.rewardsRight[state]
            if(reward == -1):
                retVal.append([state,1,-1])
            elif(reward == -10):
                retVal.append([(state[0], state[1]+1),self.lockPickProb,-1])
                retVal.append([state,1-self.lockPickProb,-1])
            else:
                retVal.append([(state[0], state[1]+1),1,-1])

        if action=='up':
            reward = self.rewardsUp[state]
            if(reward == -1):
                retVal.append([state,1,-1])
            elif(reward == -10):
                retVal.append([(state[0]-1, state[1]),self.lockPickProb,-1])
                retVal.append([state,1-self.lockPickProb,-1])
            else:
                retVal.append([(state[0]-1, state[1]),1,-1])

        if action=='down':
            reward = self.rewardsDown[state]
            if(reward == -1):
                retVal.append([state,1,-1])
            elif(reward == -10):
                retVal.append([(state[0]+1, state[1]),self.lockPickProb,-1])
                retVal.append([state,1-self.lockPickProb,-1])
            else:
                retVal.append([(state[0]+1, state[1]),1,-1])
        for i,[nextState, prob, reward] in enumerate(retVal):
            if(self.isStateTerminal(nextState)):
                retVal[i][2] = self.terminalReward   

        return retVal 

class GridworldSolution:
    def __init__(self, maze,horizonLength):
        self.env = maze
        self.actionSpace = ['left', 'right', 'up',  'down']
        self.horizonLength = horizonLength
        self.DP = np.ones((self.env.gridHeight,self.env.gridWidth,self.horizonLength),dtype = float) * -np.inf
    
    def optimalReward(self, state, k):
        optReward = -np.inf
        
        #### Write your code here
        J = np.zeros((self.horizonLength+1,self.env.gridHeight,self.env.gridWidth),dtype = float)
        for t in range(self.horizonLength-1,-1,-1):
            for i in range(self.env.gridHeight):
                for j in range(self.env.gridWidth):
                    if (self.env.isStateTerminal((i,j))== True):
                        J[t][i][j] = self.env.terminalReward + J[t+1][i][j]
                    else:
                        up = self.env.takeAction((i,j),'up')
                        down = self.env.takeAction((i,j),'down')
                        left = self.env.takeAction((i,j),'left')
                        right = self.env.takeAction((i,j),'right')
                        if(len(up)==2):
                            r_up = up[0][1]*(up[0][2]+J[t+1][up[0][0]]) + up[1][1]*(up[1][2]+J[t+1][up[1][0]])
                        else:
                            r_up = up[0][2] + J[t+1][up[0][0]]
                        if(len(down)==2):
                            r_down = down[0][1]*(down[0][2]+J[t+1][down[0][0]]) + down[1][1]*(down[1][2]+J[t+1][down[1][0]])
                        else:
                            r_down = down[0][2] +J[t+1][down[0][0]]
                        if(len(left)==2):
                            r_left = left[0][1]*(left[0][2]+J[t+1][left[0][0]]) + left[1][1]*(left[1][2]+J[t+1][left[1][0]])
                        else:
                            r_left = left[0][2]+J[t+1][left[0][0]]
                        if(len(right)==2):
                            r_right = right[0][1]*(right[0][2]+J[t+1][right[0][0]]) + right[1][1]*(right[1][2]+J[t+1][right[1][0]])
                        else:
                            r_right = right[0][2] +J[t+1][right[0][0]]
                        J[t][i][j] = max(r_up,r_down,r_left,r_right)           

        optReward = J[k][state]


        ########
        return optReward

if __name__ == "__main__":
    maze = Maze()
    solution = GridworldSolution(maze,horizonLength=5)
    print(" Horizon ",solution.horizonLength)
    optReward = solution.optimalReward((2,0),0)
    #print(optReward)
    assert optReward==28.0, 'wrong answer'