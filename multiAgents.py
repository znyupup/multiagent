# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
from util import Stack

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print newScaredTimes

        "*** YOUR CODE HERE ***"
        #calculate foodScore 
        foodDistance = float('inf')
        for food in newFood.asList():
            #print food
            dist = manhattanDistance(food, newPos)
            #print "food distance is" , dist
            #print "the number of food is " ,len(newFood.asList())
            if(dist < foodDistance):
                foodDistance = dist          
        foodScore = 2.0/(foodDistance+100) + 3.0/(len(newFood.asList())+1)
        
        #calculate ghostScore
        ghostScore = 0
        for ghostState in newGhostStates: #multi ghosts
            dist = manhattanDistance(newPos, ghostState.getPosition())
            #print "ghost dist is" , dist
            if ghostState.scaredTimer>0 and dist==0 :
                ghostScore += 100.0
            if ghostState.scaredTimer>0 and dist < ghostState.scaredTimer:
                ghostScore += (1 / (1 + dist))
            elif dist < 3:
                ghostScore -= dist / 100;
        
        score = ghostScore + foodScore
                
        return score;

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        agentNum = 0        
        results = []
        v = float('-inf')
        #first call from the min function
        actions = gameState.getLegalActions(0)
        for action in actions:
            currentState = gameState.generateSuccessor(agentNum, action)
            min = self.minValue(currentState, agentNum+1, depth)
            results.append((min, action))
            v = getMax(v, min)
       
        for result in results:
            if v == result[0] :
                nextAction = result[1]  
                                                          
        return nextAction
        
        
    def maxValue(self, state, agentNum, depth):
        #depth = depth + 1
        #print "max" , depth        
        if depth >= self.depth:
            return self.evaluationFunction(state)
        
        currentActions = state.getLegalActions(agentNum)       
        if len(currentActions) > 0:
            v = float('-inf')
        else:
            v = self.evaluationFunction(state)        
        for action in currentActions:
            currentState = state.generateSuccessor(agentNum, action)
            #agentNum = agentNum +1
            min = self.minValue(currentState, agentNum+1, depth)
            v = getMax(v, min)
        return v
            
               
    def minValue(self, state, agentNum, depth):
        #print "min" , depth        
        if depth >= self.depth:
            return self.evaluationFunction(state)
        
        currentActions = state.getLegalActions(agentNum)
        if len(currentActions) > 0:
            v = float('inf')
        else:
            v = self.evaluationFunction(state)
        for action in currentActions:
            if agentNum >= (state.getNumAgents()-1):
                currentState = state.generateSuccessor(agentNum, action)
                #depth = depth + 1
                max = self.maxValue(currentState, 0, depth+1)
                v = getMin(v, max)
            else:
                currentState = state.generateSuccessor(agentNum, action)
                #agentNum = agentNum + 1
                max = self.minValue(currentState, agentNum+1, depth)
                v = getMin(v, max)
            
        return v
        
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        agentNum = 0        
        results = []
        a = float('-inf')
        b = float('inf')
        v = float('-inf')
        
        #first call from the min function
        actions = gameState.getLegalActions(0)
        for action in actions:
            currentState = gameState.generateSuccessor(agentNum, action)
            min = self.minValue(currentState, agentNum+1, depth, a, b)
            results.append((min, action))
            v = getMax(v, min)
            if v >= b:
                return v
            a = getMax(a, v)
        
        #find the minmax action      
        for result in results:
            if v == result[0] :
                nextAction = result[1]  
                                                          
        return nextAction
    
    def maxValue(self, state, agentNum, depth, a, b):
        #print "max" , depth        
        if depth >= self.depth:
            return self.evaluationFunction(state)
        
        currentActions = state.getLegalActions(agentNum)       
        if len(currentActions) > 0:
            v = float('-inf')
        else:
            v = self.evaluationFunction(state)  
                  
        for action in currentActions:
            currentState = state.generateSuccessor(agentNum, action)
            min = self.minValue(currentState, agentNum+1, depth, a, b)
            v = getMax(v, min)
            if v > b:
                return v
            a = getMax(a, v)
            
        return v
            
               
    def minValue(self, state, agentNum, depth, a, b):
        #print "min" , depth        
        if depth >= self.depth:
            return self.evaluationFunction(state)
        
        currentActions = state.getLegalActions(agentNum)
        if len(currentActions) > 0:
            v = float('inf')
        else:
            v = self.evaluationFunction(state)
        for action in currentActions:
            if agentNum >= (state.getNumAgents()-1):
                currentState = state.generateSuccessor(agentNum, action)
                max = self.maxValue(currentState, 0, depth+1, a, b)
                v = getMin(v, max)
                if v < a:
                    return v
                b = getMin(b, v)
            else:
                currentState = state.generateSuccessor(agentNum, action)
                max = self.minValue(currentState, agentNum+1, depth, a, b)
                v = getMin(v, max) 
                if v < a:
                    return v 
                b = getMin(b, v)        
        return v

def getMax(x, y):
    if x > y :
        return x
    return y

def getMin(x, y):
    if x < y :
        return x
    return y

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        agentNum = 0        
        results = []
        v = float('-inf')
        #first call from the min function
        actions = gameState.getLegalActions(0)
        for action in actions:
            currentState = gameState.generateSuccessor(agentNum, action)
            mean = self.meanValue(currentState, agentNum+1, depth)
            results.append((mean, action))
            v = getMax(v, mean)
       
        for result in results:
            if v == result[0] :
                nextAction = result[1]  
                                                          
        return nextAction
    
    def maxValue(self, state, agentNum, depth):
        #depth = depth + 1
        #print "max" , depth        
        if depth >= self.depth:
            return self.evaluationFunction(state)
        
        currentActions = state.getLegalActions(agentNum)       
        if len(currentActions) > 0:
            v = float('-inf')
        else:
            v = self.evaluationFunction(state)        
        for action in currentActions:
            currentState = state.generateSuccessor(agentNum, action)
            #agentNum = agentNum +1
            mean = self.meanValue(currentState, agentNum+1, depth)
            v = getMax(v, mean)
        return v
            
               
    def meanValue(self, state, agentNum, depth):
        #print "min" , depth        
        if depth >= self.depth:
            return self.evaluationFunction(state)
        
        currentActions = state.getLegalActions(agentNum)
        if len(currentActions) > 0:
            v = float('inf')
        
            mean = 0.0
            for action in currentActions:
                if agentNum >= (state.getNumAgents()-1):
                    currentState = state.generateSuccessor(agentNum, action)
                    #depth = depth + 1
                    mean = mean + self.maxValue(currentState, 0, depth+1)
                    #v = getMin(v, max)
                else:
                    currentState = state.generateSuccessor(agentNum, action)
                    #agentNum = agentNum + 1
                    mean = mean + self.meanValue(currentState, agentNum+1, depth)
                    #v = getMin(v, max)
                    
            v = mean/(len(currentActions))
            
        else:
            v = self.evaluationFunction(state)
            
        return v

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Evaluate the score with the nearest food distance, the number of remaining food, 
                   and the ghost distance.
                   Then combine the score with the default score function.
    """
    "*** YOUR CODE HERE ***"
    pacPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    foodDistance = float('inf')
    #calculate food score
    for food in foods.asList():
        #print food
        dist = manhattanDistance(food, pacPosition)
        if(dist < foodDistance):
            foodDistance = dist          
    foodScore = 1.0/(foodDistance+100) + 3.0/(len(foods.asList())+1)
    
    #calculate ghostScore
    ghostScore = 0
    for ghostState in ghostStates: #multi ghosts
        dist = manhattanDistance(pacPosition, ghostState.getPosition())
        #print "ghost dist is" , dist
        if ghostState.scaredTimer>0 and dist==0 :
            ghostScore += 100.0
        if ghostState.scaredTimer>0 and dist < ghostState.scaredTimer:
            ghostScore += (1 / (1 + dist))
        elif dist < 3:
            ghostScore -= dist / 100;
    
    score = ghostScore + foodScore + currentGameState.getScore()
            
    return score;

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        self.depth = 3
        depth = 0
        agentNum = 0        
        results = []
        a = float('-inf')
        b = float('inf')
        v = float('-inf')
        
        actions = gameState.getLegalActions(0)
        
        #scores = [self.evaluationFunction(gameState, action) for action in actions]
        #bestScore = max(scores)
        #bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        #chosenIndex = random.choice(bestIndices)
        
        #first call from the min function
        
        for action in actions:
            currentState = gameState.generateSuccessor(agentNum, action)
            min = self.meanValue(currentState, agentNum+1, depth)
            score = self.newEvaluationFunction(gameState, action)
            #print "min is ", min,  "score is " , score
            newScore = min + score
            results.append((newScore, action))
            v = getMax(v, newScore)
            if v >= b:
                return v
            a = getMax(a, v)
        
        #find the minmax action      
        for result in results:
            if v == result[0] :
                nextAction = result[1]  
                                                          
        #return nextAction
        util.raiseNotDefined()
    
    def maxValue(self, state, agentNum, depth):
        #depth = depth + 1
        #print "max" , depth        
        if depth >= self.depth:
            return self.evaluationFunction(state)
        
        currentActions = state.getLegalActions(agentNum)       
        if len(currentActions) > 0:
            v = float('-inf')
        else:
            v = self.evaluationFunction(state)        
        for action in currentActions:
            currentState = state.generateSuccessor(agentNum, action)
            #agentNum = agentNum +1
            mean = self.meanValue(currentState, agentNum+1, depth)
            v = getMax(v, mean)
        return v
            
               
    def meanValue(self, state, agentNum, depth):
        #print "min" , depth        
        if depth >= self.depth:
            return self.evaluationFunction(state)
        
        currentActions = state.getLegalActions(agentNum)
        if len(currentActions) > 0:
            v = float('inf')
        
            mean = 0.0
            for action in currentActions:
                if agentNum >= (state.getNumAgents()-1):
                    currentState = state.generateSuccessor(agentNum, action)
                    #depth = depth + 1
                    mean = mean + self.maxValue(currentState, 0, depth+1)
                    #v = getMin(v, max)
                else:
                    currentState = state.generateSuccessor(agentNum, action)
                    #agentNum = agentNum + 1
                    mean = mean + self.meanValue(currentState, agentNum+1, depth)
                    #v = getMin(v, max)
                    
            v = mean/(len(currentActions))
            
        else:
            v = self.evaluationFunction(state)
            
        return v
    
    def newEvaluationFunction(self, currentGameState, action):

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        #print newScaredTimes

        "*** YOUR CODE HERE ***"
        #calculate foodScore 
        foodDistance = float('inf')
        for food in newFood.asList():
            #print food
            dist = manhattanDistance(food, newPos)
            #print "food distance is" , dist
            #print "the number of food is " ,len(newFood.asList())
            if(dist < foodDistance):
                foodDistance = dist          
        foodScore = 2.0/(foodDistance+100) + 3.0/(len(newFood.asList())+1)
        
        #calculate ghostScore
        ghostScore = 0
        for ghostState in newGhostStates: #multi ghosts
            dist = manhattanDistance(newPos, ghostState.getPosition())
            #print "ghost dist is" , dist
            if ghostState.scaredTimer>0 and dist==0 :
                ghostScore += 100.0
            if ghostState.scaredTimer>0 and dist < ghostState.scaredTimer:
                ghostScore += (300 / (1 + dist))
            elif dist < 3:
                ghostScore -= dist / 100;
        
        score = ghostScore + foodScore 
        #+ successorGameState.getScore()
                
        return score;

