import numpy as np
import random
import datetime
from tfutils import *
from tensorforce.environments.environment import Environment

###############################################################################################################
##  In reinforcement learning, there is an agent - who interacts with
##  a given Environment. Environment accepts current and future step and
##  returns reward value back to the agent
##
##  Env object has the following instance variables
##          -- entity2id_ a dictionary populated from entity2id.txt file
##          -- relation2id_ a dictionary populated from relation2id.txt file
##          -- relations a vector populated from the relations in relation2id.txt file
##          -- entity2vec populated from entity2vec.bern file
##          -- relation2vec populated from relation2vec.bern file
##          -- kb knowledge graph for path finding. This stores all relations from
##                  kb_env_rl.txt expect one corresponding to input task and its inverse.
##                  Relation from state to A->B as R and the relation from B->A as R_inverse
##          -- die number of times the agent chose the wrong path
##
##  Env is defined as class containing following methods
##  -- Initialize init()
##          Reads the entity2id.txt and relation2id.txt files and populates the 2 dictionaries self.entity2id_
##          and self.relation2id_variables, stores the relations from relation2id.txt in self.relation,
##          populates entity2vec and relation2vec from corresponding .bern files. Populates the kb object with
##          relations (corresponding to input task) from kb_env_rl.txt. Sets variable die to 0.
##
##  -- execute()
##          Called during learning reinforcement phases
##          state: is [current_position, target_position]
##		    action: an integer
##          return: (reward, [new_postion, target_position], done)
##
##  -- states()
##          takes as input current or next_state
##          returns the current and (target-current) positions as an array
##
##  -- actions()
##          takes the entity id as input
##          Returns set of valid actions for a given state from self.kb
##          in form of an array
##
##  -- reset()
##          Called from Runner before begining training
##
##  -- close()
##          Set environment to None

##
##
##
############################################################################################################


class DPEnv(Environment):
    def __init__(self, relationPath, graphPath, task=None):
        logger.info("Initalizing DeepPath TensorForce Env")
        self.task = task
        self.graphPath = graphPath
        self.relationPath = relationPath
        f1 = open(dataPath + 'entity2id.txt')
        f2 = open(dataPath + 'relation2id.txt')
        self.entity2id = f1.readlines()
        self.relation2id = f2.readlines()
        f1.close()
        f2.close()
        self.entity2id_ = {}
        self.relation2id_ = {}
        self.relations = []
        for line in self.entity2id:
            self.entity2id_[line.split()[0]] = int(line.split()[1])
        for line in self.relation2id:
            self.relation2id_[line.split()[0]] = int(line.split()[1])
            self.relations.append(line.split()[0])
        self.entity2vec = np.loadtxt(dataPath + 'entity2vec.bern')
        self.relation2vec = np.loadtxt(dataPath + 'relation2vec.bern')

        self.path = []
        self.path_relations = []

        self.action = dict(num_actions=action_space, type='int')

        # Knowledge Graph for path finding
        f = open(dataPath + 'kb_env_rl.txt')
        kb_all = f.readlines()
        f.close()

        self.kb = []
        for i in range(len(task)):
            if task[i] != None:
                relation = task[i].split()[2]
                for line in kb_all:
                    rel = line.split()[2]
                    if rel != relation and rel != relation + '_inv':
                        self.kb.append(line)

        logger.info("KG loaded")
        print("Knowledge Graph loaded")
        self.reset()



    def __str__(self):
        return 'DeepPath Env({}:{})'.format(self.relationPath, self.graphPath)

    def close(self):
        self.env = None


    def reset(self):
        logger.info ("In reset: %s", str(datetime.datetime.now()))
        sample = self.task[random.randint(0, len(self.task)-1)].split()
        self.localstate = [0, 0]
        self.localstate[0] = self.entity2id_[sample[0]]
        self.localstate[1] = self.entity2id_[sample[1]]
        self.state = self.localstate
        return self.state

    def execute(self, actions):
        '''
        This function process the interact from the agent
        state: is [current_position, target_position]
        action: an integer from
        return: ([new_postion, target_position], done. reward)
        '''
        state = self.state
        logger.info("In execute(): {ss}".format(ss=self.state))
        print("Agent proessing. In execute(): states - ", self.state)

        reward = 0
        done = 0  # Whether the episode has finished
        curr_pos = state[0]
        target_pos = state[1]
        chosen_relation = self.relations[actions]
        choices = []
        for line in self.kb:
            triple = line.rsplit()
            e1_idx = self.entity2id_[triple[0]]

            if curr_pos == e1_idx and triple[2] == chosen_relation and triple[1] in self.entity2id_:
                choices.append(triple)

        if len(choices) == 0:
            logger.info("Incorrect Path. End episode")
            reward = -1
            next_state = state  # stay in the initial state
            done = 1
            return (next_state, done,reward)
        else:  # find a valid step
            logger.info("Found a valid step, check action")
            path = random.choice(choices)
            self.path.append(path[2] + ' -> ' + path[1])
            self.path_relations.append(path[2])

            new_pos = self.entity2id_[path[1]]
            new_state = [new_pos, target_pos]
            self.state = new_state

            if new_pos == target_pos:
                print 'Yay, Found a path:', self.path
                logger.info("Yay, Found a path: {p}".format(p=self.path))
                done = 1
                reward = 1
                new_state = None
            return (new_state, done, reward)


    def states(self, idx_list=None):
        return dict(shape=state_dim, type='float')



    def actions(self, entityID=0):
        return dict(num_actions=action_space, type='int')

