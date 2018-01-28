
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from tfutils import *
import os

from tensorforce.agents import VPGAgent
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from Tforcedp import DPEnv


###############################################################################################################
##  This is the main function of the tensorforce
##  implementation of the Deep Path program
##
##  The program takes following argument as parameters
##
##  -r  or --relation = relation name
##  -e  or --episodes = Number of episodes
##                      default : 500
##  -a  or --agent    = Agent Name
##                      default : vpg
##                      allowed values : vpg or dqn (lowercase)
##  -D  or --debug    = Show Debug Logs
##                      default : False
##
##
##
##
##
##

############################################################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--relation', help="Number of episodes")
    parser.add_argument('-e', '--episodes', type=int, default=500, help="Number of episodes")
    parser.add_argument('-a', '--agent', type=str, default='vpg', help="VPG or DQN Agent")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    args = parser.parse_args()
    print("Running DeepPath-TensorForce")


    if args.relation:  # relation is defined
        relation = args.relation
        logger.info('Relation set to %s', relation)
    else:
        logger.error("Error : No Relation name provided!")
        return

    graphPath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
    relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'
    if not os.path.exists(relationPath):
        logger.info('Incorrect relation specified  %s', relation)
        print('Incorrect relation specified ', relation)
    f = open(relationPath)
    data = f.readlines()
    f.close()


    # Initialize the DeePath Environment class
    environment = DPEnv(graphPath, relationPath, task=data)

    network_spec = [
        dict(type='dense', size=512, activation='relu'),
        dict(type='dense', size=1024, activation='relu')
    ]

    step_optimizer = dict(type='adam', learning_rate=1e-3)
    agent = None

    if args.agent == 'vpg':
        logger.info('Initializing VPGAgent')
        agent = VPGAgent(states_spec=dict(shape=state_dim, type='float'),
                         actions_spec=dict(num_actions=action_space, type='int'),
                         network_spec=network_spec, optimizer=step_optimizer,
                         discount=0.99, batch_size=1000)
    elif args.agent == 'dqn':
        logger.info('Initializing DQNAgent')
        agent = DQNAgent(states_spec=dict(shape=state_dim, type='float'),
                         actions_spec=dict(num_actions=action_space, type='int'),
                         network_spec=network_spec, optimizer=step_optimizer,
                         discount=0.99, batch_size=1000)

    logger.info('Initializing Runner')
    runner = Runner(agent=agent, environment=environment)


    report_episodes = args.episodes / 50  # default episodes = 500

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            logger.info(
                "Finished episode {ep} after {ts} timesteps. Steps Per Second ".format(ep=r.episode, ts=r.timestep))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 50 rewards: {}".format(sum(r.episode_rewards[-50:]) / 50))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))
    print("Starting {agent} for Environment".format(agent=agent))
    runner.run(episodes=args.episodes, max_episode_timesteps=1, episode_finished=episode_finished)
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))
    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))
    environment.close()


if __name__ == '__main__':
    main()
