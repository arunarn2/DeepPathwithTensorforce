# Deep Reinforcement Learning for Knowledge Graph Reasoning
We study the problem of learning to reason in large scale knowledge graphs (KGs). More specifically, we describe a novel reinforcement learning framework for learning multi-hop relational paths: we use a policy-based agent with continuous states based on knowledge graph embeddings, which reasons in a KG vector-space by sampling the most promising relation to extend its path. In contrast to prior work, our approach includes a reward function that takes the **accuracy**, **diversity**, and **efficiency** into consideration. Experimentally, we show that our proposed method outperforms a path-ranking based algorithm and knowledge graph embedding methods on Freebase and Never-Ending Language Learning datasets.


## Access the dataset
Download the knowledge graph dataset [NELL-995](http://cs.ucsb.edu/~xwhan/datasets/NELL-995.zip)

# Pre-requisites
## For Orginal Code
Python 2.7.14rc1
NumPy Version 1.13.3
TensorFlow Version 1.4.0

## TensorForce Implementation
Python 2.7.14rc1
NumPy Version 1.13.3
TensorForce Version 0.3.2

## How to run Tensorforce Code
1. Unzip the NELL-995 dataset in the top level code directory of your project executing following command
   * `wget http://cs.ucsb.edu/%7Exwhan/datasets/NELL-995.zip`
   * `unzip NELL-995.zip`

   This command will create [NELL-995/folder].

   See Format of the Dataset section for more details about the dataset.


2. Run the following script within `tfscripts/`
   * ` python deepPath_main.py -a <agent_name> -r <relation_name> `

   Example
   * ` python deepPath_main.py -a vpg -r concept_athletehomestadium`
3. Parameter accepted by the program
    - `-r`  or `--relation` = relation name (Mandatory)
    - `-e`  or `--episodes` = Number of episodes, default 500
    - `-a`  or `--agent` = Agent Name, default `vpg`
        - We have implemented two agents:
            - `vpg`
            - `dqn`
    - `-D`  or `--debug` = Debug Log, default `False`

## How to run original code
1. unzip the NELL, put the data folder in the code directory
2. run the following scripts within `scripts/`
    *   `./pathfinder.sh ${relation_name}`  # find the reasoning paths, this is RL training, it might take sometime
    *   `./fact_prediction_eval.py ${relation_name}` # calculate & print the fact prediction results
    *   `./link_prediction_eval.sh ${relation_name}` # calculate & print the link prediction results

    Examples (the relation_name can be found in `NELL-995/tasks/`):
    * `./pathfinder.sh concept_athletehomestadium`
    * `./fact_prediction_eval.py concept_athletehomestadium`
    * `./link_prediction_eval.sh concept_athletehomestadium`
3. Since we already put the reasoning paths in the dataset, you can directly run fact_prediction_eval.py or link_prediction_eval.sh to get the final results for each reasoning task

## Format of the Dataset
1. `raw.kb`: the raw kb data from NELL system
2. `kb_env_rl.txt`: we add inverse triples of all triples in `raw.kb`, this file is used as the KG for reasoning
3. `entity2vec.bern/relation2vec.bern`: transE embeddings to represent out RL states, can be trained using [TransX implementations by thunlp](https://github.com/thunlp/Fast-TransX)
4. `tasks/`: each task is a particular reasoning relation
    * `tasks/${relation}/*.vec`: trained TransH Embeddings
    * `tasks/${relation}/*.vec_D`: trained TransD Embeddings
    * `tasks/${relation}/*.bern`: trained TransR Embedding trained
    * `tasks/${relation}/*.unif`: trained TransE Embeddings
    * `tasks/${relation}/transX`: triples used to train the KB embeddings
    * `tasks/${relation}/train.pairs`: train triples in the PRA format
    * `tasks/${relation}/test.pairs`: test triples in the PRA format
    * `tasks/${relation}/path_to_use.txt`: reasoning paths found the RL agent
    * `tasks/${relation}/path_stats.txt`: path frequency of randomised BFS

## Implementation Details

### Environment

A new environment class was created to interface with tensorforce library.

```python
environment = DPEnv(graphPath, relationPath, task=data)
```

### Network Definition
We have defined Neural Network with 2 hidden layer using tensoforce's API.

```python
network_spec = [
        dict(type='dense', size=512, activation='relu'),
        dict(type='dense', size=1024, activation='relu')
    ]
```

### Agent Creation
We are using `VPGAgent()` and `DQNAgent()` from `tensorforce.agents` API to create the agents.

```python

from tensorforce.agents import VPGAgent
from tensorforce.agents import DQNAgent

[...]

    agent = VPGAgent(states_spec=dict(shape=state_dim, type='float'),
                         actions_spec=dict(num_actions=action_space, type='int'),
                         network_spec=network_spec, optimizer=step_optimizer,
                         discount=0.99, batch_size=1000)

    agent = DQNAgent(states_spec=dict(shape=state_dim, type='float'),
                         actions_spec=dict(num_actions=action_space, type='int'),
                         network_spec=network_spec, optimizer=step_optimizer,
                         discount=0.99, batch_size=1000)

```

### Optimization Algorithm
We have used `adam` is the Optimization Algorithm, however `RMSprop`,`AdaGrad` or other algorithms can also be used.

```python
step_optimizer = dict(type='adam', learning_rate=1e-3)
```

### Execution
`runner` from `tensorforce.execution` API has been used for learning.

```python
from tensorforce.execution import Runner

runner = Runner(agent=agent, environment=environment)
runner.run(episodes = args.episodes, max_episode_timesteps=report_episodes, episode_finished=episode_finished)

```



## Cite
```
@InProceedings{wenhan_emnlp2017,
  author    = {Xiong, Wenhan and Hoang, Thien and Wang, William Yang},
  title     = {DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP 2017)},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {ACL}
}
```

## Acknowledgement
* [TransX implementations by thunlp](https://github.com/thunlp/Fast-TransX)
* [Ni Lao's PRA code](http://www.cs.cmu.edu/~nlao/)
