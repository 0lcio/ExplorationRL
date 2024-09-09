import numpy as np
from test_q_learning.agent_q_learning import q_learning
from custum_env import GridMappingEnv
import pickle


def run(episodes, is_training=True):
    env = GridMappingEnv(n=5)

    # Addestramento dell'agente
    if is_training:
        q_table = q_learning(env, episodes=episodes)
    else:
        f = open("test_q_learning/q_table_cust.pkl", 'rb')
        q_table = pickle.load(f)
        f.close()

    print(q_table)

    # Valutazione dell'agente
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, _, done, _ = env.step(action)
        env.render()

    env.close()


run(episodes=10000, is_training=True)
