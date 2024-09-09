import numpy as np
import pickle


def q_learning(env, episodes=500, learning_rate_a=0.1, discount_factor_g=0.9, epsilon=1, epsilon_decay_rate = 0.0001 ):
    # Inizializza la Q-table con dimensione (n * n, 4)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(episodes):
        print(f"Episode:{episode}")
        state = env.reset()
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                # Azione casuale (esplorazione)
                action = env.action_space.sample()
            else:
                # Azione migliore (sfruttamento)
                # action = np.argmax(q_table[state])
                action = np.argmax(q_table[state, :])

            next_state, reward, done, _ = env.step(action)

            # Aggiornamento della Q-table
            # q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

            q_table[state, action] = q_table[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q_table[next_state, :]) - q_table[state, action]
            )

            state = next_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if (epsilon == 0):
            learning_rate_a = 0.0001

    f = open("q_table_cust.pkl", "wb")
    pickle.dump(q_table, f)
    f.close()

    return q_table

