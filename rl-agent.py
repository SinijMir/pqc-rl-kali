#!/usr/bin/env python3
# rl-agent.py – RL-оптимизация параметров Kyber под Qiskit
import random, time, math
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import Aer

# 1. Гиперпараметры
STATE = {'k': 3, 'n': 256, 'q': 3329}        # Kyber-512
MUTATE_STEP = 5                              # шаг мутации
EPISODES = 10                                # кол-во эпох
REWARD_HISTORY = []

# 2. Функция мутации
def mutate(state):
    new = state.copy()
    key = random.choice(['k', 'n', 'q'])
    if key == 'k':      new['k']   = max(2, new['k']   + random.choice([-1, 1]))
    if key == 'n':      new['n']   = max(128, new['n'] + random.choice([-16, 16]))
    if key == 'q':      new['q']   = max(2, new['q']   + random.choice([-64, 64]))
    return new

# 3. Функция награды (чем меньше время и размер, тем лучше)
def reward(state):
    backend = Aer.get_backend('aer_simulator')
    qc = QuantumCircuit(state['n'])
    qc.h(range(state['n']))
    qc.measure_all()
    t0 = time.time()
    backend.run(qc).result()
    t1 = time.time()
    size = state['n'] * state['k'] * 2
    return -(t1 - t0 + size / 1000)          # минус, чтобы maximise

# 4. RL-цикл
best_state, best_reward = STATE, reward(STATE)
for ep in range(EPISODES):
    new_state = mutate(best_state)
    new_reward = reward(new_state)
    REWARD_HISTORY.append(new_reward)
    if new_reward > best_reward:
        best_state, best_reward = new_state, new_reward
        print(f"Эпоха {ep+1}: новый лучший reward = {best_reward:.4f}, параметры = {best_state}")

# 5. Отчёт
print("\n=== RL-отчёт ===")
print(f"Лучший набор параметров: {best_state}")
print(f"Лучший reward: {best_reward:.4f}")
print(f"Средняя награда: {sum(REWARD_HISTORY)/len(REWARD_HISTORY):.4f}")
print("=== End of RL-agent ===")
