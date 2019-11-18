import numpy as np
import gym
env = gym.make('FrozenLake8x8-v0')
env.seed(0)

def evaluate_policy(env, policy):
  total_rewards = 0.0
  for _ in range(100):
    obs = env.reset()
    while True:
      action = policy[obs]
      obs, reward, done, info = env.step(action)
      total_rewards += reward
      if done:
        break
  return total_rewards/100

def crossover(policy1, policy2):
  new_policy = policy1.copy()
  for i in range(16):
      rand = np.random.uniform()
      if rand > 0.5:
          new_policy[i] = policy2[i]
  return new_policy

def mutation(policy):
  new_policy = policy.copy()
  for i in range(64):
    rand = np.random.uniform()
    if rand < 0.05:
      new_policy[i] = np.random.choice(4)
  return new_policy

k=25
policy_pop = [np.random.choice(4, size=((64))) for _ in range(100)]
for idx in range(25):
  policy_scores = [evaluate_policy(env, pp) for pp in policy_pop]
  policy_ranks = list(reversed(np.argsort(policy_scores)))
  elite_set= [policy_pop[x] for x in policy_ranks[:k]]
  select_probs = np.array(policy_scores) / np.sum(policy_scores)
  child_set = [crossover(
      policy_pop[np.random.choice(range(100), p=select_probs)], 
      policy_pop[np.random.choice(range(100), p=select_probs)])
      for _ in range(100 - k)]
  k-=1
  mutated_list = [mutation(c) for c in child_set]
  policy_pop = elite_set
  policy_pop += mutated_list
policy_score = [evaluate_policy(env, pp) for pp in policy_pop]
best_policy = policy_pop[np.argmax(policy_score)]
print('Best actions score =', (np.max(policy_score)),'best actions =', best_policy.reshape(8,8))
env.close()
