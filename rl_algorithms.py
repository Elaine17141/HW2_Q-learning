import numpy as np
from cliff_walking import CliffWalkingEnv

def epsilon_greedy_policy(Q, state, epsilon, n_actions):
    """
    epsilon-greedy 策略
    以 1-epsilon 的機率選擇最大 Q 值的動作，以 epsilon 的機率隨機選擇動作。
    """
    if np.random.rand() < epsilon:
         # 隨機探索
        return np.random.randint(n_actions)
    else:
        # 利用 (Exploitation)，如果有多個相同的最大值，就隨機從中選一個（處理起步全為 0 的情況）
        q_values = Q[state[0], state[1]]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)

def run_sarsa(env, episodes=500, alpha=0.5, gamma=0.9, epsilon=0.1):
    """
    SARSA 演算法 (On-policy)
    """
    # 初始化 Q table，所有狀態動作對的值皆為 0
    Q = np.zeros((env.shape[0], env.shape[1], env.n_actions))
    # 儲存每回合的總獎勵 (Reward Sum)
    rewards_per_episode = np.zeros(episodes)
    
    for ep in range(episodes):
        state = env.reset()
        # On-policy：第一步就從當前 policy 決定 action
        action = epsilon_greedy_policy(Q, state, epsilon, env.n_actions)
        total_reward = 0
        done = False
        
        while not done:
            # 執行動作，觀察環境
            next_state, reward, done = env.step(action)
            
            # On-policy：根據目前的 policy 去決定「下一步的行動」，並以此來更新 Q 值
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, env.n_actions)
            
            # SARSA Update 公式：Q(S,A) <- Q(S,A) + alpha * [R + gamma * Q(S', A') - Q(S,A)]
            # 如果抵達終點，則 Q(S', A') 為 0
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * Q[next_state[0], next_state[1], next_action]
                
            td_error = td_target - Q[state[0], state[1], action]
            Q[state[0], state[1], action] += alpha * td_error
            
            # 推進狀態與動作
            state = next_state
            action = next_action
            total_reward += reward
            
        rewards_per_episode[ep] = total_reward
        
    return Q, rewards_per_episode

def run_q_learning(env, episodes=500, alpha=0.5, gamma=0.9, epsilon=0.1):
    """
    Q-learning 演算法 (Off-policy)
    """
    # 初始化 Q table
    Q = np.zeros((env.shape[0], env.shape[1], env.n_actions))
    # 儲存每回合的總獎勵 (Reward Sum)
    rewards_per_episode = np.zeros(episodes)
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 從當前 policy 決定 action
            action = epsilon_greedy_policy(Q, state, epsilon, env.n_actions)
            
            # 執行動作，觀察環境
            next_state, reward, done = env.step(action)
            
            # Off-policy：不理會我們下一步「事實上」會選什麼動作，
            # 總是假設我們會選擇能獲得「最大 Q 值」的動作來進行更新
            if done:
                td_target = reward
            else:
                best_next_action = np.argmax(Q[next_state[0], next_state[1]])
                td_target = reward + gamma * Q[next_state[0], next_state[1], best_next_action]
                
            td_error = td_target - Q[state[0], state[1], action]
            Q[state[0], state[1], action] += alpha * td_error
            
            # 推進狀態
            state = next_state
            total_reward += reward
            
        rewards_per_episode[ep] = total_reward
        
    return Q, rewards_per_episode

if __name__ == "__main__":
    env = CliffWalkingEnv()
    
    episodes = 500
    alpha = 0.5
    gamma = 0.9
    epsilon = 0.1
    
    print("--- 執行 SARSA ---")
    sarsa_q, sarsa_rewards = run_sarsa(env, episodes, alpha, gamma, epsilon)
    print(f"SARSA 最後 10 回合的平均獎勵: {np.mean(sarsa_rewards[-10:]):.1f}")
    
    print("\n--- 執行 Q-Learning ---")
    q_learning_q, q_learning_rewards = run_q_learning(env, episodes, alpha, gamma, epsilon)
    print(f"Q-Learning 最後 10 回合的平均獎勵: {np.mean(q_learning_rewards[-10:]):.1f}")
