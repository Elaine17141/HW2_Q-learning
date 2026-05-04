import sys
import os

# 路徑保護：確保當前目錄被加入 sys.path，解決跨目錄執行時的模組找不到問題
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
import matplotlib.pyplot as plt
from cliff_walking import CliffWalkingEnv
from rl_algorithms import run_sarsa, run_q_learning

def plot_rewards(sarsa_rewards, q_rewards):
    """繪製 500 回合的累積獎勵曲線"""
    plt.figure(figsize=(10, 6))
    
    x = np.arange(1, len(sarsa_rewards) + 1)
    
    plt.plot(x, sarsa_rewards, label='Sarsa', color='c')
    plt.plot(x, q_rewards, label='Q-learning', color='r')
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward Sum for Episode')
    plt.title('Sarsa Vs. Q-Learning Cliff Walking\nEpsilon=0.1, Alpha=0.5\n(averaged over 50 runs)')
    plt.ylim([-100, 0]) 
    plt.legend()
    plt.grid(True)
    plt.savefig('rewards_curve.png')
    print("累積獎勵圖已儲存為 rewards_curve.png")

def get_best_policy(Q):
    """根據 Q-table 取出每個狀態下最佳的動作"""
    policy = np.zeros((4, 12), dtype=int)
    for i in range(4):
        for j in range(12):
            policy[i, j] = np.argmax(Q[i, j])
    return policy

def plot_policy(policy, title, filename):
    """使用 Matplotlib 將策略網格與箭頭畫出來"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # 網格設定
    ax.set_xlim(0, 12)
    ax.set_ylim(4, 0) # Matplotlib Y軸由上往下較符合矩陣視覺
    
    # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
    arrow_dict = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
    for i in range(4):
        for j in range(12):
            # 填入網格顏色與文字
            if i == 3 and 1 <= j <= 10:
                # 懸崖
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='gray', edgecolor='k', alpha=0.5))
                ax.text(j + 0.5, i + 0.5, 'CLIFF', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
            elif i == 3 and j == 0:
                # 起點
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='lightgreen', edgecolor='k', alpha=0.5))
                ax.text(j + 0.5, i + 0.5, 'START', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
            elif i == 3 and j == 11:
                # 終點
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='lightblue', edgecolor='k', alpha=0.5))
                ax.text(j + 0.5, i + 0.5, 'GOAL', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
            else:
                # 一般路徑，顯示最佳動作箭頭
                a = policy[i, j]
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='k'))
                ax.text(j + 0.5, i + 0.5, arrow_dict[a], ha='center', va='center', fontsize=20, color='blue')
                
    ax.set_xticks(np.arange(13))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='k', linestyle='-', linewidth=2)
    ax.set_title(title, fontsize=16)
    plt.savefig(filename)
    print(f"策略圖已儲存為 {filename}")

if __name__ == "__main__":
    env = CliffWalkingEnv()
    
    runs = 50
    episodes = 500
    
    all_sarsa_rewards = np.zeros((runs, episodes))
    all_q_rewards = np.zeros((runs, episodes))
    
    print(f"開始執行 {runs} 次獨立訓練來平均結果...")
    for r in range(runs):
        print(f"執行第 {r+1}/{runs} 次訓練...", end="\r")
        # 教科書與老師圖表參數：gamma=1.0, alpha=0.5
        sarsa_q, sarsa_rewards = run_sarsa(env, episodes=episodes, alpha=0.5, gamma=1.0)
        q_q, q_rewards = run_q_learning(env, episodes=episodes, alpha=0.5, gamma=1.0)
        
        all_sarsa_rewards[r] = sarsa_rewards
        all_q_rewards[r] = q_rewards
    
    print("\n訓練完成！")
    
    avg_sarsa_rewards = np.mean(all_sarsa_rewards, axis=0)
    avg_q_rewards = np.mean(all_q_rewards, axis=0)
    
    # 畫學習曲線
    plot_rewards(avg_sarsa_rewards, avg_q_rewards)
    
    # 取得策略並畫圖
    sarsa_policy = get_best_policy(sarsa_q)
    q_policy = get_best_policy(q_q)
    
    plot_policy(sarsa_policy, 'Final Policy - SARSA (Safe Path)', 'sarsa_policy.png')
    plot_policy(q_policy, 'Final Policy - Q-learning (Optimal Path)', 'q_learning_policy.png')
