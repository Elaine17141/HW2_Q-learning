import sys
import os

# [測試代碼] 檢查 sys.path 是否包含當前目錄
print("=== 系統路徑檢查 ===")
print("當前檔案所在目錄:", os.path.dirname(os.path.abspath(__file__)))
print("當前 sys.path 列表:")
for p in sys.path:
    print(" -", p)
print("====================\n")

import numpy as np
import matplotlib.pyplot as plt
from cliff_walking import CliffWalkingEnv
from rl_algorithms import run_sarsa, run_q_learning

def moving_average(a, n=10):
    """計算移動平均值，讓曲線更加平滑易讀"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_rewards(sarsa_rewards, q_rewards):
    """繪製 500 回合的累積獎勵曲線"""
    plt.figure(figsize=(10, 6))
    
    # 建議加上移動平均以去除噪音，讓趨勢比較明顯
    sarsa_ma = moving_average(sarsa_rewards, 10)
    q_ma = moving_average(q_rewards, 10)
    
    # 移動平均後長度會減少，設定對應的 X 軸讓圖表對齊
    x = np.arange(10, len(sarsa_rewards) + 1)
    
    plt.plot(x, sarsa_ma, label='SARSA', color='blue')
    plt.plot(x, q_ma, label='Q-learning', color='red')
    
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode (Smoothed)')
    plt.title('SARSA vs Q-learning on Cliff Walking (500 Episodes)')
    # 最佳路徑的獎勵為 -13，安全路徑為 -17。原本的 [-100, -20] 會導致最後收斂的線條消失在圖表上方
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
    
    print("訓練 SARSA...")
    sarsa_q, sarsa_rewards = run_sarsa(env, episodes=500)
    
    print("訓練 Q-learning...")
    q_q, q_rewards = run_q_learning(env, episodes=500)
    
    # 畫學習曲線
    plot_rewards(sarsa_rewards, q_rewards)
    
    # 取得策略並畫圖
    sarsa_policy = get_best_policy(sarsa_q)
    q_policy = get_best_policy(q_q)
    
    plot_policy(sarsa_policy, 'Final Policy - SARSA (Safe Path)', 'sarsa_policy.png')
    plot_policy(q_policy, 'Final Policy - Q-learning (Optimal Path)', 'q_learning_policy.png')
