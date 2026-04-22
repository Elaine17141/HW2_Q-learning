import numpy as np

class CliffWalkingEnv:
    """
    Cliff Walking Environment (4x12 Grid World)
    
    Start state: (3, 0)
    Goal state: (3, 11)
    Cliff: (3, 1) to (3, 10)
    
    Actions:
        0: UP
        1: RIGHT
        2: DOWN
        3: LEFT
    """
    def __init__(self):
        self.shape = (4, 12)
        self.start_state = (3, 0)
        self.goal_state = (3, 11)
        self.n_actions = 4
        self.action_space = np.arange(self.n_actions)
        
        self.current_state = self.start_state

    def reset(self):
        """
        重置環境狀態回到起點
        :return: state (row, col)
        """
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        """
        執行動作並返回新的狀態、獎勵與是否結束
        :param action: int, 動作 (0:上, 1:右, 2:下, 3:左)
        :return: next_state, reward, done
        """
        row, col = self.current_state

        # 計算下一步位置 (限制在邊界內)
        if action == 0:   # 上
            row = max(row - 1, 0)
        elif action == 1: # 右
            col = min(col + 1, self.shape[1] - 1)
        elif action == 2: # 下
            row = min(row + 1, self.shape[0] - 1)
        elif action == 3: # 左
            col = max(col - 1, 0)
        else:
            raise ValueError("無效的動作！動作必須介於 0 到 3 之間。")

        next_state = (row, col)
        
        # 判斷是否掉入懸崖
        # 懸崖位於底部 (3, 1) 到 (3, 10)
        if next_state[0] == 3 and 1 <= next_state[1] <= 10:
            reward = -100
            next_state = self.start_state
            done = False
        # 判斷是否抵達終點
        elif next_state == self.goal_state:
            reward = -1
            done = True
        # 一般移動
        else:
            reward = -1
            done = False

        self.current_state = next_state
        
        return next_state, reward, done

    def render(self):
        """
        簡單的圖形化印出當前環境狀態
        x: 起點, T: 終點, C: 懸崖, A: 智能體目前位置, o: 安全區域
        """
        for r in range(self.shape[0]):
            row_str = ""
            for c in range(self.shape[1]):
                if (r, c) == self.current_state:
                    row_str += "A " # Agent
                elif (r, c) == self.start_state:
                    row_str += "S " # Start
                elif (r, c) == self.goal_state:
                    row_str += "G " # Goal
                elif r == 3 and 1 <= c <= 10:
                    row_str += "C " # Cliff
                else:
                    row_str += "o " # Safe path
            print(row_str)
        print("-" * 25)

if __name__ == "__main__":
    # 引入訓練演算法與繪圖函數
    from rl_algorithms import run_sarsa, run_q_learning
    from plot_results import plot_rewards

    env = CliffWalkingEnv()
    print("初始狀態：")
    env.render()
    
    print("--- 開始訓練 SARSA ---")
    sarsa_q, sarsa_rewards = run_sarsa(env, episodes=500)
    
    print("--- 開始訓練 Q-Learning ---")
    q_learning_q, q_learning_rewards = run_q_learning(env, episodes=500)
    
    print("--- 產出獎勵曲線圖 ---")
    plot_rewards(sarsa_rewards, q_learning_rewards)
