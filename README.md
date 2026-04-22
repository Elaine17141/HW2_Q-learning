# Cliff Walking: Q-learning vs. SARSA

## 📌 Project Overview
本專案為強化學習 (Reinforcement Learning) 的經典作業，旨在實作與比較 **On-policy (SARSA)** 與 **Off-policy (Q-learning)** 演算法在 Cliff Walking (懸崖探索) 網格世界環境中的行為差異。

透過這個 4x12 的網格環境，我們觀察這兩種演算法如何在探索 (Exploration) 與利用 (Exploitation) 的妥協中，學習到截然不同的路徑策略。

## ⚙️ Implementation Details
- **Environment**: 4x12 Grid World 
  - 起點：`(3,0)`
  - 終點：`(3,11)`
  - 懸崖：`(3,1)` 到 `(3,10)`，踩入懸崖獲得 -100 reward 且被傳送回起點。一般移動 reward 為 -1。
- **Algorithms**: SARSA (On-policy) & Q-learning (Off-policy)
- **Policy**: $\epsilon$-greedy strategy
- **Hyperparameters**:
  - 探索率 ($\epsilon$, Epsilon): `0.1`
  - 學習率 ($\alpha$, Alpha): `0.5`
  - 折扣因子 ($\gamma$, Gamma): `0.9`
  - 訓練回合數 (Episodes): `500`

## 📊 結果分析 (Results Analysis)

### 1. 學習表現與收斂性 (Learning Performance & Stability)
![Learning Curve](./rewards_curve.png)

從累積獎勵曲線可以觀察到：
* **Q-learning** 的波動非常劇烈，且在整個訓練過程中，其平均獎勵一直處於較低（負值較大）的狀態。這是因為 Q-learning 學習到了貼著懸崖的最短路徑，但在 $\epsilon$-greedy 策略的探索機制下，有 10% 的機率會發生隨機行為，導致它頻繁地掉入懸崖並獲得 -100 的巨大懲罰。
* **SARSA** 雖然在初始學習階段同樣會經歷波動，但它很快就收斂到一個相對穩定且較高的平均獎勵。這是因為 SARSA 學習到了一條遠離懸崖的安全路徑，成功避開了探索時因「手抖」而摔落懸崖的風險。

### 2. 策略行為與理論對比 (Strategic Behavior & Theoretical Contrast)

| 演算法 | 策略路徑圖 | 本質差異與行為解釋 |
| :---: | :---: | :--- |
| **Q-learning**<br>*(Off-policy)* | ![Q-learning Policy](./q_learning_policy.png) | **理論最佳、行為激進 (Aggressive)**<br>Q-learning 更新時使用的是 $\max_a Q(S', a)$，它永遠**樂觀地假設**未來會採取最佳行動，完全忽略了現實中 $\epsilon$-greedy 帶來的隨機性。因此，它找到了理論上的最短路徑（貼著懸崖走），但代價是在訓練期間經常因隨機探索而墜崖。 |
| **SARSA**<br>*(On-policy)* | ![SARSA Policy](./sarsa_policy.png) | **考量現實、行為保守 (Conservative)**<br>SARSA 更新時使用的是實際採取的行動 $Q(S', A')$。由於它**真實感受到** $\epsilon$-greedy 探索帶來的墜崖風險，為了在訓練過程中最大化期望報酬，它學會了「繞遠路」來確保安全。這條路徑雖然較長，但非常安穩。 |

## 🎯 結論 (Conclusion)

根據上述的實驗結果，我們總結出兩種演算法的適用場景：

* **Q-learning (適合完美的模擬環境)**：
  由於它不受當下探索策略的干擾，能夠穩定收斂到**絕對最佳解 (Optimal Policy)**。如果環境是純粹的電腦模擬器，且在最終部署測試時可以將 $\epsilon$ 降為 0（取消探索），那麼 Q-learning 是極佳的選擇，因為它的最終策略是最有效率的。
* **SARSA (適合真實物理世界 / 機器人應用)**：
  由於它將探索的風險（如馬達的硬體誤差、隨機抖動）也一併納入學習考量中，學出來的策略更加**強健且安全 (Safe & Robust)**。在現實世界的機器人應用中，跌入懸崖（硬體損壞）的成本是不可逆且極其昂貴的，這時候 SARSA 這種會主動規避高風險區域的保守演算法將會是首選。

## 🚀 How to Run

### 1. 安裝依賴環境
我們建議使用 Python 3.8 以上的開發環境，請在終端機內輸入以下指令安裝所需套件：

```bash
pip install -r requirements.txt
```

### 2. 執行主程式
本專案已經將環境、演算法訓練與圖表繪製整合，只要執行以下指令，即會進行 500 回合訓練並在當下目錄產出相應結果圖：

```bash
python plot_results.py
```
> *(執行完畢後會產生 `rewards_curve.png`, `sarsa_policy.png`, 與 `q_learning_policy.png`)*
