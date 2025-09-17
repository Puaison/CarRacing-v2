# 🚗 Reinforcement Learning with DDQN and Prioritized Experience Replay on CarRacing-v2

<p align="center">
<img src="demo.gif" width="500"/>
</p>

## 📌 Overview
This project is a **Reinforcement Learning (RL)** implementation applied to the [`CarRacing-v2`](https://gymnasium.farama.org/environments/box2d/car_racing/) environment from OpenAI Gym.  
Specifically, it uses a **Double Deep Q-Network (DDQN)** combined with a **Prioritized Experience Replay (PER) Buffer**.  
The agent learns directly from **image observations**, leveraging the convolutional architecture introduced by DeepMind for Atari games, showing its adaptability to different environments.

---

## 📂 Project Structure
The project is organized as follows:

- **`model_fast.pt`** → The fastest model obtained (default one loaded).  
- **`model_precise.pt`** → The most precise model, slower but more stable.  
- **`student.py`** → Core implementation: RL training loop, preprocessing, PER buffer, and utilities.  
- **`other_implementation.py`** → Alternative PER buffer implementation using **SumTree**.  

---

## 🖼 Pre-Processing
Since the RL agent expects **four stacked grayscale frames of size 84×84**, preprocessing includes:
1. Conversion to grayscale.  
2. Cropping to 84×84 to remove irrelevant parts.  
3. Normalization (pixel values scaled between 0–1).  
4. Frame stacking (last 4 images) to capture temporal information.  

Additionally:
- At the beginning of each episode, the first **60 frames are skipped** with no actions.  
- A `super_restart` function ensures proper environment resets.  

---

## 🔄 The Spin-off Turn Problem
One of the toughest challenges in this RL task was handling **hairpin turns at high speed**, where the car may spin 180° and drive in the wrong direction.  
By tuning PER hyperparameters, the agent was able to recover in most of these scenarios, balancing **speed vs. precision** trade-offs.

---

## 🎯 Reward Reshaping
Reward assignment logic for the RL agent:
- Standard environment rewards are preserved.  
- If no positive reward is collected for **100 steps**, a penalty of **-100** is applied (simulating a “death”).  
- Maximum episode length reduced from 1000 to **930 steps** to avoid reward miscalculations when laps finished prematurely.  
- Transitions are stored in the PER buffer with maximum priority to ensure at least one replay.  

---

## 🧠 Prioritized Experience Replay (PER)
PER allows the RL agent to focus training on **rare and informative transitions** by sampling based on **TD-error**:

- **Sampling probability:**  
  ![Sampling probability](https://latex.codecogs.com/svg.latex?P(i)%20%3D%20%5Cfrac%7Bp_i%5E%5Calpha%7D%7B%5Csum_n%20p_n%5E%5Calpha%7D)

- **Importance Sampling (IS) weights:**  
  ![IS weights](https://latex.codecogs.com/svg.latex?w_i%20%3D%20%5Cleft%28%5Cfrac%7B1%7D%7BN%7D%20%5Ccdot%20%5Cfrac%7B1%7D%7BP%28i%29%7D%5Cright%29%5E%7B%5Cbeta%7D)


- Implementations:
  - **Vanilla PER**: simple array-based.  
  - **SumTree PER**: more efficient (O(log n) updates), faster, and less memory intensive.  

Both achieved **similar performance**, but SumTree scales better with larger buffers.

---

## 🤖 DDQN and RL Training
- Two neural networks are used:
  - **Q-network** (updated every step).  
  - **Target network** (updated every 100 steps).  
- RL Training phases:
  - **Burn-in**: 20,000 random steps to populate the buffer.  
  - **ϵ-greedy exploration**: initial ϵ = 0.7 (decay 0.99 per episode).  
  - **Batch updates**: mini-batches of 64, PER-based sampling.  

**TD-error used for priority / updates:**  
![TD-error](https://latex.codecogs.com/svg.latex?%5Cdelta_j%20%3D%20R_j%20%2B%20%5Cgamma%20Q_%7B%5Ctext%7Btarget%7D%7D%28S_j%2C%20%5Coperatorname%7Bargmax%7D_a%20Q%28S_j%2C%20a%29%29%20-%20Q%28S_%7Bj-1%7D%2C%20A_%7Bj-1%7D%29)

The **loss function** incorporates IS weights:  
![Loss](https://latex.codecogs.com/svg.latex?L%20%3D%20%5Cmathrm%7Bmean%7D%28%28%5Cdelta%5E2%29%20%5Ccdot%20w%29)

---

## 📊 Results and Comparisons
- **α = 0.1, β = 0.1** → stable, but struggled in rare/critical situations.  
- **α = 0.6, β = 0.4 → 1 (annealed)** → significant improvements, especially in recovering from spin-off turns.  

**Buffer size impact:**
| Buffer Size | Highest Mean Reward | Cruise Mean Reward |
|-------------|----------------------|--------------------|
| 50,000      | 520                  | 300–400            |
| 100,000     | 592                  | 400–500            |

Moreover, accordingly to the number of update steps of the algorithm, I noticed different features:
- With reasoneably numbers of iterations, a **Faster model** model is obtained → more aggressive, faster laps, but riskier.  
- By doing more iterations, a **More Precise model** is obtained → safer, fewer mistakes but slower in completing the lap. 

---

## 🚀 Future Work
- Investigate smarter strategies for **sample replacement** in the PER buffer.  
- Explore hybrid approaches to balance **speed and precision** in agent behavior.  
- Extend to **continuous action spaces** and compare with policy-gradient RL methods.  

---

## 📚 References
1. [DDQN Solution Notebook (KRL Group)](https://github.com/KRLGroup/RL_2024/blob/main/p06_DDQN/DDQN_sol.ipynb)  
2. [Van Hasselt et al. – Double Q-learning (2015)](https://arxiv.org/abs/1509.06461)  
3. [Howuhh – Prioritized Experience Replay (GitHub)](https://github.com/Howuhh/prioritized_experience_replay)  
4. [Mnih et al. – Atari DQN (2013)](https://arxiv.org/abs/1312.5602)  
5. [Schaul et al. – PER (2016)](https://arxiv.org/abs/1511.05952)  
