# DAPO算法介绍

**核心目标**：DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)算法旨在解决大规模LLM强化学习中的训练不稳定、探索不足和效率低下问题，提升复杂推理能力。基于GRPO改进，核心公式如下：

## 基础目标函数
$$
\begin{aligned}
\mathcal{J}_{\text{DAPO}}(\theta) &= \mathbb{E}_{(q,a)\sim\mathcal{D},\{o_i\}_{i=1}^G\sim\pi_{\theta_{\text{old}}}(\cdot|q)} \\
&\left[ \frac{1}{\sum_{i=1}^G |o_i|}\sum_{i=1}^G \sum_{t=1}^{|o_i|} 
\min\left( r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\left( r_{i,t}(\theta),\ 1-\varepsilon_{\text{low}},\ 1+\varepsilon_{\text{high}} \right)\hat{A}_{i,t} \right) 
\right]\\
&\text{s.t.}\ \ 0 < \left| \{ o_i \mid \texttt{equivalent}(a,o_i) \} \right| < G
\end{aligned}
$$

其中：
- $r_{i,t}(\theta)=\dfrac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}$（重要性采样比）
- $\hat{A}_{i,t}=\dfrac{R_i-\text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}$（组标准化优势函数）

## 关键技术公式
**(1) Clip-Higher**  
   - **问题**：传统PPO/GRPO的上限剪切（$\varepsilon_{\text{high}}=0.2$）抑制低概率“探索性token”的更新，导致熵崩溃（样本多样性丧失）。  
   - **方案**：解耦剪切范围 → **$\varepsilon_{\text{low}}=0.2$（保持下界约束）** , **$\varepsilon_{\text{high}}=0.28$（放宽上界约束）**。  
   - **效果**：提升低概率token的探索空间，显著增加策略熵和样本多样性。
$$
\min\left( r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\left( r_{i,t}(\theta),\ 1-\varepsilon_{\text{low}},\ 1+\varepsilon_{\text{high}} \right)\hat{A}_{i,t} \right) 
$$

**(2) Dynamic Sampling**   
   - **问题**：当整组样本全对/全错时，优势函数为零，梯度信号消失，训练效率下降。  
   - **方案**：动态过采样并过滤掉全对/全错的样本组，确保批次内包含**混合正确率**的样本。  
   - **效果**：稳定梯度方向，加速收敛，且生成开销可控。
$$\text{s.t.}\ \ 0 < \left| \{ o_i \mid \texttt{equivalent}(a,o_i) \} \right| < G$$

**(3) Token-Level Loss**  
   - **问题**：GRPO的样本级损失平均削弱长文本中关键token的梯度信号。  
   - **方案**：改为**Token级损失计算**，使长序列中的低质量模式（如重复文本）获得更强惩罚。  
   - **效果**：抑制过长低质生成，健康控制响应长度和熵。
$$
\mathcal{J}_{\text{DAPO}} = \frac{1}{\sum_{i=1}^G |o_i|}\sum_{i=1}^G \sum_{t=1}^{|o_i|}  \text{loss}(o_{i,t}) \quad (\text{非} \ \frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} )
$$

**(4) Overlong Reward Shaping**  
   - **问题**：直接截断超长样本并惩罚（R=-1）引入奖励噪声，干扰训练。  
   - **方案**：分段柔性惩罚：  
$$R_{\text{length}}(y) = 
\begin{cases} 
0, & |y| \leq L_{\max}-L_{\text{cache}} \\ 
\frac{(L_{\max}-L_{\text{cache}})-|y|}{L_{\text{cache}}}, & L_{\max}-L_{\text{cache}} < |y| \leq L_{\max} \\ 
-1, & |y| > L_{\max}
\end{cases}$$
   - **效果**：减少误惩罚，提升训练稳定性。

> 开源：完整实现见[verl框架](https://github.com/volcengine/verl)，数据集[DAPO-Math-17K](https://dapo-sia.github.io/)

> 论文：Yu et al. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](http://arxiv.org/abs/2503.14476) (2025)