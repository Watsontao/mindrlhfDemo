# GRPO算法介绍

GRPO（Group Relative Policy Optimization）是DeepSeek在2024年提出的用于大语言模型（LLM）后训练的强化学习算法，旨在解决传统PPO算法在LLM训练中的效率问题。

## 核心思想

GRPO的核心创新是**去除值函数模型**（Critic），通过**组内相对优势估计**替代传统的优势函数计算：

$$
\hat{A}_{i,t} = \frac{r_i - \operatorname{mean}(\boldsymbol{r})}{\operatorname{std}(\boldsymbol{r})}
$$

其中：
- $r_i$ 是回答$o_i$的奖励分数
- $\boldsymbol{r} = \{r_1,\cdots,r_G\}$ 是同一问题下$G$个回答的奖励集合
- 优势值在组内标准化

## 算法流程

1. **采样阶段**：
   - 对每个问题$q$，用当前策略$\pi_{\theta_{\text{old}}}$生成$G$个回答$\{o_i\}_{i=1}^G$
   - 使用奖励模型计算每个回答的奖励$r_i$

2. **优势估计**：
   $$
   \hat{A}_{i,t} = \frac{r_i - \mu}{\sigma} \quad \text{其中} \quad \mu=\operatorname{mean}(\boldsymbol{r}), \sigma=\operatorname{std}(\boldsymbol{r})
   $$

3. **目标函数**：
   $$
   \begin{aligned}
   \mathcal{J}_{\text{GRPO}}(\theta) &= \mathbb{E}[q\sim P(Q),\{o_i\}_{i=1}^G\sim\pi_{\theta_{\mathrm{old}}}(O|q)] \\
   & \left[ \frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left( \min\left[ r_t(\theta)\hat{A}_{i,t}, \operatorname{clip}\left(r_t(\theta),1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t} \right] - \beta\mathrm{D}_{\mathrm{KL}}[\pi_\theta \parallel \pi_{\text{ref}}] \right) \right]
   \end{aligned}
   $$
   - $|o_i|$：回答长度
   - $r_t(\theta)$：token级重要性采样比率
    $$
    r_t(\theta) =
    \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})}
    $$
   - clip机制：限制策略更新幅度，避免过大更新导致不稳定
    $$
        \operatorname{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)=
        \begin{cases} 
        1+\varepsilon & \text{if } r_t(\theta) > 1+\varepsilon \\
        1-\varepsilon & \text{if } r_t(\theta) < 1-\varepsilon \\
        r_t(\theta) & \text{otherwise}
        \end{cases}
    $$
    - $\mathrm{D}_{\mathrm{KL}}[\pi_\theta \parallel \pi_{\text{ref}}]$：KL散度惩罚项，防止策略偏离参考模型$\pi_{\text{ref}}$太远，$\beta$是超参数。[KL散度估计方法](http://joschu.net/blog/kl-approx.html)如下：
    $$
    \mathrm{D}_{\mathrm{KL}}\left[\pi_\theta||\pi_{ref}\right]=\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}-\log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}-1
    $$

4. **优化阶段**：
    - 使用梯度下降最小化目标函数$\mathcal{J}_{\text{GRPO}}(\theta)$
    - 更新策略参数$\theta$

## 关键优势

1. **无值函数模型**：
   - 显著节省显存
   - 减少计算复杂度

2. **稳定训练**：
   - Clip机制限制策略更新幅度
   - KL散度防止偏离参考模型太远

3. **高效优势估计**：
   - 组内相对比较避免绝对值依赖
   - 标准化处理提升数值稳定性

> 论文：Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) (2024)