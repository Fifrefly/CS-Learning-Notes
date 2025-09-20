# Transformer 核心概念与结构

## 0. 核心思想与动机（Why Transformer）
为什么选择自注意力（Self-Attention）而不是 RNN/CNN？
- 并行能力：自注意力一次性计算整个序列的两两交互，训练可完全并行；RNN 需按时间步串行，难以充分利用现代加速器。
- 长距离依赖：任意两位置间的“信息路径长度”在注意力中为 $O(1)$，在 RNN 中为 $O(n)$，后者易出现梯度消失/爆炸，难以建模远距离依赖。
- 内容寻址（content-based addressing）：注意力以“内容相似度”直接路由信息，减少固定顺序偏置；顺序信息通过“位置编码”显式注入。
- 可扩展性取舍：自注意力在序列长度上是 $O(n^2)$ 的时间/显存开销，但换来更强的全局建模与并行吞吐；由此衍生出许多高效注意力/稀疏注意力改进。

## 1.Encoder 的结构
一个Encoder包含N个Encoder层，以下以一个Encoder层为例解析结构。
### 1.1 输入
一层Encoder的输入可以分为两类，第一类是第一个Encoder层的输入，其由单词的嵌入(见1.1.1)和位置的编码(见1.1.2)之和组成；第二类是其他Encoder层的输入，即上一层的输出。
#### 1.1.1 单词嵌入(Words Embedding)
词嵌入的实现：
1.创建嵌入矩阵 $E$，$E$ 的维度为 $V\times d_{model}$，其中 $V$ 为词汇表大小，$d_{model}$ 为嵌入维度。
2.随机初始化E
3.在训练中学习：通过反向传播算法更新E的值，使得嵌入向量能够更好地表示单词的语义信息。
4.词表与权重共享：
   - 当源语言与目标语言使用共享的子词词表（如通过 BPE/wordpiece 在双语语料上联合训练）时，Encoder 输入嵌入矩阵 $A$、Decoder 输入嵌入矩阵 $B$ 与解码端 pre-softmax 线性层权重矩阵 $C$ 采用统一参数：$A=B$，且 $C=A^{\top}=B^{\top}$。这带来参数效率与跨语言知识迁移。
   - 若源/目标词表无法共享（例如脚本完全不同且未进行子词联合训练），通常保持 Decoder 端的权重 tying：$B$ 与 $C$ 共享（等价于 $C=B^{\top}$），而 Encoder 使用独立的 $A$。
   - 共享仅针对权重矩阵本身，softmax 的偏置 $b_{vocab}$ 通常不与嵌入共享。

5.嵌入查找：对于输入序列中的每个单词，根据其在词汇表中的索引，从嵌入矩阵E中查找对应的嵌入向量，形成输入序列的嵌入表示。
6.数值尺度与正则化：
   - 嵌入缩放：实践中常在与位置编码相加前，对 token 嵌入乘以 \(\sqrt{d_{model}}\) 以匹配数值尺度。
   - Dropout 位置：原论文在“嵌入+位置编码之和”之后施加 Dropout，而不是对嵌入权重矩阵本身做 Dropout。
   - 动机补充：因为初始时词嵌入权重矩阵方差较小，导致嵌入向量的初始值非常小，因此乘以 $\sqrt{d_{model}}$ 有助于平衡词嵌入与幅值在 $[-1,1]$ 的正弦/余弦位置编码的相对尺度，避免相加时词嵌入被位置编码淹没。

#### 1.1.2 位置编码(Position Encoding)
由于Transformer模型没有循环神经网络(RNN)或卷积神经网络(CNN)那样的结构来捕捉序列中的位置信息，因此需要引入位置编码来为模型提供单词在序列中的位置信息。位置编码的实现方法有多种，以下介绍一种常用的方法，即使用正弦和余弦函数生成位置编码。
1.位置编码矩阵的维度为 $(\text{max\_len},\ d_{model})$，其中 $\text{max\_len}$ 是序列的最大长度，$d_{model}$ 是嵌入维度。
2.对于位置 $i$ 和维度 $j$，位置编码的计算公式如下：
   $$
   \mathrm{PE}(i,2j) = \sin\!\left( \frac{i}{10000^{\frac{2j}{d_{model}}}} \right),\quad
   \mathrm{PE}(i,2j+1) = \cos\!\left( \frac{i}{10000^{\frac{2j}{d_{model}}}} \right)
   $$
3.将位置编码矩阵与输入嵌入矩阵相加，得到包含位置信息的输入表示；随后常接 Dropout。

【性质与替代方案】
- 线性可移性：对任何偏移 k，位置 pos+k 的正弦/余弦编码可由位置 pos 的编码通过线性变换得到，便于模型学习相对位置信息。
- 外推性：该编码不依赖数据拟合，理论上能外推到训练未见的更长序列（实际效果也受模型/任务影响）。
- 相对位置编码：在注意力打分中直接注入相对位置信息（例如通过可学习偏置或键/值修正），在许多任务上效果更佳。
- RoPE/旋转位置编码：通过对 Q/K 施加相位旋转把相对位置信息融入内积，兼具外推与相对位置信号。
- ALiBi：对注意力分数添加随距离线性衰减的偏置，简单高效且对长序列有良好外推。

### 1.2 多头自注意力机制(Multi-Head Self-Attention)
多头自注意力机制是Transformer模型的核心组件之一，它能够让模型在处理序列数据时关注不同位置的信息，从而捕捉更丰富的上下文关系。以下是多头自注意力机制的详细解析。
#### 1.2.1 自注意力机制(Self-Attention)
自注意力机制的主要思想是通过计算序列中每个位置与其他位置的相关性来生成新的表示。具体步骤如下：
1.输入表示：假设输入序列的表示为 $X$，维度为 $(\text{seq\_len},\ d_{model})$，其中 $\text{seq\_len}$ 是序列长度，$d_{model}$ 是嵌入维度。
2.线性变换：通过三个不同的线性变换，将输入表示 $X$ 映射到查询（Query）、键（Key）和值（Value）空间，得到 $Q,K,V$ 矩阵：
   $$
   Q = X\, W_Q,\quad K = X\, W_K,\quad V = X\, W_V
   $$
   其中 $W_Q, W_K, W_V$ 是可学习的权重矩阵，维度分别为 $d_{model}\times d_k,\ d_{model}\times d_k,\ d_{model}\times d_v$，$d_k$ 和 $d_v$ 是查询/键和值的维度。
3.计算注意力得分：通过点积计算查询和键之间的相似度：
   $$
		ext{scores} = \frac{Q K^{\top}}{\sqrt{d_k}}
   $$
   其中 $\sqrt{d_k}$ 是缩放因子，通过减小方差使数据更稳定，同时防止接下来的 $\operatorname{softmax}$ 在输入值过大时梯度消失。
   【缩放原因（方差分析）】设 $q,k$ 分量独立同分布且 $\mathbb{E}[q_i]=\mathbb{E}[k_i]=0,\ \operatorname{Var}(q_i)=\operatorname{Var}(k_i)=1$，则
   $$
   \operatorname{Var}(q\cdot k) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] = d_k.
   $$
   因此使用 $\frac{q\cdot k}{\sqrt{d_k}}$ 可将方差拉回到 $1$ 的量级，避免 $\operatorname{softmax}$ 进入饱和区导致梯度消失。
4.$\operatorname{softmax}$ 归一化：
   $$
		ext{attention\_weights} = \operatorname{softmax}(\text{scores})
   $$
5.加权求和：
   $$
		ext{output} = \text{attention\_weights}\, V
   $$

注意力中的两类 Mask（训练与推理常用）：
- Padding Mask：避免模型“注意”到 <pad> 位置，保证表示不受填充影响，适用于 Encoder 与 Decoder 的所有注意力。
- Causal/未来遮蔽：仅用于 Decoder 的自注意力，防止看到当前位置右侧的信息，保障自回归训练/推理的一致性。
（时机：在 $\operatorname{softmax}$ 之前，将被遮蔽位置的分数加上一个非常大的负数，例如 $-\infty$。）
#### 1.2.2 多头注意力机制(Multi-Head Attention)
多头注意力机制通过并行计算多个自注意力头，允许模型在不同的子空间中关注不同的位置，从而捕捉更丰富的上下文信息。具体步骤如下：
1.头的数量：设定多头注意力机制的头数为 $h$。
2.线性变换：对于每个头 $i$，使用独立的线性变换将输入表示 $X$ 映射到查询、键和值空间，得到 $Q_i, K_i, V_i$：
   $$
   Q_i = X\, W_{Q_i},\quad K_i = X\, W_{K_i},\quad V_i = X\, W_{V_i}
   $$
   其中 $W_{Q_i}, W_{K_i}, W_{V_i}$ 是第 $i$ 个头的可学习权重矩阵，维度分别为 $d_{model}\times d_k,\ d_{model}\times d_k,\ d_{model}\times d_v$。
3.计算注意力：对于每个头 $i$，
   $$
		ext{output}_i = \operatorname{Attention}(Q_i, K_i, V_i)
   $$
4.连接头的输出：
   $$
		ext{output} = \operatorname{Concat}(\text{output}_1,\ldots,\text{output}_h)
   $$
5.线性变换：
   $$
		ext{output} = \text{output}\, W_O
   $$
   其中 $W_O$ 是可学习的权重矩阵，维度为 $h\, d_v \times d_{model}$。

#### 1.2.3 为什么多头更好
- 多视角表示：不同头可在不同“表示子空间”中学习不同关系（语法、语义、邻近等），信息更丰富、鲁棒性更好。
- 低秩限制的缓解：单头点积注意力本质上是受限的双线性形式，多头相当于并行若干低维注意力再汇合，提升表达力。
- 计算平衡：将 $d_{model}$ 划分到 $h$ 个头，每头维度 $d_k = d_{model}/h$，可在不增大总体计算的情况下增加头数。
   大致成本对比：$h$ 个 $d_k$ 维注意力的矩阵乘法总 FLOPs 约等于一个 $d_{model}$ 维的注意力（因为 $h\cdot d_k = d_{model}$）。

#### 1.2.4 为什么每个头要降维
- 若不降维，$d_{model}$ 直接用于每个头，$h$ 个头的计算/参数开销将线性放大 $h$ 倍。
- 令 $d_k=d_v=d_{model}/h$，则总体计算与单头 $d_{model}$ 基本同量级，同时保留多视角优势。
- 示例：$d_{model}=512,\ h=8\Rightarrow d_k=d_v=64$。

#### 1.2.5 为什么 Q/K/V 需要独立投影
若共享投影或直接使用输入 $x$，得分退化为 $x_i\cdot x_j$，仅衡量同空间相似度；独立的 $W_Q, W_K$ 使得 $(x_i W_Q)\cdot(x_j W_K)$ 可学习非对称、上下文依赖的匹配，更灵活地捕捉复杂关系。

#### 1.2.6 点乘注意力与加法注意力对比
点乘注意力（Transformer 采用）：
$$
\operatorname{Attention}(Q,K) = \operatorname{softmax}\!\left( \frac{QK^{\top}}{\sqrt{d_k}} \right).
$$
加法注意力（Bahdanau）：
$$
e(q,k) = v^{\top} \tanh(W_q q + W_k k),\quad \alpha = \operatorname{softmax}(e).
$$
两者在小维度下性能相近；点乘可充分利用矩阵乘法并行且显存友好。对大 $d_k$，未缩放的点乘会退化，加入 $1/\sqrt{d_k}$ 后训练更稳定。
### 1.3 归一化和残差连接(Normalization and Residual Connection)
为了提高模型的训练稳定性和收敛速度，Transformer模型在每个子层之后都使用了归一化和残差连接。具体步骤如下：
1.残差连接：将子层的输入与子层的输出相加，形成残差连接：
   $$
		ext{output} = \text{LayerInput} + \text{SubLayerOutput}
   $$
2.归一化：对残差连接的结果进行归一化处理，通常使用层归一化（Layer Normalization）：
   $$
		ext{output} = \mathrm{LayerNorm}(\text{output})
   $$

为什么用 LayerNorm 而不是 BatchNorm？
- BN 依赖 batch 统计量，受 batch size 和 padding 影响显著；NLP 中批内长度不一致常需 padding，统计量容易偏移。
- LN 在每个样本的特征维度上归一化，与 batch 无关，面对变长序列更稳定，也更适合小批量甚至单样本推理。

【进一步说明：Pre-LN vs Post-LN】
- 原论文采用 Post-LN（SubLayer -> Add 残差 -> LayerNorm），深层时可能梯度传播不够稳定，需较强的 warmup。
- 许多后续实现采用 Pre-LN（先 LayerNorm 再进入子层，输出与残差相加），更利于深层稳定训练与收敛，一般也更易与较大学习率配合。
### 1.4 前馈神经网络(Feed-Forward Neural Network)
前馈神经网络是Transformer模型中的另一个重要组件，它位于每个Encoder层的自注意力机制之后。前馈神经网络的主要作用是对每个位置的表示进行非线性变换，并且通过提高特征维度使模型提取到更丰富的特征，从而增强模型的表达能力。具体步骤如下：
1.输入表示：假设前馈神经网络的输入表示为 $X$，维度为 $(\text{seq\_len},\ d_{model})$。
2.激活函数：在第一个线性变换之后，通常会使用ReLU激活函数引入非线性，从而增强模型的表达能力。
3.线性变换：通过两个线性变换将输入表示映射到一个更高维度的空间，然后再映射回原始的嵌入维度：
   $$
      ext{hidden} = \operatorname{ReLU}(X\, W_1 + b_1),\quad
      ext{output} = \text{hidden}\, W_2 + b_2
   $$
其中 $W_1, W_2$ 是可学习的权重矩阵，$b_1, b_2$ 是偏置项，维度分别为 $d_{model}\times d_{ff}$、$d_{ff}\times d_{model}$，且通常 $d_{ff} \approx 4\, d_{model}$。
4.归一化和残差连接：与自注意力机制类似，前馈神经网络的输出也会经过残差连接和层归一化处理：
   $$
      ext{output} = \mathrm{LayerNorm}(X + \text{output})
   $$

【激活函数选择】
- 原版使用 $\operatorname{ReLU}$，实现简单但可能出现“Dying ReLU”。
- 现代变体常用 $\operatorname{GELU}$ 或 $\operatorname{SwiGLU}$ 等更平滑/门控的激活，通常表现更佳（如 BERT/GPT 使用 GELU）。
## 2.Decoder 的结构
一个Decoder包含N个Decoder层，以下以一个Decoder层为例解析结构。
### 2.1 输入
一层Decoder的输入可以分为两类，第一类是第一个Decoder层的输入，其由目标单词的嵌入(见2.1.1)和位置的编码(见2.1.2)之和组成；第二类是其他Decoder层的输入，即上一层的输出。此外，Decoder还需要Encoder的输出作为辅助信息。
#### 2.1.1 目标单词嵌入(Target Words Embedding)
目标单词嵌入的实现与输入单词嵌入类似，具体步骤如下：
1.创建嵌入矩阵 $E_t$，$E_t$ 的维度为 $V_t\times d_{model}$，其中 $V_t$ 为目标词汇表大小，$d_{model}$ 为嵌入维度。
2.随机初始化E_t
3.在训练中学习：通过反向传播算法更新E_t的值，使得嵌入向量能够更好地表示目标单词的语义信息。
4.权重共享：
   - 若采用与源语言联合训练的共享子词词表，Decoder 输入嵌入 $B$ 与 Encoder 输入嵌入 $A$ 共享，并与输出线性层权重 $C$ 进行 tying，满足 $A = B$，且 $C = A^{\top} = B^{\top}$。
   - 若不共享词表，则保持 Decoder 端的权重 tying：$B$ 与 $C$ 共享（$C = B^{\top}$），而 Encoder 端使用独立的 $A$。
   - 共享同样不包含 softmax 偏置 $b_{vocab}$，偏置通常单独学习。
5.嵌入查找：对于目标序列中的每个单词，根据其在目标词汇表中的索引，从嵌入矩阵E_t中查找对应的嵌入向量，形成目标序列的嵌入表示。
6.正则化：为了防止过拟合，可以对嵌入矩阵E_t进行正则化处理，例如使用dropout技术随机丢弃部分嵌入向量。
#### 2.1.2 位置编码(Position Encoding)
位置编码的实现方法与输入序列的位置编码类似，具体步骤如下：
1.位置编码矩阵的维度为 $(\text{max\_len},\ d_{model})$，其中 $\text{max\_len}$ 是目标序列的最大长度，$d_{model}$ 是嵌入维度。
2.对于位置 $i$ 和维度 $j$，位置编码的计算公式如下：
   $$
   \mathrm{PE}(i,2j) = \sin\!\left( \frac{i}{10000^{\frac{2j}{d_{model}}}} \right),\quad
   \mathrm{PE}(i,2j+1) = \cos\!\left( \frac{i}{10000^{\frac{2j}{d_{model}}}} \right)
   $$
3.将位置编码矩阵与目标输入嵌入矩阵相加，得到包含位置信息的目标输入表示。
（同样地，此处通常也会在相加后施加 Dropout。）
### 2.2 掩码多头自注意力机制(Masked Multi-Head Self-Attention)
掩码多头自注意力机制与多头自注意力机制类似，但在计算注意力得分时引入了掩码(Mask)操作，以防止模型在生成目标序列时看到未来的信息。具体步骤如下：
1.输入表示：假设目标输入序列的表示为 $Y$，维度为 $(\text{seq\_len},\ d_{model})$。
2.线性变换：
   $$
   Q = Y\, W_Q,\quad K = Y\, W_K,\quad V = Y\, W_V
   $$
   其中 $W_Q, W_K, W_V$ 维度同上，为 $d_{model}\times d_k,\ d_{model}\times d_k,\ d_{model}\times d_v$。
3.计算注意力得分：
   $$
		ext{scores} = \frac{Q K^{\top}}{\sqrt{d_k}}
   $$
4.掩码操作：为了防止模型在生成目标序列时看到未来的信息，对注意力得分矩阵进行掩码。设掩码矩阵 $M\in\mathbb{R}^{\text{seq\_len}\times \text{seq\_len}}$，定义：
   $$
   M_{ij} =
   \begin{cases}
   0, & j\le i,\\
   -\infty, & j>i.
   \end{cases}
   $$
   具体加法为：
   $$
		ext{scores} \leftarrow \text{scores} + M
   $$
5.$\operatorname{softmax}$ 归一化：
   $$
		ext{attention\_weights} = \operatorname{softmax}(\text{scores})
   $$
6.加权求和：
   $$
		ext{output} = \text{attention\_weights}\, V
   $$
### 2.3 编码器-解码器注意力机制(Encoder-Decoder Attention)
编码器-解码器注意力机制允许Decoder在生成目标序列时关注输入序列的相关信息，从而更好地捕捉输入和输出之间的关系。具体步骤如下：
1.输入表示：假设 Decoder 的输入表示为 $Y$，维度为 $(\text{seq\_len}_t,\ d_{model})$；Encoder 的输出为 $Z$，维度为 $(\text{seq\_len}_s,\ d_{model})$。
2.线性变换：
   $$
   Q = Y\, W_Q,\quad K = Z\, W_K,\quad V = Z\, W_V
   $$
   其中 $W_Q, W_K, W_V$ 维度为 $d_{model}\times d_k,\ d_{model}\times d_k,\ d_{model}\times d_v$。
3.计算注意力得分：
   $$
		ext{scores} = \frac{Q K^{\top}}{\sqrt{d_k}}
   $$
4.$\operatorname{softmax}$ 归一化：
   $$
		ext{attention\_weights} = \operatorname{softmax}(\text{scores})
   $$
5.加权求和：
   $$
		ext{output} = \text{attention\_weights}\, V
   $$
直观理解：这一步好比翻译时每生成一个词，都会回看整句原文（Encoder 的 $K,V$），基于当前生成状态的查询 $Q$ 找到最相关的源片段并汲取信息。
### 2.4 归一化和残差连接(Normalization and Residual Connection)
归一化和残差连接的实现与Encoder中的类似，具体步骤如下：
1.残差连接：将子层的输入与子层的输出相加，形成残差连接：
   $$
		ext{output} = \text{LayerInput} + \text{SubLayerOutput}
   $$
2.归一化：对残差连接的结果进行归一化处理，通常使用层归一化（Layer Normalization）：
   $$
		ext{output} = \mathrm{LayerNorm}(\text{output})
   $$
### 2.5 前馈神经网络(Feed-Forward Neural Network)
前馈神经网络的实现与Encoder中的类似，具体步骤如下：
1.输入表示：假设前馈神经网络的输入表示为 $X$，维度为 $(\text{seq\_len},\ d_{model})$。
2.激活函数：在第一个线性变换之后，通常会使用ReLU激活函数引入非线性，从而增强模型的表达能力。
3.线性变换：通过两个线性变换将输入表示映射到一个更高维度的空间，然后再映射回原始的嵌入维度：
   $$
      ext{hidden} = \operatorname{ReLU}(X\, W_1 + b_1),\quad
   ext{output} = \text{hidden}\, W_2 + b_2
   $$
其中 $W_1, W_2$ 是可学习的权重矩阵，$b_1, b_2$ 是偏置项，维度分别为 $d_{model}\times d_{ff}$、$d_{ff}\times d_{model}$，且通常 $d_{ff} \approx 4\, d_{model}$。
4.归一化和残差连接：与自注意力机制类似，前馈神经网络的输出也会经过残差连接和层归一化处理：
   $$
	ext{output} = \mathrm{LayerNorm}(X + \text{output})
   $$

### 2.6 训练与推理的差异（Teacher Forcing vs 自回归生成）
- 并行化体现：自注意力的大矩阵乘法允许对整段序列并行；Encoder 完全并行；Decoder 训练可并行，推理严格自回归（串行），可借助 KV Cache 降低重复计算。
- 训练阶段（并行）：Decoder 输入为“右移一位”的真实目标序列（teacher forcing），并用 Causal Mask 确保位置 $i$ 只能看到 $\le i$ 的信息；可对整段序列并行计算，大幅提升训练吞吐。
- 推理阶段（串行自回归）：以 <SOS> 开始，逐步生成一个 token 并将其回馈到下一步，直至生成 <EOS> 或达到最大长度；该过程天然串行，速度慢于训练。
- KV Cache：推理时缓存过往步骤的 K/V，下一步仅对新 token 做投影并与缓存拼接，避免重复计算，显著提速与降显存。
- 解码策略：
   - Greedy/Beam Search：贪心或小束搜索，适合翻译等确定性较强任务。
   - 采样：Top-k / Top-p（nucleus）与温度 Temperature，适合开放式生成，平衡多样性与连贯性。
   - 早停：遇到 <EOS> 或无改进时停止，或设置长度惩罚等启发式。

### 2.7 信息泄露与残差连接（不会发生）
Decoder 的残差是按位置（position-wise）进行的：位置 $i$ 的残差仅为 $x_i + \text{SubLayer}(x)_i$。由于掩码自注意力确保 $\text{SubLayer}(x)_i$ 只依赖于 $\le i$ 的信息，残差不会引入 $j>i$ 的未来信息，也就不存在泄露。
## 3.Linear and Softmax
在Decoder的最后一层之后，通常会添加一个线性变换层和一个softmax层，用于将Decoder的输出映射到目标词汇表的概率分布。具体步骤如下：
1.线性变换：通过一个线性变换将 Decoder 的输出表示映射到目标词汇表的维度：
   $$
		ext{logits} = \text{output}\, W_{vocab} + b_{vocab}
   $$
其中 $W_{vocab}$ 是可学习的权重矩阵，具体与嵌入共享关系如下：
   - 若源/目标使用共享词表（BPE 等联合训练），设统一嵌入矩阵为 $E$（$A=B=E$，维度为 $V_{shared}\times d_{model}$），则 $W_{vocab} = E^{\top}$（$C = A^{\top} = B^{\top}$），其维度为 $(d_{model},\ V_{shared})$。
   - 若不共享词表，设 Decoder 端嵌入为 $E_t$（$B=E_t$，维度为 $V_t\times d_{model}$），则 $W_{vocab} = E_t^{\top}$（$C = B^{\top}$），其维度为 $(d_{model},\ V_t)$。
   - 偏置 $b_{vocab}$ 通常不与嵌入共享，独立学习，维度与词表大小一致。

2.$\operatorname{softmax}$ 归一化：
   $$
		ext{probabilities} = \operatorname{softmax}(\text{logits})
   $$
3.预测目标单词：在训练过程中，通常会使用交叉熵损失函数来计算预测的概率分布与真实目标单词之间的差异，从而指导模型的学习。在推理过程中，可以通过选择概率最高的单词作为预测结果，或者使用采样方法从概率分布中选择单词。

## 4. 训练技巧与正则化（Practical Tips）
- Dropout：
   - 施加位置：嵌入+位置编码之和后、各子层（注意力/FFN）输出与残差相加前；论文中典型值 p≈0.1。
   - 注意：一般不对嵌入权重矩阵做 Dropout，而是对激活（token 表示）做。
   - 推理/测试：推理与评估阶段需关闭 Dropout（例如将模型置于 eval 模式），以使用完整表示并获得确定性输出。
- 标签平滑（Label Smoothing）：
   - 将 one-hot 的 1 缓和为 1-ε（例如 ε=0.1），其余类分到 ε/(V-1)。
   - 益处：降低过度自信与过拟合、提升泛化与校准（calibration）。
- 学习率与 Warmup：
   - Noam 调度：\(lr = d_{model}^{-0.5} \cdot \min(step^{-0.5},\ step\cdot warmup^{-1.5})\)，常见 warmup 步数约 4k～8k。
   - Adam 超参：β1=0.9，β2=0.98，ε=1e-9（原论文配置）。
- 梯度裁剪：
   - 使用全局范数裁剪（如 1.0）缓解梯度爆炸，提升训练稳定性。
- Mask 细节：
   - 训练与推理均需严格应用 Padding Mask；Decoder 自注意力还需 Causal Mask，确保与自回归目标一致。