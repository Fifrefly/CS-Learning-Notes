# Transformer 核心概念与结构
## 1.Encoder 的结构
一个Encoder包含N个Encoder层，以下以一个Encoder层为例解析结构。
### 1.1 输入
一层Encoder的输入可以分为两类，第一类是第一个Encoder层的输入，其由单词的嵌入(见1.1.1)和位置的编码(见1.1.2)之和组成；第二类是其他Encoder层的输入，即上一层的输出。
#### 1.1.1 单词嵌入(Words Embedding)

#### 1.1.2 位置编码(Position Encoding)