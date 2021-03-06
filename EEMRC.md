# Event Extraction by Answering (Almost) Natural Questions(EMNLP 2020)

## 1 框架概述

![image-20210521195811377](./picture/image-20210521195811377.png)

- BERT_QA_Trigger	识别触发词

sentence 将被编码为[CLS] <question> [SEP] <sentence> [SEP]

- BERT_QA_Arg	抽取触发词和事件元素
- 使用动态阈值来淘汰对应多余的事件元素

## 触发词检测问题生成

​	使用固定短语 action

[CLS] action [SEP] As part of the 11-billion-dollar sale ... [SEP]

## 对于论元问题生成

- 1 元素角色名(e.g., artifact, agent, place)

- 2 类别加角色名(WH+rolename)

- 3 利用ACE注释中每个参数角色描述来构建问题（更贴近自然）

  ![image-20210521201456867](./picture/image-20210521201456867.png)

- 最后在问题中加入触发词检测所抽取的标记`<trigger>`

  <WH_word> is the <argument> in <trigger>?

## QA Models

使用bert作为基本模型来得到输入语句作为QA_Trigger和QA_arg的输入序列的上下文向量表示

[CLS] <question> [SEP] <sentence>[SEP]

对于输入序列 $(e_1，e_2...,e_n)$ 以及对于准备抽取元素的输入序列$(a_1,a_2...a_m)$



![image-20210521203700111](./picture/image-20210521203700111.png)

![image-20210521203728090](./picture/image-20210521203728090.png)

我们使用$E$ 和$ A$ 表示结果 $E \in R^{N*H} A \in R^{M*H} $

在输出层，利用不同的解码策略，BERT_QA_Trigger 预测事件类型，而BERT_QA_arg预测每个元素的开始和结束位置偏移 

### 事件触发词检测

引用$ W_{tr} \in R^{H*T}$ $T$是事件类型数+1(额外一个表示不存在)

$P_{tr}=softmax(EW_{tr})\in R^{N*T}$

$P_{tr}$可以认为是每个词是哪种事件类型的预测概率矩阵，我们简单地取整个矩阵最大值为触发词和事件类型。

### 论元检测

$W_s、W_e \in R^H*1$ 分别为起始，结束的权重矩阵

$P_s=softmax(AW_s) \quad P_e=softmax(AW_e)$

对于对应论元无回答的句子，我们将使得第一个词（[cls]）的起始预测和结束的概率最大



![image-20210530222358561](./picture/image-20210530222358561.png)



![image-20210530222431424](./picture/image-20210530222431424.png)

## 论文实现

![image-20210531204013968](./picture/image-20210531204013968.png)

![image-20210531210238715](./picture/image-20210531210238715.png)

# 复现进度 

![image-20210531203405640](./picture/image-20210531203405640.png)

## 可能的改进思路

如其实验所示，其trigger 抽取的损失会进一步影响到后续抽取，提升trigger至关重要。

- 可以考虑在第一步trigger抽取的基础上，对论元采取不考虑事件类型的全抽取，以不分类型的全抽取结果置信度反向影响trigger结果的选取，

- 全抽取可能失去了事件类型信息而导致信息丢失 且进行了大量不必要的工作，故我们可以取trigger结果的前n个结果来进行对于事件论元抽取，再依据论元抽取结果接着选取最终的trigger结果

  <WH_word> is the <argument> in <trigger>

- 转移到中文上（官方是字编码、全词编码（使用BERT-wwm））。

