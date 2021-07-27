<p>
    <center><h1>隐马尔可夫</h1></center>
	<br />
    <p name="top" id="top" align="center">
        <b>作者：</b><b><a href="https://www.cnblogs.com/dan-baishucaizi/">elfin</a></b>&nbsp;&nbsp;
        <b>参考资料来源：<a href="">《统计学习方法》第二版</a></b>
	</p>
</p>

---

**目录**：

* [一、前向概率计算](#一、前向概率计算)
* [二、后向概率计算](#二、后向概率计算)
* [三、给定模型与观测，在时刻t处于状态qi的概率](#三、给定模型与观测，在时刻t处于状态qi的概率)
* [四、给定模型与观测，t时刻处于qi，下一时刻处于qj的概率](#四、给定模型与观测，t时刻处于qi，下一时刻处于qj的概率)
* [五、一些重点期望值](#五、一些重点期望值)
    * [5.1 在观测O下，状态i出现的期望值](#5.1 在观测O下，状态i出现的期望值)
    * [5.2 在观测O下，由状态i转移的期望](#5.2 在观测O下，由状态i转移的期望)
    * [5.3 在观测O下，由状态i转移到j的期望](#5.3 在观测O下，由状态i转移到j的期望)
* [六、模型学习](#六、模型学习)



隐马尔可夫：前向后向概率计算、模型参数学习、模型预测

---

<p align="right">
    <b><a href="#top">Top</a></b>&nbsp;<b>---</b>&nbsp;<b><a href="#bottom">Bottom</a></b>
</p>



## 一、前向概率计算

以下A是状态概率转移矩阵，B是观测概率矩阵，hmm是一个模型的实例化。

```python
A = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])
B = np.array([
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
])
result = hmm.get_forward_prob([6, 3, 1, 2, 4, 2])
hmm = HMM(A, B, [0.2, 0.4, 0.4])
result = hmm.get_forward_prob([1, 2, 1])
```

---

<p align="right">
    <b><a href="#top">Top</a></b>&nbsp;<b>---</b>&nbsp;<b><a href="#bottom">Bottom</a></b>
</p>



## 二、后向概率计算

```python
A = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])
B = np.array([
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
])
result = hmm.get_forward_prob([6, 3, 1, 2, 4, 2])
hmm = HMM(A, B, [0.2, 0.4, 0.4])
result = hmm.get_backward_prob([1, 2, 1])
```

这里前向、后向概率是一致的！

---

<p align="right">
    <b><a href="#top">Top</a></b>&nbsp;<b>---</b>&nbsp;<b><a href="#bottom">Bottom</a></b>
</p>

## 三、给定模型与观测，在时刻t处于状态qi的概率

<img src="http://latex.codecogs.com/gif.latex?\gamma_{t}\left ( i \right )=\frac{\alpha_{t}(i)\beta_{t}(i)}{\sum_{j=1}^{N}\alpha_{t}(j)\beta_{t}(j)}">

[](http://latex.codecogs.com/gif.latex?\gamma_{t}\left ( i \right )=\frac{\alpha_{t}(i)\beta_{t}(i)}{\sum_{j=1}^{N}\alpha_{t}(j)\beta_{t}(j)})




```python
A = np.array([
    [0, 1, 0],
    [0.2, 0.35, 0.45],
    [0.4, 0.14, 0.46]
])
B = np.array([
    [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
    [0.23, 0.2, 0.175, 0.14, 0.135, 0.12],
    [0.24, 0.2, 0.175, 0.13, 0.135, 0.12]
])
hmm = HMM(A, B, [1 / 3, 1 / 3, 1 / 3])
result = hmm.get_qi2t_prob([6, 3, 1, 2, 4, 2], 3, 2)
```



---

<p align="right">
    <b><a href="#top">Top</a></b>&nbsp;<b>---</b>&nbsp;<b><a href="#bottom">Bottom</a></b>
</p>



## 四、给定模型与观测，t时刻处于qi，下一时刻处于qj的概率

$$
\xi _{t}\left ( i,j \right )=P\left ( i_{t}=q_{i},i_{t+1}=q_{j}|O,\lambda  \right )=\frac{P\left ( i_{t}=q_{i},i_{t+1}=q_{j},O|\lambda  \right )}{P\left ( O|\lambda  \right )}\\
P\left ( i_{t}=q_{i},i_{t+1}=q_{j},O|\lambda  \right )=\alpha_{t}(i)a_{ij}b_{j}\left( o_{t+1} \right)\beta_{t+1}(j)\\
\alpha_{t}(i)\beta_{t}(i)=P\left ( i_{t}=q_{i},O|\lambda  \right )
$$



```python
A = np.array([
    [0, 1, 0],
    [0.2, 0.35, 0.45],
    [0.4, 0.14, 0.46]
])
B = np.array([
    [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
    [0.23, 0.2, 0.175, 0.14, 0.135, 0.12],
    [0.24, 0.2, 0.175, 0.13, 0.135, 0.12]
])
hmm = HMM(A, B, [1 / 3, 1 / 3, 1 / 3])
result = hmm.get_qi2t_qj2next_prob([6, 3, 1, 2, 4, 2], 3, 1, 2)
```



---

<p align="right">
    <b><a href="#top">Top</a></b>&nbsp;<b>---</b>&nbsp;<b><a href="#bottom">Bottom</a></b>
</p>



## 五、一些重点期望值

### 5.1 在观测O下，状态i出现的期望值

$$
1 - \sum_{t=1}^{T} \left(1 -\gamma_{t}\left ( i \right ) \right )
$$



---

<p align="right">
    <b><a href="#top">Top</a></b>&nbsp;<b>---</b>&nbsp;<b><a href="#bottom">Bottom</a></b>
</p>

### 5.2 在观测O下，由状态i转移的期望

$$
1 - \sum_{t=1}^{T-1} \left(1 -\gamma_{t}\left ( i \right ) \right )
$$



---

<p align="right">
    <b><a href="#top">Top</a></b>&nbsp;<b>---</b>&nbsp;<b><a href="#bottom">Bottom</a></b>
</p>

### 5.3 在观测O下，由状态i转移到j的期望

$$
1 - \sum_{t=1}^{T-1} \left(1 -\xi _{t}\left ( i,j \right ) \right )

$$



---

<p align="right">
    <b><a href="#top">Top</a></b>&nbsp;<b>---</b>&nbsp;<b><a href="#bottom">Bottom</a></b>
</p>



## 六、模型学习

关于模型的学习的理论知识见[《HMM隐马尔可夫.md》](HMM隐马尔可夫.md) 。

关于EM算法，我们将E步M步整合到`update_param`方法中，你只需要控制迭代停止的条件。







---

<p align="right">
    <b><a href="#top">Top</a></b>&nbsp;<b>---</b>&nbsp;<b><a href="#bottom">Bottom</a></b>
</p>

<p align="right">
    <b><a href="#top">Top</a></b>&nbsp;<b>---</b>&nbsp;<b><a href="#bottom">Bottom</a></b>
</p>
<p id="bottom" name="bottom">
	<b>完！</b>
</p>
