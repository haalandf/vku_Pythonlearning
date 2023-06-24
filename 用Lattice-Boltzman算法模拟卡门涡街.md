<center>
    <font size="6">
        <b>用Lattice-Boltzmann算法模拟卡门涡街</b>
    </font>
    <br>
    <font size="4">
        管理学院 PB22151726 方浩然
    </font>
</center>
<hr>

## 0.简介
&emsp;&emsp;卡门涡街是由西奥多·冯·卡门的学生在实验中发现、冯·卡门在理论上研究阐明的。这是圆柱绕流的一种特殊情况（雷诺数$Re:47 < Re < 10^{5}$）,也是计算流体力学（Computational Fluid Dynamics）中常见而基础的模拟项目。当然，目前主流的CFD模拟软件包括Fluent、cfx等等，而python由于计算能力的限制并不是十分适用于CFD模拟，但在限制时间范围和精度之后，也能够在python中实现简单的CFD模拟，其核心算法即本文所采用的Lattice-Boltzmann算法。本文构造了一个512*32的网格，并在网格上定义流场以及物理碰撞过程，最终生成MP4格式视频，成功实现了卡门涡街的模拟。

## 1.科学原理介绍

### 1.1 Navier-Stokes方程
&emsp;&emsp;对流场内体积为 $\delta V$ ，速度为 $\boldsymbol{u}$ 的流体体元用牛顿第二定律有
$$\displaystyle
(\rho \delta V)\frac{{\rm D}\boldsymbol{u}}{{\rm D}t}=-(\nabla p)\delta V + viscousForces$$

&emsp;&emsp;这个公式意味着流体元的质量 $\rho \delta V$ 乘以加速度 $\frac{{\rm D}\boldsymbol{u}}{{\rm D}t}$ 等于作用在流体元上的净压力加上因为粘滞压带来的粘滞力，净压力写作 $-(\nabla\rho)\delta V$ 是由Gauss定理来的：
$$\displaystyle
\oint_{S}(-p){\rm d}\boldsymbol{S}=\int_{\delta V}(-\nabla p){\rm d}V=-(\nabla p)\delta V$$
左边的面积分是作用在面 $S$ 上所有基本压力的总和.
&emsp;&emsp;接下来估计公式中的粘滞项，对流体体元所受应力和剪切力分析，可以得到牛顿粘滞性定律：
$$\displaystyle
\tau_{ij}=2\rho vS_{ij}$$
其中， $S_{ij}$ 是应变率张量 $\frac{1}{2}[\frac{\partial u_i}{\partial x_j}+\frac{\partial u_j}{\partial x_i}]$.
&emsp;&emsp;把牛顿粘滞性定律跟流体形式的牛顿定律写在一起，经过计算可以得到<b>Navier-Stokes方程</b>：
$$\displaystyle
\frac{{\rm D}\boldsymbol{u}}{{\rm D}t}=-\nabla(\frac{p}{\rho})+v\nabla^{2}\boldsymbol{u}$$
其中，关于 $\boldsymbol{u}$ 的边界条件是 $\boldsymbol{u}=0$.
<br>
在矢量场 $\boldsymbol{A}(x,t)$ 中，根据链式法则可以得到
$$\displaystyle
\frac{{\rm D}\boldsymbol{A}}{{\rm D}t}=\frac{\partial \boldsymbol{A}}{\partial t}+(\boldsymbol{u}·\nabla)\boldsymbol{A}$$
因此，我们可以把Navier-Stokes方程改写为
$$\displaystyle
\frac{{\rm D}\boldsymbol{u}}{{\rm D}t}=\frac{\partial \boldsymbol{u}}{\partial t}+(\boldsymbol{u}·\nabla)\boldsymbol{u}=-\nabla(\frac{p}{\rho})+v\nabla^{2}\boldsymbol{u}
$$
需要注意到，上式包含 $\boldsymbol{u}$ 的<b>非线性项 $(\boldsymbol{u}·\nabla)\boldsymbol{u}$ </b>，正是这一项带来了复杂而丰富的流体力学现象。
<br>
&emsp;&emsp;接下来我们讨论流场的旋度 $\boldsymbol{\omega}$ ，改写上式再对改写后的NS方程取旋度，就得到了 $\boldsymbol{\omega}$ 的演化方程：
$$\displaystyle
\frac{\partial \boldsymbol{\omega}}{\partial t}=\nabla\times[\boldsymbol{u}\times\boldsymbol{\omega}]+v\nabla^{2}\boldsymbol{\omega}$$
因为 $\nabla\times(\boldsymbol{u}\times\boldsymbol{\omega})=(\boldsymbol{\omega}·\nabla)\boldsymbol{u}-(\boldsymbol{u}·\nabla)\boldsymbol{\omega}$ ，所以将上式改写为对流导数的形式：
$$\displaystyle
\frac{{\rm D}\boldsymbol{\omega}}{{\rm D}t}=(\boldsymbol{\omega}·\nabla)\boldsymbol{u}+v\nabla^{2}\boldsymbol{\omega}$$

&emsp;&emsp;在二维运动上， $\boldsymbol{u}(x,y)=(u_x,u_y,0),\boldsymbol{\omega}=(0,0,\omega)$ ,上式右侧的第一项 $(\boldsymbol{\omega}·\nabla)\boldsymbol{u}$ 退化为 $0$ ，所以
$$\displaystyle
\frac{{\rm D}\boldsymbol{\omega}}{{\rm D}t}=v\nabla^{2}\boldsymbol{\omega}$$

&emsp;&emsp;该式说明平面流场没有了涡流拉伸，流体元的涡量只会因为粘滞力而改变。而且在二维流场中，涡度是对流，不能凭空产生或毁灭，并且它可以通过对流从一个地方到另一个地方，但是 $\int\omega{\rm d}V$ 对于任何地方的涡度团都是守恒的，在圆柱绕流中，流场从圆柱的表面获得了涡度（角动量），并通过速度场对流扩散。从而产生了<b>卡门涡街（Karman vortex street）</b>现象。

![figure1.1](/理论示意图.jpg)
<center>
    <font size="1">
    图1.1&emsp;卡门涡街的理论示意图（图片来自网络）
    </font>
</center>

### 1.2 Lattice-Boltzmann方法
&emsp;&emsp;假设流体是理想气体，且宏观速度 $\boldsymbol{u}=0$ ，在温度为 $T$ 时处于热平衡状态，那么分子的热运动满足统计力学的玻尔兹曼分布，对于二维气体来说，满足以下分布函数：
$$\displaystyle
D(\vec{v})=\frac{m}{2\pi kT}e^{-m|\vec{v}|^{2}/2kT}$$

其中， $m$ 表示分子质量， $k$ 为玻尔兹曼常数。玻尔兹曼分布函数在分量 $v_x$ 和 $v_y$ 上积分时，给出一个特定分子在一个速率范围的可能性（注意归一化）。

&emsp;&emsp;在lattice-Boltzman方法中，我们把时间和空间离散化，在每个格子中处理离散的速度矢量，也就是D2Q9格子，如图1.2所示。

![figure1.2](D2Q9lattice.png)
<center>
    <font size="1">
    图1.2&emsp;D2Q9网格示意图
    </font>
</center>

&emsp;&emsp;接下来，我们需要把概率附加到这九个速度矢量上，尽可能精确地模拟连续Boltzmann分布。根据玻尔兹曼分布函数，最优的概率分布是 $\omega_{0}=\frac{4}{9} , \omega_{1}=\omega_{2}=\omega_{3}=\omega_{4}=\frac{1}{9} , \omega_{5}= \omega_{6}=\omega_{7}=\omega_{8}=\frac{1}{36}$. 

&emsp;&emsp;这些权重具有明确的定性属性，且它们满足归一化。同时，他们也能够预测相同时刻下 $v_x$ 和 $v_y$ 的值，以及 $v_x$ 和 $v_y$ 的次方（直到四次方）。

&emsp;&emsp;在用lattice-Boltzmann方法进行模拟时，最基础的动态变量是每个格子的9个方向密度，因此，我们需要构建9个二维数组去表示不同的方向密度，代码实现如1.3所示。

```python
height = 32     # 网格高度
width = 512     # 网格长度
# 微观晶格方向密度
n0 = numpy.zeros((height, width))
nN = numpy.zeros((height, width))
nS = numpy.zeros((height, width))
nE = numpy.zeros((height, width))
nW = numpy.zeros((height, width))
nNW = numpy.zeros((height, width))
nNE = numpy.zeros((height, width))
nSE = numpy.zeros((height, width))
nSW = numpy.zeros((height, width))
```

<center>
    <font size="1">
    代码1.3&emsp;定义微观晶格方向密度
    </font>
</center>

&emsp;&emsp;重要的是在上面所构建的网格内定义流场，以及物理碰撞过程。根据 $\omega_i$ 的性质，有
$$\displaystyle
\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}v_{x}^{2}D(\vec{v}){\rm d}v_x{\rm d}v_y=\sum_{i=0}^{8}(e_{i,x}·c)^{2}\omega_{i}, etc.$$

&emsp;&emsp;左式是根据玻尔兹曼分布计算得到的 $v_{x}^{2}$ 的均值，而右式则是经过格子方法离散化得到的相同的平均值。其中， $\omega_i$ 应取到合适的值（即上文所列），所以常数 $c$ 必须只和温度有关：
$$\displaystyle
c^2=\frac{3kT}{m}$$

&emsp;&emsp;上述讨论同样适用于静止的流体。而对于宏观速度不为0的流体，每个分子的合速度就是宏观速度 $\vec{u}$ 加上热速度 $\vec{v}$ ：
$$\displaystyle
\vec{e_i}·c=\vec{u}+\vec{v}$$

而对于热速度分布的玻尔兹曼函数不受宏观速度 $\vec{u}$ 的影响，所以
$$\displaystyle
D(\vec{v})\rightarrow\frac{m}{2\pi kT}\exp(-\frac{m}{2kT}|\vec{e_i}c-\vec{u}|^{2})$$

再对上式作泰勒展开，得到
$$\displaystyle
D(\vec{v})\rightarrow\omega_{i}\bigg[ 1+\frac{3\vec{e_i}·\vec{u}}{c}+\frac{9}{2}(\frac{\vec{e_i}·\vec{u}}{c})^{2}-\frac{3}{2}\frac{|\vec{u}|^{2}}{c^{2}}\bigg].$$

&emsp;&emsp;在任意一个给定的时刻，每个格子的9个方向密度会有任意的正值。根据这九个参量，我们可以计算总的密度 $\rho$ ,以及平均（宏观）速度在x和y方向上的分量 $u_x$ 和 $u_y$ 。再根据这三个宏观量，我们可以求出热平衡状态下，每个格子微观的九个方向密度：
$$\displaystyle
n^{eq}=\rho \omega_{i}[1+3\vec{e_i}·\vec{u}+\frac{9}{2}(\vec{e_i}·\vec{u})^{2}-\frac{3}{2}|u|^{2}].$$

这个式子和概率的表达式相同，只需要乘上总的密度 $\rho$ 并取 $c=1$ 。如果把所有的9个方向密度设为这些平衡值，可以模拟分子间的碰撞过程，从而使它们更接近热平衡的状态。但是达到平衡的时间不一定和模拟的步长相同。因此，更一般的过程是将用一个可变的分数将每个 $n_i$ 的值改为平衡值：
$$\displaystyle
n_{i}^{new}=n_{i}^{old}+\omega(n_{i}^{eq}-n_{i}^{old})$$

这里的 $\omega$ 是一个介于0和2之间的可变的数。 $\omega$ 越小，意味着碰撞需要更长的时间才能使密度达到平衡；而 $\omega$ 越大意味着碰撞发生得越快。

&emsp;&emsp;再通过进一步的分析，在单位时空离散程度 $\Delta x=\Delta t=1$ 时，我们可以定义和碰撞系数 $\omega$ 相关的运动粘度系数: 
$$\displaystyle
\nu=\frac{1}{3}(\frac{1}{\omega}-\frac{1}{2})$$

当 $\omega=1$ 时，$\nu=\frac{1}{6}$ ，当 $\omega=0$ 时，$\nu$ 正无穷大，而当 $\omega=2$ 时，$\nu=0$.  

所以，lattice-Boltzmann算法如下：

<b>(1).流场</b>
通过遍历，复制相应的 $n_{i}$ 值，将所有分子移动到相邻的或对角线的晶格位置。

<b>(2).碰撞</b>
对于每个格子，执行以下操作：
(a. 根据9个微观的方向密度，计算宏观密度 $\rho$ 和速度分量 $u_x$ 和 $u_y$ ；
(b. 再根据这三个宏观变量，计算平衡的数密度 $n_{i}^{eq}$ ；
(c. 重复更新每个格子的9个方向密度。

<b>(3).边界条件</b>
我们使用一个布尔数组来标记哪些晶格位置包含障碍物而不是流体。然后在每一个流动步骤期间或之后，任何通常会流入这些部位之一的流体，应该直接反弹到它来自的部位，再以相反的速度移动。

## 2. 程序设计方案
&emsp;&emsp;上文已经介绍了核心算法Lattice-Boltzmann方法，接下来介绍程序的具体设计方案：

### 2.0 需要调用的库

```python
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
```

<center>
    <font size="1">
    代码2.1&emsp;需要调用的库
    </font>
</center>

&emsp;&emsp;需要调用numpy进行数组运算，调用matplotlib进行动画生成。

### 2.1 定义网格和物理参数
&emsp;&emsp;首先定义一个512*32的网格，它由9组代表不同方向密度的二维数组构成，如代码图1.3所示。
&emsp;&emsp;然后定义粘度系数visco和碰撞系数omega以及流体的初速度，这样就构建好了一个基本的物理模型。当然，在lattice-Boltzmann算法中还需要定义以下变量：边界（布尔值）、宏观密度、x速度和y速度以及平方速度。
&emsp;&emsp;代码实现如下：

```python
# 参数
height = 32     # 网格高度
width = 512     # 网格长度
viscosity = 0.010   # 粘度
omega = 1./(3*viscosity + 0.5)  # 碰撞系数
u0 = 0.2        # 初速度（向东）

# 微观晶格方向密度
n0 = numpy.zeros((height, width))
nN = numpy.zeros((height, width))
nS = numpy.zeros((height, width))
nE = numpy.zeros((height, width))
nW = numpy.zeros((height, width))
nNW = numpy.zeros((height, width))
nNE = numpy.zeros((height, width))
nSE = numpy.zeros((height, width))
nSW = numpy.zeros((height, width))

# 边界
bar = numpy.zeros((height, width))

# 宏观速度密度
rho = numpy.zeros((height, width))    # 宏观密度
ux = numpy.zeros((height, width))    # x速度
uy = numpy.zeros((height, width))   # y速度
speed2 = numpy.zeros((height, width))  # 平方速度
```

<center>
    <font size="1">
    代码2.2&emsp;定义网格和物理参数
    </font>
</center>

### 2.2 定义流场
&emsp;&emsp;通过函数strea()定义一个包含边界的流场，代码实现如下：

```python
def stream():

    # 遍历每个单元格
    for x in range(0, width-1):
        for y in range(1, height-1):
            # 向北运动
            nN[y, x] = nN[y+1, x]
            # 向西北运动
            nNW[y, x] = nNW[y+1, x+1]
            # 向西运动
            nW[y, x] = nW[y, x+1]
            # 向南运动
            nS[height-y-1, x] = nS[height-y-1-1, x]
            # 向西南运动
            nSW[height-y-1, x] = nSW[height-y-1-1, x+1]
            # 向东运动
            nE[y, width-x-1] = nE[y, width-(x+1)-1]
            # 向东北运动
            nNE[y, width-x-1] = nNE[y+1, width-(x+1)-1]
            # 向东南运动
            nSE[height-y-1, width-x-1] = nSE[height-y-1-1, width-(x+1)-1]

    # 边界条件
    x += 1
    for y in range(1, height-1):
        nN[y, x] = nN[y+1, x]
        nS[height-y-1, x] = nS[height-y-1-1, x]
```

<center>
    <font size="1">
    代码2.3&emsp;定义流场函数stream()
    </font>
</center>

### 2.3 定义边界的反弹性质
&emsp;&emsp;通过函数bounce()定义边界的性质，代码实现如下：

```python
def bounce():

    # 遍历每个内部单元格
    for x in range(2, width-2):
        for y in range(2, height-2):

            # 如果单元格包含边界
            if (bar[y, x]):

                # 返回方向密度并调转方向
                nN[y-1, x] = nS[y, x]
                nS[y+1, x] = nN[y, x]
                nE[y, x+1] = nW[y, x]
                nW[y, x-1] = nE[y, x]
                nNE[y-1, x+1] = nSW[y, x]
                nNW[y-1, x-1] = nSE[y, x]
                nSE[y+1, x+1] = nNW[y, x]
                nSW[y+1, x-1] = nNE[y, x]

                # 清除边界单元格的信息
                n0[y, x] = 0
                nN[y, x] = 0
                nS[y, x] = 0
                nE[y, x] = 0
                nW[y, x] = 0
                nNE[y, x] = 0
                nNW[y, x] = 0
                nSE[y, x] = 0
                nSW[y, x] = 0
```

<center>
    <font size="1">
    代码2.4&emsp;定义边界性质的函数bounce()
    </font>
</center>

### 2.4 定义碰撞过程
&emsp;&emsp;通过函数collide()定义碰撞过程，代码实现如下：

```python
def collide():
    one9th = 1./9.
    one36th = 1./36.

    # 除去顶部、底部和左侧的单元格
    for x in range(1, width-1):
        for y in range(1, height-1):

            # 跳过含有边界的单元格
            if (bar[y, x]):
                continue

            else:
                # 宏观密度
                rho[y, x] = n0[y, x] + nN[y, x] + nE[y, x] + nS[y, x] + \
                    nW[y, x] + nNE[y, x] + nSE[y, x] + nSW[y, x] + nNW[y, x]
                # 宏观速度
                if (rho[y, x] > 0):
                    ux[y, x] = (nE[y, x] + nNE[y, x] + nSE[y, x] - nW[y, x] -
                                nNW[y, x] - nSW[y, x]) * (1-(rho[y, x]-1)+((rho[y, x]-1)**2.))
                    uy[y, x] = (nN[y, x] + nNE[y, x] + nNW[y, x] - nS[y, x] -
                                nSE[y, x] - nSW[y, x]) * (1-(rho[y, x]-1)+((rho[y, x]-1)**2.))

                # 一些参数...
                one9th_rho = one9th * rho[y, x]
                one36th_rho = one36th * rho[y, x]
                vx3 = 3 * ux[y, x]
                vy3 = 3 * uy[y, x]
                vx2 = ux[y, x] * ux[y, x]
                vy2 = uy[y, x] * uy[y, x]
                vxvy2 = 2 * ux[y, x] * uy[y, x]
                v2 = vx2 + vy2
                speed2[y, x] = v2
                v215 = 1.5 * v2

                # 迭代..密度
                nE[y, x] += omega * \
                    (one9th_rho * (1 + vx3 + 4.5*vx2 - v215) - nE[y, x])
                nW[y, x] += omega * \
                    (one9th_rho * (1 - vx3 + 4.5*vx2 - v215) - nW[y, x])
                nN[y, x] += omega * \
                    (one9th_rho * (1 + vy3 + 4.5*vy2 - v215) - nN[y, x])
                nS[y, x] += omega * \
                    (one9th_rho * (1 - vy3 + 4.5*vy2 - v215) - nS[y, x])
                nNE[y, x] += omega * (one36th_rho * (1 + vx3 +
                                                     vy3 + 4.5*(v2+vxvy2) - v215) - nNE[y, x])
                nNW[y, x] += omega * (one36th_rho * (1 - vx3 +
                                                     vy3 + 4.5*(v2-vxvy2) - v215) - nNW[y, x])
                nSE[y, x] += omega * (one36th_rho * (1 + vx3 -
                                                     vy3 + 4.5*(v2-vxvy2) - v215) - nSE[y, x])
                nSW[y, x] += omega * (one36th_rho * (1 - vx3 -
                                                     vy3 + 4.5*(v2+vxvy2) - v215) - nSW[y, x])

                # 质量守恒
                n0[y, x] = rho[y, x] - (nE[y, x]+nW[y, x]+nN[y, x]+nS[y, x] +
                                        nNE[y, x]+nSE[y, x]+nNW[y, x]+nSW[y, x])
```

<center>
    <font size="1">
    代码2.5&emsp;定义碰撞过程的函数collide()
    </font>
</center>

### 2.5 程序初始化
&emsp;&emsp;首先需要将数组n0、nN、nS、nE、nW、nNW、nNE、nSE、nSW根据宏观初速度u0进行初始化，然后在某个位置范围设置bar数组的值，即规定圆柱的位置和大小。如此，圆柱绕流的模拟程序就设置好了。

```python
def initialize(xtop, ytop, yheight, u0=u0):
    one9th = 1./9.
    one36th = 1./36.
    four9ths = 4./9.
    xcoord = 0
    ycoord = 0

    count = 0
    for x in range(width):
        for y in range(height):
            n0[y, x] = four9ths * (1 - 1.5*(u0**2.))
            nN[y, x] = one9th * (1 - 1.5*(u0**2.))
            nS[y, x] = one9th * (1 - 1.5*(u0**2.))
            nE[y, x] = one9th * (1 + 3*u0 + 4.5*(u0**2.) - 1.5*(u0**2.))
            nW[y, x] = one9th * (1 - 3*u0 + 4.5*(u0**2.) - 1.5*(u0**2.))
            nNE[y, x] = one36th * (1 + 3*u0 + 4.5*(u0**2.) - 1.5*(u0**2.))
            nSE[y, x] = one36th * (1 + 3*u0 + 4.5*(u0**2.) - 1.5*(u0**2.))
            nNW[y, x] = one36th * (1 - 3*u0 + 4.5*(u0**2.) - 1.5*(u0**2.))
            nSW[y, x] = one36th * (1 - 3*u0 + 4.5*(u0**2.) - 1.5*(u0**2.))

            rho[y, x] = n0[y, x] + nN[y, x] + nS[y, x] + nE[y, x] + \
                nW[y, x] + nNE[y, x] + nSE[y, x] + nNW[y, x] + nSW[y, x]

            ux[y, x] = (nE[y, x] + nNE[y, x] + nSE[y, x] - nW[y, x] - nNW[y, x] -
                        nSW[y, x]) * (1-(rho[y, x]-1)+((rho[y, x]-1)**2.))
            uy[y, x] = (nN[y, x] + nNE[y, x] + nNW[y, x] - nS[y, x] - nSE[y, x] -
                        nSW[y, x]) * (1-(rho[y, x]-1)+((rho[y, x]-1)**2.))

            if (xcoord == xtop):
                if (ycoord >= ytop):
                    if (ycoord < (ytop+yheight)):
                        count += 1
                        bar[ycoord, xcoord] = 1

            xcoord = (xcoord+1) if xcoord < (width-1) else 0
            ycoord = ycoord if (xcoord != 0) else (ycoord + 1)
```

<center>
    <font size="1">
    代码2.6&emsp;定义初始化过程的函数initialize(xtop, ytop, yheight, u0=u0)
    </font>
</center>

### 2.6 动画文件生成
&emsp;&emsp;使用matplotlib中的pyplot和animation模组，在重复迭代strea()、bounce()和collide()三个函数的过程中生成图像数组，以600fps帧率生成15s的MP4视频文件并保存。

```python
# 制作动画
fps = 600
nSeconds = 12

fig = plt.figure(figsize=(20, 5))

initialize(25, 11, 10)

for i in range(10):
    stream()
    bounce()
    collide()

a = speed2
im = plt.imshow(a.reshape(height, width))


def animate_func(i):
    stream()
    bounce()
    collide()
    im.set_array(speed2.reshape(height, width))
    return [im]


anim = animation.FuncAnimation(
    fig,
    animate_func,
    frames=nSeconds * fps,
    interval=1000 / fps,
)

print('Done!')

# 保存
f = r"./simulation.mp4"
writervideo = animation.FFMpegWriter(fps=600)
anim.save(f, writer=writervideo)

print('save!')
```

<center>
    <font size="1">
    代码2.7&emsp;生成simulation.mp4视频文件
    </font>
</center>

## 3 创新性描述

&emsp;&emsp;在学习了N-S方程和latti-Boltzmann之后，笔者掌握了基本的思路，自主完成了网格的搭建和数组迭代部分的程序，运用的其实也都是课内所学的基础内容。（如有雷同，纯属巧合）
&emsp;&emsp;通过调用matplotlib生成了视频文件，能够更直观地展示流体运动的全过程。当然，在测试过程中也发现了运行速度慢、数据溢出等情况。

## 4.运行方法和参数设置

&emsp;&emsp;所有参数均已内置（见代码2.2），直接运行程序即可。

&emsp;&emsp;该程序还有许多需要改进的地方，比如说运行时间长、占用内存大等。不过笔者经过许多尝试后仍然无法很好地解决这些问题/(ㄒoㄒ)/~~

&emsp;&emsp;运行结果即为

## 5.学习心得和收获

&emsp;&emsp;首先，通过本次大作业，我较为系统地学习了计算流体力学的一些基础知识，比如说N-S方程和LBM模拟方法。掌握了多维数组叠加构造物理空间属性这个神奇而强大的思路，编程的思维有了很大的提升。

&emsp;&emsp;其次，在Debug和程序优化的过程中，我收获到了十分深刻的绝望感😰先是二维数组的行列坐标弄反花费掉了一个下午的时间，再是程序迭代到第7000次后发生了数据下溢的情况，又花费了一个晚上的时间……也算是一种体验吧。

&emsp;&emsp;而且在最后撰写报告的时候，看了一下之前的文档范例……发现这个项目已经有人做过（类似的）了，而且功能还比我的更加完善(ˉ▽ˉ；)...

## 参考资料：

“1.1Navier-Stokes方程——流体中的牛顿第二定律”https://zhuanlan.zhihu.com/p/624347343

“1.2Navier-Stokes方程——对流导数”https://zhuanlan.zhihu.com/p/624352446

“为什么会出现卡门涡街？”https://www.zhihu.com/question/42116401

"Lattice-Boltzmann Fluid Dynamics"——Physics 3300, Weber State University, Spring Semester, 2012
https://physics.weber.edu/schroeder/javacourse/LatticeBoltzmann.pdf