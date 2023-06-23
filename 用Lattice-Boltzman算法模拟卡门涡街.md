<center>
    <font size="6">
        <b>用Lattice-Boltzman算法模拟卡门涡街</b>
    </font>
    <br>
    <font size="4">
        管理学院 PB22151726 方浩然
    </font>
</center>
<hr>

## 0.简介
&emsp;&emsp;卡门涡街是由西奥多·冯·卡门的学生在实验中发现、冯·卡门在理论上研究阐明的。这是圆柱绕流的一种特殊情况（雷诺数$Re:47 < Re < 10^{5}$）,也是计算流体力学（Computational Fluid Dynamics）中常见而基础的模拟项目。当然，目前主流的CFD模拟软件包括Fluent、cfx等等，而python由于计算能力的限制并不是十分适用于CFD模拟，但在限制时间范围和精度之后，也能够在python中实现简单的CFD模拟，其核心算法即本文所采用的Lattice-Boltzman算法。本文构造了一个512*32的网格，并在网格上定义流场以及物理碰撞过程，最终生成MP4格式视频，成功实现了卡门涡街的模拟。

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
![理论示意图](/理论示意图.jpg)
<center>
    <font size="1">
    卡门涡街的理论示意图（图片来自网络）
    </font>
</center>

### 1.2 Lattice-Boltzman方法
&emsp;&emsp;