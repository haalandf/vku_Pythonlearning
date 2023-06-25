import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["figure.figsize"] = (50, 3)

# 参数
height = 32     # 网格高度
width = 300     # 网格长度
viscosity = 0.008   # 粘度
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


# 流场
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


# 反弹
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


# 碰撞
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


# 初始化
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


# 制作动画
fps = 600
nSeconds = 15

fig = plt.figure(figsize=(20, 5))

initialize(25, 11, 10)

for i in range(10):
    stream()
    bounce()
    collide()

print('done!')

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
f = r"./simulation_1.mp4"
writervideo = animation.FFMpegWriter(fps=600)
anim.save(f, writer=writervideo)

print('save!')
