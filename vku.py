import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["figure.figsize"] = (50, 3)

# 参数
height = 32     # 网格高度
width = 512     # 网格长度
viscosity = 0.008   # 粘度
omega = 1./(3*viscosity + 0.5)  # 松弛系数
u0 = 0.2        # 初速度（向东）

# 微观晶格方向密度
n0 = numpy.zeros(height*width)
nN = numpy.zeros(height*width)
nS = numpy.zeros(height*width)
nE = numpy.zeros(height*width)
nW = numpy.zeros(height*width)
nNW = numpy.zeros(height*width)
nNE = numpy.zeros(height*width)
nSE = numpy.zeros(height*width)
nSW = numpy.zeros(height*width)

# 边界
bar = numpy.zeros(height*width)

# 宏观速度密度
rho = numpy.zeros(height*width)    # 细胞密度
ux = numpy.zeros(height*width)    # x速度
uy = numpy.zeros(height*width)    # y速度
speed2 = numpy.zeros(height*width)  # 平方速度


# 流场
def stream():

    # 遍历每个单元格
    for x in range(0, width-1):
        for y in range(1, height-1):
            # 向北运动
            nN[y*width + x] = nN[y*width + x + width]
            # 向西北运动
            nNW[y*width + x] = nNW[y*width + x + width + 1]
            # 向西运动
            nW[y*width + x] = nW[y*width + x + 1]
            # 向南运动
            nS[(height-y-1)*width + x] = nS[(height-y-1-1)*width + x]
            # 向西南运动
            nSW[(height-y-1)*width + x] = nSW[(height-y-1-1)
                                              * width + x + 1]
            # 向东运动
            nE[y*width + (width-x-1)] = nE[y*width + (width-(x+1)-1)]
            # 向东北运动
            nNE[y*width + (width-x-1)] = nNE[y*width +
                                             width + (width-(x+1)-1)]
            # 向东南运动
            nSE[(height-y-1)*width + (width-x-1)] = nSE[(height-y-1-1)*width +
                                                        (width-(x+1)-1)]

    # 边界条件
    x += 1
    for y in range(1, height-1):
        nN[y*width + x] = nN[y*width + x + width]
        nS[(height-y-1)*width + x] = nS[(height-y-1-1)*width + x]


# 反弹
def bounce():

    # 遍历每个内部单元格
    for x in range(2, width-2):
        for y in range(2, height-2):

            # 如果单元格包含边界
            if (bar[y*width + x]):

                # 返回方向密度并调转方向
                nN[(y-1)*width + x] = nS[y*width + x]
                nS[(y+1)*width + x] = nN[y*width + x]
                nE[y*width + x + 1] = nW[y*width + x]
                nW[y*width + (x-1)] = nE[y*width + x]
                nNE[(y-1)*width + (x+1)] = nSW[y*width + x]
                nNW[(y-1)*width + (x-1)] = nSE[y*width + x]
                nSE[(y+1)*width + (x+1)] = nNW[y*width + x]
                nSW[(y+1)*width + (x-1)] = nNE[y*width + x]

                # 清除边界单元格的信息
                n0[y*width + x] = 0
                nN[y*width + x] = 0
                nS[y*width + x] = 0
                nE[y*width + x] = 0
                nW[y*width + x] = 0
                nNE[y*width + x] = 0
                nNW[y*width + x] = 0
                nSE[y*width + x] = 0
                nSW[y*width + x] = 0


# 碰撞
def collide():
    one9th = 1./9.
    one36th = 1./36.

    # 除去顶部、底部和左侧的单元格
    for x in range(1, width-1):
        for y in range(1, height-1):

            i = y*width + x

            # 跳过含有边界的单元格
            if (bar[i]):
                continue

            else:
                # 宏观密度
                rho[i] = n0[i] + nN[i] + nE[i] + nS[i] + \
                    nW[i] + nNE[i] + nSE[i] + nSW[i] + nNW[i]
                # 宏观速度
                if (rho[i] > 0):
                    ux[i] = (nE[i] + nNE[i] + nSE[i] - nW[i] -
                             nNW[i] - nSW[i]) * (1-(rho[i]-1)+((rho[i]-1)**2.))
                    uy[i] = (nN[i] + nNE[i] + nNW[i] - nS[i] -
                             nSE[i] - nSW[i]) * (1-(rho[i]-1)+((rho[i]-1)**2.))

                # 一些参数...
                one9th_rho = one9th * rho[i]
                one36th_rho = one36th * rho[i]
                vx3 = 3 * ux[i]
                vy3 = 3 * uy[i]
                vx2 = ux[i] * ux[i]
                vy2 = uy[i] * uy[i]
                vxvy2 = 2 * ux[i] * uy[i]
                v2 = vx2 + vy2
                speed2[i] = v2
                v215 = 1.5 * v2

                # 迭代..密度
                nE[i] += omega * \
                    (one9th_rho * (1 + vx3 + 4.5*vx2 - v215) - nE[i])
                nW[i] += omega * \
                    (one9th_rho * (1 - vx3 + 4.5*vx2 - v215) - nW[i])
                nN[i] += omega * \
                    (one9th_rho * (1 + vy3 + 4.5*vy2 - v215) - nN[i])
                nS[i] += omega * \
                    (one9th_rho * (1 - vy3 + 4.5*vy2 - v215) - nS[i])
                nNE[i] += omega * (one36th_rho * (1 + vx3 +
                                   vy3 + 4.5*(v2+vxvy2) - v215) - nNE[i])
                nNW[i] += omega * (one36th_rho * (1 - vx3 +
                                   vy3 + 4.5*(v2-vxvy2) - v215) - nNW[i])
                nSE[i] += omega * (one36th_rho * (1 + vx3 -
                                   vy3 + 4.5*(v2-vxvy2) - v215) - nSE[i])
                nSW[i] += omega * (one36th_rho * (1 - vx3 -
                                   vy3 + 4.5*(v2+vxvy2) - v215) - nSW[i])

                # 质量守恒
                n0[i] = rho[i] - (nE[i]+nW[i]+nN[i]+nS[i] +
                                  nNE[i]+nSE[i]+nNW[i]+nSW[i])


# 初始化
def initialize(xtop, ytop, yheight, u0=u0):
    one9th = 1./9.
    one36th = 1./36.
    four9ths = 4./9.
    xcoord = 0
    ycoord = 0

    count = 0
    for i in range(height*width):
        n0[i] = four9ths * (1 - 1.5*(u0**2.))
        nN[i] = one9th * (1 - 1.5*(u0**2.))
        nS[i] = one9th * (1 - 1.5*(u0**2.))
        nE[i] = one9th * (1 + 3*u0 + 4.5*(u0**2.) - 1.5*(u0**2.))
        nW[i] = one9th * (1 - 3*u0 + 4.5*(u0**2.) - 1.5*(u0**2.))
        nNE[i] = one36th * (1 + 3*u0 + 4.5*(u0**2.) - 1.5*(u0**2.))
        nSE[i] = one36th * (1 + 3*u0 + 4.5*(u0**2.) - 1.5*(u0**2.))
        nNW[i] = one36th * (1 - 3*u0 + 4.5*(u0**2.) - 1.5*(u0**2.))
        nSW[i] = one36th * (1 - 3*u0 + 4.5*(u0**2.) - 1.5*(u0**2.))

        rho[i] = n0[i] + nN[i] + nS[i] + nE[i] + \
            nW[i] + nNE[i] + nSE[i] + nNW[i] + nSW[i]

        ux[i] = (nE[i] + nNE[i] + nSE[i] - nW[i] - nNW[i] -
                 nSW[i]) * (1-(rho[i]-1)+((rho[i]-1)**2.))
        uy[i] = (nN[i] + nNE[i] + nNW[i] - nS[i] - nSE[i] -
                 nSW[i]) * (1-(rho[i]-1)+((rho[i]-1)**2.))

        if (xcoord == xtop):
            if (ycoord >= ytop):
                if (ycoord < (ytop+yheight)):
                    count += 1
                    bar[ycoord*width + xcoord] = 1

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
    interval=1000 / fps,  # in ms
)

print('Done!')

# 保存
f = r"./animation4.mp4"
writervideo = animation.FFMpegWriter(fps=600)
anim.save(f, writer=writervideo)

print('save!')
