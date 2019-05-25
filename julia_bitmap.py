#!/usr/bin/python2
import sys
import time
import numpy as np
import opti as opt
import imageio

ITER_NUM = 250
xmin, xmax = (-0.7, 0.3)
ymin, ymax = (-0.3, 1.3)
c = 0.345 - 0.45j
# xmin, xmax = (-2.35, 0.85)
# ymin, ymax = (-1.25, 1.25)
# c = -0.75 + 0.j
m = 15

width = 1080
yrange = np.abs(ymax - ymin)
xrange = xmax - xmin
height = np.int(yrange * width / xrange)
print(width, height)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        m = int(sys.argv[1])
    path = 'm' + str(m) + '.png'

    xaxis = np.linspace(xmin, xmax, width)
    yaxis = np.linspace(ymin, ymax, height)

    bitmap = np.zeros((height, width))

    start_time = time.time()

    cc = np.array([c.real, c.imag])
    size = np.array([width, height])
    opt.julia_main(size, m, ITER_NUM, 1, cc, xaxis, yaxis, bitmap)
    print("执行时间 {} 秒".format(round(time.time() - start_time, 2)))

    if width < height:
        bitmap = bitmap.T
    imageio.imwrite(path, bitmap)



