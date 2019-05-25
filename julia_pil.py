#!/usr/bin/python2
import sys
import time
import numpy as np
from PIL import Image
import opti as opt

ITER_NUM = 250
EPS = 0.01
BAILOUT = 2

xmin, xmax = (-0.7, 0.3)
ymin, ymax = (-0.3, 1.3)
c = 0.345 - 0.45j
p = 2
m = 5

# c = -0.7+0.27015j

width = 1920
yrange = np.abs(ymax - ymin)
xrange = xmax - xmin
height = np.int(yrange * width / xrange)
print(width, height)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        m = int(sys.argv[1])
    path = 'p{}.png'

    xaxis = np.linspace(xmin, xmax, width)
    yaxis = np.linspace(ymin, ymax, height)

    bitmap = Image.new("RGB", (height, width), "black")

    start_time = time.time()

    cc = np.array([c.real, c.imag])
    size = np.array([width, height])
    opt.julia_main(size, m, ITER_NUM, 2, cc, xaxis, yaxis, bitmap)

    print("执行时间 {} 秒".format(round(time.time() - start_time, 2)))

    bitmap.save(path.format(time.time()), "PNG", optimize=True)

