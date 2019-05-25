#!/usr/bin/python2
import sys
import time
import numpy as np
import opti as opt
from PIL import Image

if __name__ == '__main__':
    if len(sys.argv) == 2:
        m = int(sys.argv[1])
    width = 1000
    height = 1000
    path = 'pictures/jn{}.png'
    bitmap = Image.new("RGB", (width, height), "white")
    pix = bitmap.load()

    start_time = time.time()

    size = np.array([width, height])

    # opt.julia_normal(size, 2, .8, 1000, np.array([0.345, -0.45]), pix)
    opt.julia_normal(size, 4, .8, 1000, np.array([0.345, -0.45]), pix)
    # opt.julia_normal(size, 25, .8, 1000, np.array([-0.7, 0.27015]), pix)
    # opt.julia_normal(size, 12, .8, 1000, np.array([0.285, -0.01]), pix)
    # opt.julia_normal(size, 35, 1., 1000, np.array([-0.70176, 0.3842]), pix)
    # opt.julia_normal(size, 30, .8, 1000, np.array([-0.835, -0.2321]), pix)
    # opt.julia_normal(size, 40, .8, 1000, np.array([-0.8, 0.156]), pix)
    # opt.julia_normal(size, 40, .8, 1000, np.array([-0.835, -0.2321]), pix)

    print("执行时间 {} 秒".format(round(time.time() - start_time, 2)))
    bitmap.save(path.format(time.time()), "PNG", optimize=True)
    bitmap.show()



