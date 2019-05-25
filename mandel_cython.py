import opti as opt
from PIL import Image
import time
import numpy as np

w, h = 1024, 768
bitmap = Image.new("RGB", (w, h), "white")
pix = bitmap.load()
start = time.time()

# opt.m_loop(w, h, [-2.35, 0.8500000000000001], [-1, 1], 200, 6, pix)
opt.m_loop(w, h, [-0.6966666666666667, -0.3766666666666666], [0.42000000000000004, 0.62], 1000, 6, pix)
# opt.m_loop(w, h, [-0.5876296296296296, -0.5556296296296296], [0.5525185185185185, 0.5725185185185185], 300, 25, pix)

# 测试Stripe Average Coloring
# opt.sac_loop(w, h, -2.5, -2.0, 4, 5.0, 1000, pix)
# opt.sac_loop(w, h, -0.6966666666666667, 0.42000000000000004, 0.3, 4.0, 1000, pix)

# super_samples = 1
# iterations = 100
# # center = np.array([-0.740, 0.208, 0.01, 1000])
# center = np.array([-0.535, 0.61, 0.15, 100])
# opt.mac_loop(w, h, center, super_samples, iterations, pix)

print("执行时间 {} 分钟".format(round((time.time() - start) / 60, 2)))
bitmap.save("pictures/{}.png".format(time.strftime("%Y-%m-%d-%H:%M:%S")), "PNG", optimize=True)
bitmap.show()
