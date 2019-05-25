import math
import random


def clamp(x):
    return max(0, min(x, 255))


def create_palette():
    palette = [(0, 0, 0)]
    red_b = 2 * math.pi / (random.randint(0, 128) + 128)
    red_c = 256 * random.random()
    green_b = 2 * math.pi / (random.randint(0, 128) + 128)
    green_c = 256 * random.random()
    blue_b = 2 * math.pi / (random.randint(0, 128) + 128)
    blue_c = 256 * random.random()
    for i in range(256):
        r = clamp(int(256 * (0.5 * math.sin(red_b * i + red_c) + 0.5)))
        g = clamp(int(256 * (0.5 * math.sin(green_b * i + green_c) + 0.5)))
        b = clamp(int(256 * (0.5 * math.sin(blue_b * i + blue_c) + 0.5)))
        palette.append((r, g, b))

    return palette


def get_color(palette):
    colours = len(palette)

    def color(i):
        return palette[i % colours]

    return color


def get_julia_set_by_count(width, height, c, delta, zoom=1., iterations=255, pixels=None, color_func=None):
    if width > height:
        scale = width / height
    else:
        scale = 1

    for x in range(width):
        for y in range(height):
            zx = scale * (x - width / 2) / (0.5 * zoom * width) + delta[0]
            zy = 1.0 * (y - height / 2) / (0.5 * zoom * height) + delta[1]
            i = iterations

            while zx * zx + zy * zy < 4 and i > 0:
                tmp = zx * zx - zy * zy + c[0]
                zy, zx = 2.0 * zx * zy + c[1], tmp
                i -= 1
            if pixels is not None:
                pixels[x, y] = color_func(i)
