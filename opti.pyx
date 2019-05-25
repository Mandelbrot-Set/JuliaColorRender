cimport cython
import numpy as np

cdef extern from "math.h":
    float log(float theta)
    float sqrt(float theta)
    float sin(float theta)
    float atan2(float x, float y)
    float pow(float x, float y)
    int floor(float x)

# Functions for "Smooth Iteration Count"
cdef float smooth_iter(float z_real, float z_imag, int iters):
    cdef:
        float z_mod, c_1
    z_mod = sqrt(z_real * z_real + z_imag * z_imag)
    c_1 = log(2) / log(z_mod)
    return iters + 1 + log(c_1) / log(2)


# Functions for Triangle Inequality Average method for colouring fractals
# Pre: zs has at least two elements
cdef t(double minus_real, double minus_imag, double zn_real, double zn_imag, double const_real, double const_imag):
    cdef:
        double abs_zn_minus1, mn, Mn

    abs_minus = sqrt(minus_real * minus_real + minus_imag * minus_imag)
    abs_const = sqrt(const_real * const_real + const_imag * const_imag)
    abs_zn = sqrt(zn_real * zn_real + zn_imag * zn_imag)

    if abs_minus > abs_const:
        mn = abs_minus - abs_const
    else:
        mn = abs_const - abs_minus
    Mn = abs_minus + abs_const

    if (Mn- mn) == 0:
        print('t-> INFINITY')
        return 9999999999
    else:
        return (abs_zn - mn) / (Mn - mn)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef julia_main(long[:] size, int m, int iterations, int flag, double[:] c_ri, double[:] xaxis, double[:] yaxis, bitmap):
    cdef:
        float index=0
        int   width=size[0]
        int   height=size[1]

    for row in range(width):
        for col in range(height):
            index = julia_escape_time(complex(xaxis[row], yaxis[col]), complex(c_ri[0], c_ri[1]), iterations, m)

            if flag == 1:
                bitmap[col][row] = index
            else:
                bitmap.putpixel((col, row), (0, int(index*255), 0))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef julia_normal(long[:] size, int m, float zoom, int iterations, double[:] c_ri, bitmap):
    cdef:
        float index=0
        int   w=size[0]
        int   h=size[1]
        float scale = 1.
    if w > h:
        scale = w / h
    for x in range(w):
        for y in range(h):
            zx = scale * (x - w / 2) / (0.5 * zoom * w)
            zy = 1.0 * (y - h / 2) / (0.5 * zoom * h)

            d = julia_escape_time(complex(zx, zy), complex(c_ri[0], c_ri[1]), iterations, m)
            if  np.isnan(d) or np.isinf(d):
                d = 0
            # 白色为主
            bitmap[x, y] = (int(d*255), int(d*255), int(d*255))
            # 蓝色为主
            # bitmap[x, y] = (0, 0, int(d*255))


cdef float julia_escape_time(complex z, complex c, int iterations, int m):
    cdef:
        int max_i = iterations + 1
        int z_iter = 0
        float d, smooth_count, sum1, sum2, tmp

    zr, zi = [], []
    for z_iter in range(1, max_i):
        tmp = z.real * z.real - z.imag * z.imag + c.real
        z.imag, z.real = 2.0 * z.real * z.imag + c.imag, tmp
        zr.append(z.real)
        zi.append(z.imag)
        if z.real * z.real + z.imag * z.imag >= 4:
            break

    d = smooth_iter(zr[z_iter - 1], zi[z_iter - 1], z_iter) % iterations

    last_iter_num = z_iter if z_iter < m else m

    if z_iter - last_iter_num == 0:
        return 0

    for n in range(last_iter_num, z_iter):
        sum1 += t(zr[n - 2], zi[n-2], zr[n - 1], zi[n - 1], c.real, c.imag)

    sum1 = sum1 / (z_iter - last_iter_num)

    zr, zi = zr[:-1], zi[:-1]
    for nn in range(last_iter_num, z_iter):
        sum2 += t(zr[nn - 2], zi[nn-2], zr[nn - 1], zi[nn - 1], c.real, c.imag)

    sum2 = sum2 / (z_iter - last_iter_num)

    return d * sum1 + (1-d) * sum2


"""
The following is about to Mandelbrot implementation.
"""
cdef double translate(double value, double left_min, double left_max,
               double right_min, double right_max):
    cdef:
        double left_span, right_span, value_scaled

    left_span = left_max - left_min
    right_span = right_max - right_min
    value_scaled = float(value - left_min) / float(left_span)

    return right_min + (value_scaled * right_span)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef m_loop(int w, int h, xm, ym, int iterations, int m, pix):
    cdef:
        int x, y
        double re, rm
        float d = 0

    for x in range(w):
        for y in range(h):
            # Mandelbrot
            re = 0.9 * translate(x, 0, w, xm[0], xm[1])
            im = translate(y, 0, h, ym[1], ym[0])
            d = mandelbrot(re, im, m, iterations)

            if  np.isnan(d) or np.isinf(d):
                d = 0
            g = int(d*255)
            if g > 255:
                g = 255
            pix[x, y] = (g, g, g)

cdef float mandelbrot(double creal, double cimag, int m, int maxiter):
    cdef:
        double real2, imag2
        double real = 0., imag = 0.
        int z_iter=0
        int max_i = maxiter + 1
        float sum1 = 0
        float sum2 = 0

    zr, zi = [], []
    for z_iter in range(1, max_i):
        real2 = real*real
        imag2 = imag*imag

        zr.append(real2)
        zi.append(imag2)

        if real2 + imag2 > 4.0:
            break
        imag = 2* real*imag + cimag
        real = real2 - imag2 + creal

    d = smooth_iter(zr[z_iter - 1], zi[z_iter - 1], z_iter) % maxiter

    last_iter_num = z_iter if z_iter < m else m

    if z_iter - last_iter_num == 0:
        return 0

    for n in range(last_iter_num, z_iter):
        sum1 += t(zr[n - 2], zi[n-2], zr[n - 1], zi[n - 1], creal, cimag)

    sum1 = sum1 / (z_iter - last_iter_num)

    zr, zi = zr[:-1], zi[:-1]
    for nn in range(last_iter_num, z_iter):
        sum2 += t(zr[nn - 2], zi[nn-2], zr[nn - 1], zi[nn - 1], creal, cimag)

    sum2 = sum2 / (z_iter - last_iter_num)

    return d * sum1 + (1-d) * sum2

# Stripe Average Coloring
# http://www.fractalforums.com/general-discussion/stripe-average-coloring/
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sac_loop(int w, int h, float xmin, float ymin, float wh, float stripes, int iterations, pix):
    cdef:
        float xmax = xmin + wh
        float ymax = ymin + wh
        float dx = (xmax-xmin)/w
        float dy = (ymax-ymin)/h
        float x = xmin
        float y = ymin
        float zr, zi, last_zr, last_zi, orbit_count= 0
        float zrr, zii, two_ri = 0.
        float last_orbit, small_count, frac
        int orbit_color

    for i in range(w):
        y = ymin
        for j in range(h):
            zr, zi = x, y
            last_zr, last_zi = x, y
            orbit_count = 0

            for n in range(iterations):
                zrr = zr * zr
                zii = zi * zi
                two_ri = 2 * zr * zi
                zr, zi = zrr-zii + x, two_ri + y
                if zrr + zii > 10000:
                    break
                orbit_count += 0.5 + 0.5 * sin(stripes*atan2(zi, zr))
                last_zr = zr
                last_zi = zi
            if n == iterations:
                pix[i, j] = (255, 255, 255)
            else:
                last_orbit = 0.5+0.5*sin(stripes*atan2(last_zi, last_zr))
                small_count = orbit_count-last_orbit
                orbit_count /= n
                small_count /= n-1

                # frac = -1+log(0.5*log(10000))/log(2)-log(0.5*log(last_zr*last_zr+last_zi*last_zi))/log(2)
                frac = -1+log(2.0*log(10000))/log(2)-log(0.5*log(last_zr*last_zr+last_zi*last_zi))/log(2)
                # frac = 1+(log(log(10000)/log(sqrt(last_zr*last_zr+last_zi*last_zi)))/log(2))
                # frac =1.0+(log2(log(EscapeRadiusSquared)/log(z2)))

                mix = frac * orbit_count + (1-frac) * small_count
                if  np.isnan(mix) or np.isinf(mix):
                    mix = 255
                orbit_color = int(mix*255)

                # pix[i, j] = (orbit_color << 21) + (orbit_color << 10) + orbit_color * 8
                # pix[i, j] = (0, 0, orbit_color)
                pix[i, j] = (orbit_color, orbit_color, orbit_color)
            y += dy
        x += dx


def compute_orbit(zc, int iterations, float escape_radius, orbit):
    escaped = False
    num_point = 1

    orbit.clear()
    orbit.append([0, 0])
    er = escape_radius * escape_radius

    for i in range(1, iterations):
        zp = orbit[i-1]

        orbit.append([0,0])
        zn = orbit[i]

        zn[0] = zp[0]*zp[0] - zp[1]*zp[1] + zc.real
        zn[1] = zp[0]*zp[1] + zp[1]*zp[0] + zc.imag
        num_point += 1

        mr = zn[0]*zn[0] + zn[1]*zn[1]
        if mr > er:
            escaped = True
            break

    return escaped, num_point

cdef mag(r, i):
    cdef:
        double s

    s = r * r + i * i
    if s > 0:
        s = sqrt(s)

    return s

def compute_smooth(orbit_, num_point, escape_radius):
    zc = orbit_[num_point - 1]
    mc = mag(zc[0], zc[1])

    return 1 + 1 / log(2) * log(log(escape_radius) / log(mc))


# sets global var avgSum
def triangle_in_eq(orbit, numPoints, c):

    avgSum = {'s1': 0, 's2': 0}

    if numPoints < 3:
        return 0

    mc = mag(c.real, c.imag)

    for i in range(2, numPoints):
        zp = orbit[i - 1]
        zc = orbit[i]

        # tc = zp ^ 2
        re = zp[0] * zp[0] - zp[1] * zp[1]
        im = zp[0] * zp[1] + zp[1] * zp[0]

        mp = mag(re, im)
        m = np.abs(mp - mc)
        M = mp + mc
        avgSum['s1'] = avgSum['s2']
        avgSum['s2'] = (mag(zc[0], zc[1]) - m) / (M - m)

    avgSum['s1'] /= numPoints - 3
    avgSum['s2'] /= numPoints - 2

    return avgSum


def get_pal(t):
    t = 1 - t
    return {
        'r': pow(t, 3.0),
        'g': pow(t, 1.0),
        'b': pow(t, 0.2)
    }

def clamp(float n):
    n *= 255
    if n < 0:
        return 0
    if n > 255:
        return 255

    return floor(n)


def render(c):
    return clamp(c['r']), clamp(c['g']), clamp(c['b'])

# Mandelbrot Average Colorings
# http://jsfiddle.net/ndwL7tby/
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mac_loop(int w, int h, double[:] center, int super_samples, int iterations, pix):
    cdef:
        int bh = h
        int bw = w
        float radius = center[2]
        float minRe = center[0] - radius
        float maxRe = center[0] + radius
        float minIm = center[1] - radius*(h/w)
        float maxIm = center[1] + radius*(h/w)
        float escape_radius = center[3]

    in_color = {'r': 1, 'g': 1, 'b': 1}
    orbit = []
    for x in range(w):
        for y in range(h):
            p = {'r': 0, 'g': 0, 'b': 0}
            for sj in range(super_samples):
                for si in range(super_samples):
                    re = translate(x, 0, w, minRe, maxRe)
                    im = translate(y, 0, h, maxIm, minIm)
                    c = complex(re, im)
                    escaped_, num_points = compute_orbit(c, iterations, escape_radius, orbit)

                    if escaped_:
                        sm = compute_smooth(orbit, num_points, escape_radius)
                        avg_sum = triangle_in_eq(orbit, num_points, c)
                        t = avg_sum['s1'] * (1 - sm) + avg_sum['s2'] * sm
                        m = get_pal(t)
                        p['r'] += m['r']
                        p['g'] += m['g']
                        p['b'] += m['b']
                    else:
                        p['r'] += in_color['r']
                        p['g'] += in_color['g']
                        p['b'] += in_color['b']

            p['r'] /= super_samples * super_samples
            p['g'] /= super_samples * super_samples
            p['b'] /= super_samples * super_samples

            rgb = render(p)
            pix[x, y] = rgb
