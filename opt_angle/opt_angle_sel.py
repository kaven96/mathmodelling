#
#   Optimization
#
# Firstly we have to select our mathematical model of blade fastener behaviour under centrifugal loads.
# We can describe stresses, appearing in fatener's teeth by 3 equations:
#
## a) sigma_sm = K_sm * P_j / (b * S/2)
## b) sigma_iz = K_iz * P_j / (b * S/2)
## c) tau_sr = K_sr * P_j / (b * S/2)
#
# where: 
# K_sm = f(gamma, beta, phi)
# K_iz = f(gamma, beta, phi) 
# K_sr = f(gamma, beta, phi)
# P_j -- actual centrifugal force, acting on each fastener's tooth. It can be found as summ of:
# - centrifugal force from blade's airfoil;
# - centrifugal force from bondage shelf (is it correct terminology?);
# - centrifugal force from fastener,
# devided by number of teeth.
# Value of firts two components of actual centrifugal forces can be found from 3D model.
# The 3rd component -- as the volume of truncated pyramid multiplied by material density value.
# Volume of truncated pyramid, in turn, can be found like this:
# V = 1 / 3 * H * (S_1 + sqrt(S_1 * S_2) + S_2)
# where S -- is function of b (fastener width), l (teeth length) and h_l (fastener height)
#
# The function we're going to find minimum of is described by equation:
# sigma_sm + sigma_iz + tau_sr
# constraining equations are: sigma_sm, sigma_iz, tau_sr which must be less or equal [sigma] or [tau].
#
# Boundaries for angles are:
# gamma = [55, 65] degrees
# beta = [105, 120] degrees
# phi = [20, 50] degrees


# First of all let's import pyhon modules, necessary for our optimization problem:
import numpy as np
from numpy import sin, cos, tan
from scipy.optimize import NonlinearConstraint, Bounds, minimize, BFGS, SR1


# Then we are going to define all of the functions we need for this problem:
def K_sm(x):
    gamma = x[0]
    beta = x[1]
    phi = x[2]
    psi = beta - (np.radians(90) - phi/2)
    m1 = (2 * sin(beta - gamma) / sin(gamma) - 0.7 * tan(1 / (gamma / 2)) - 0.05 / cos(psi))*cos(psi)
    return 1 / m1


def K_iz(x):
    gamma = x[0]
    beta = x[1]
    phi = x[2]
    psi = beta - (np.radians(90) - phi/2)
    m1 = (2 * sin(beta - gamma) / sin(gamma) - 0.7 * tan(1 / (gamma / 2)) - 0.05 / cos(psi))*cos(psi)
    m4 = (2 * sin(beta - gamma) / sin(gamma) - 0.35 * tan(1 / (gamma / 2)) + 0.35 * tan(1 / (beta / 2))) * sin(gamma) / sin(beta - gamma)
    m5 = m1 / 2 / cos(psi) + 0.35 * tan(1 / (beta / 2)) - m4 / 2 * sin(beta - 90) + 0.05 / cos(psi)
    return 6 * m5 / m4**2 / cos(psi)


def K_sr(x):
    gamma = x[0]
    beta = x[1]
    phi = x[2]
    psi = beta - (np.radians(90) - phi/2)
    m3 = (2 * sin(beta - gamma) / sin(gamma) - 0.35 * tan(1 / (gamma / 2)) - 0.05 / cos(psi))*sin(gamma) / sin(beta - gamma + phi / 2)
    return 1 / m3





#gamma_first_approx = np.radians(49)
#beta_first_approx = np.radians(121)
#phi_first_approx = np.radians(17)


# 1. Define bounds for required angles:
bounds = Bounds([np.radians(55), np.radians(105), np.radians(20)], [np.radians(65), np.radians(120), np.radians(50)])


# 2. Define nonlinear constraints for optimizing function
def cons_func(x):
    gamma = x[0]
    beta = x[1]
    phi = x[2]
    psi = beta - (np.radians(90) - phi/2)

    # Input data
    S = 3.5 # mm
    b = 30 # mm
    blade_vol = 3394.4957 # mm3
    mat_density = 8760 # kg / m3
    n = 19200 # rpm
    blade_Rcm = 167.89 #mm
    z = 3


    # Fillet radius as function of S (joint step)
    def r_func(S):
        return 0.1737 * S - 0.0146 # magic constants are taken by approximation of values from industry standard

    # Fastener height
    def h_f(z, S, phi): # z -- number of teeth pairs; phi in radians
        return cos(phi / 2) * S * (z - 0.5)


    # This function is to find profile width (l_l) of the upper and bootom parts of fastener 
    def l_l_first_last(z, S, r, gamma, beta, phi): # z -- number of teeth pairs
        l_d1 = S - 2 * (r / sin(gamma / 2) * cos(gamma / 2 - beta + np.radians(90) - phi / 2) - r) + \
            S * sin(beta - gamma) / sin(gamma) * cos(beta - np.radians(90) + phi / 2) + 8 * sin(phi / 2)
        double_delta = 0.05 * S
        l_l1 = l_d1 - double_delta
    
        l_last = l_l1 - 2 * sin(phi / 2) * S * (z - 0.5)
        return [l_l1, l_last]


    # Fir-tree joint approximate volume
    def vol_f(l_u, l_b, h, b): # _u -- upper section; _b -- bottom section; 
        S = (l_u + l_b) / 2 * h
        V = S * b
        R_cm = h / 3 * (2 * l_u +l_b) / (l_b + l_u) # http://ru.solverbook.com/question/centr-tyazhesti-trapecii/
        return  [V, R_cm]


    def P_j(vol_u, vol_f, dens, n, R_u, R_f):    # vol_u -- volume of blade above fastener upper section; 
        vol = vol_u + vol_f                      # vol_f -- volume of fastener; dens -- material density; 
        omega = 2 * np.pi * n / 60               # n = rot. speed [rpm]; R_u --  upper blade part center of mass radius;
        dist = R_u - R_f                         # R_f -- fastener center of mass radius
        R = R_f + dist / (1 + vol_f / vol_u)     # dist -- distance between center of masses of blade and fastener
        return vol * dens * omega**2 * R         # R -- the whole blade center of mass


    def l_h(l_l1, S, r, gamma, beta, phi):
        l_h = l_l1 - 2 * (S * sin(beta - gamma) / 2 / sin(gamma) - r / tan(gamma/2)) * \
              cos(beta + phi/2 - np.radians(90)) + 4 * r * sin(beta + phi/2 - np.radians(90)) - 4 * r
        return l_h


    l_upper = l_l_first_last(z, S, r_func(S), gamma, beta, phi)[0]
    l_lower = l_l_first_last(z, S, r_func(S), gamma, beta, phi)[1]
    fast_height = h_f(z, S, phi)
    fast_vol = vol_f(l_upper, l_lower, fast_height, b)
    CFF = P_j(blade_vol * 10**(-9), fast_vol[0] * 10**(-9), mat_density, n, blade_Rcm * 10**(-3), fast_vol[1] * 10**(-3)) / z / 2 # centrifugal force from the whole blade per 1 tooth


    m1 = (2 * sin(beta - gamma) / sin(gamma) - 0.7 * tan(gamma / 2)**(-1) - 0.05 / cos(psi))*cos(psi)
    m3 = (2 * sin(beta - gamma) / sin(gamma) - 0.35 * tan(gamma / 2)**(-1) - 0.05 / cos(psi))*sin(gamma) / sin(beta - gamma + phi / 2)
    m4 = (2 * sin(beta - gamma) / sin(gamma) - 0.35 * tan(1 / (gamma / 2)) + 0.35 * tan(1 / (beta / 2))) * sin(gamma) / sin(beta - gamma)
    m5 = m1 / 2 / cos(psi) + 0.35 * tan(beta / 2)**(-1) - m4 / 2 * sin(beta - 90) + 0.05 / cos(psi)
    sigma_sm = 1 / m1 * CFF / b / S * 2
    sigma_iz = 6 * m5 / m4**2 / cos(psi) * CFF / b / S * 2
    sigma_sr = 1 / m3 * CFF / b / S * 2
    return [sigma_sm, sigma_iz, sigma_sr]


sigma_sm_dop =  2.3 * 10**8 # Pa [sigma_sm] = 2.3 * 10**8 Khronin (str. 153)
sigma_iz_dop = 2.0 * 10**8
tau_sr_dop = 1.2 * 10**8



#K_sm_max = sigma_sm_dop * b * 10**(-3) * S * 10**(-3) / 2 / CFF
#K_iz_max = sigma_iz_dop * b * 10**(-3) * S * 10**(-3) / 2 / CFF
#K_sr_max = tau_sr_dop * b * 10**(-3) * S * 10**(-3) / 2 / CFF

nonlinear_constraint = NonlinearConstraint(cons_func, 0, [sigma_sm_dop, sigma_iz_dop, tau_sr_dop], jac = '2-point', hess = BFGS())


# 3. Define optimizing function
def optim_func(x):
    gamma = x[0]
    beta = x[1]
    phi = x[2]
    return K_sm(x) + K_iz(x) + K_sr(x)

x0 = np.array([np.radians(20), np.radians(65), np.radians(115)])
res = minimize(optim_func, x0, method = 'trust-constr', jac = "2-point", hess = SR1(), constraints=nonlinear_constraint, options={'verbose': 1}, bounds=bounds)

print("gamma, beta, phi: {}".format(np.degrees(res.x)))
#print("l_l1: {}".format(l_l_first_last(z, S, r_func(S), res.x[0], res.x[1], res.x[2])[0]))
#print("h_l5: {}".format(h_f(z, S, res.x[2])))
#print("l_h1: {}".format(l_h(l_l_first_last(z, S, r_func(S), res.x[0], res.x[1], res.x[2])[0], S, r_func(S), res.x[0], res.x[1], res.x[2])))
#print("r: {}".format(r_func(S)))
#print("r_n: {}".format(2 * r_func(S)))
# print("test l_h1: {}".format(l_h(5.998, 1.8, 0.3, np.radians(55), np.radians(105), np.radians(30))))