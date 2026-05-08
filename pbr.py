# -*- coding: utf-8 -*-
from compair import oblique_shock as obq
from compair import normal_shock as nos
from compair import isentropic as isen
from compair import nozzle
import numpy as np

def find_pbr(mach, angle, area_ratio=1.41, gamma=1.4):
    '''
    Find back pressure ratio

    Parameter
    ---------
    mach : float
        Condition mach number
    angle: list
        Angle of the ramp [deg]
    
    Return
    ------
    pbr_max : float
        Maximum back pressure ratio
    pbr_min : float
        Minimum back pressure ratio
    '''
    m = mach

    # Find maximum back pressure
    machs, _, pratio, _ = max_pbr(m, angle)

    if machs[-2] < 1:
        raise ValueError("The exit flow is not supersonic. Please reduce the number of ramps or the angle of the ramps.")
    else:
        # Total pressure ratio
        p0 = isen.p0_p(machs[-1])
        # At diffuser, subsonic
        mach_ = nozzle.mach_by_area_ratio(area_ratio, gamma=1.4, x0=0.1)
        p0_p = isen.p0_p(mach_)
        pratio = np.append(pratio, p0/p0_p)
        pbr_max = np.prod(pratio)

    # Find minimum back pressure
    machs, _, pratio, _ = min_pbr(m, angle)
    # At diffuser, supersonic
    mach_ = nozzle.mach_by_area_ratio(area_ratio, gamma=1.4, x0=1+1e-8)
    pratio_ = isen.p0_p(machs[-1])/isen.p0_p(mach_)
    pratio = np.append(pratio, pratio_)

    mach, p, p0 = nor_pressure(mach_)
    machs = np.append(machs, mach)
    pratio = np.append(pratio, p)
    pbr_min = np.prod(pratio)

    return np.round(pbr_max, 3), np.round(pbr_min, 3)


def obq_pressure(mach, theta, gamma=1.4):
    if mach > 1:
        mach2, _, p2_p1, p02_p01, beta = obq.solve(mach, theta, gamma)
    else:
        mach2, beta, p2_p1, p02_p01 = mach, 90.0, 1.0, 1.0    

    return mach2, beta, p2_p1, p02_p01


def nor_pressure(mach, gamma=1.4):
    if mach > 1:
        mach2, _, p2_p1, p02_p01 = nos.solve(mach, gamma)
    else:
        mach2, p2_p1, p02_p01 =  mach, 1.0, 1.0

    return mach2, p2_p1, p02_p01


def max_pbr(mach, delta):
    # Mutli stage ramp
    stats = []

    machn = mach
    for d in delta:
        stats.append(obq_pressure(machn, d))
        machn = stats[-1][0]

    # Normal shock after the last ramp
    mach, p, p0 = nor_pressure(machn)
    stats.append((mach, 90, p, p0))

    # Return mach, beta, pratio, p0ratio
    return np.array(stats).T


def min_pbr(mach, delta):
    # Mutli stage ramp
    stats = []

    machn = mach
    for d in delta:
        stats.append(obq_pressure(machn, d))
        machn = stats[-1][0]

    # Return mach, beta, pratio, p0ratio
    return np.array(stats).T
