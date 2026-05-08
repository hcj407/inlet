import numpy as np
import ast
import os
from matplotlib import pyplot as plt
import subprocess
from inlet.pbr import nor_pressure, obq_pressure, max_pbr
from compair import oblique_shock as obq
from compair import normal_shock as nos
from compair import cone_shock as cone


def coord(mach, delta, arg, ini, dist=0.025, ar=1.41, exp=6):
    '''
    mach : float
        Mach number
    delta : list
        Ramp's angle
    arg : namespace
        Mode and geometry type
    dist : float
        psi step
    throat : float
        Throat length
    ar : float
        Area ratio

    Return
    ------
    x : ndarray
        Intake's coordinates
    '''
    if arg.gtype == '2d':
        stats = []

        machn = mach
        for d in delta[:-1]:
            stats.append(obq_pressure(machn, d))
            machn = stats[-1][0]

        # Normal shock after the last ramp
        mach, p, p0 = nor_pressure(machn)
        stats.append((mach, 90, p, p0))

        # Return mach, beta, pratio, p0ratio
        _, beta, _, _ = np.array(stats).T

        # Convert deg to rad
        d, b = np.deg2rad([delta, beta])
        d0 = np.cumsum(np.insert(d,0,0))[:-1]

    # axi and performance geometry
    else:
        beta = []
        *_, beta1 = cone.solve_cone(mach, delta[0])
        beta.append(beta1)

        d = np.deg2rad(delta)
        b = np.deg2rad(beta)
        d0 = np.cumsum(np.insert(d,0,0))[:-1]

        x1 = 1
        y1 = np.tan(np.deg2rad(delta[0]))

        if len(delta) == 3:
            # Find oblique shock
            xi, yi = x1, y1

            thetas, p0 = [], []
            theta = delta[0]
            while theta < beta1:    
                # Cone 유동 해석
                m1, rho1, p1, p01, beta1, theta = cone.solve(mach, delta[0], theta)
                
                # 2nd ramp에서 유동 꺽임 정도는 같다.
                m2, rho2, p2, p02, beta2 = obq.solve(m1, delta[1])
                beta = theta + beta2

                # 다음 위치 계산
                xi += dist*np.cos(np.deg2rad(beta))
                yi += dist*np.sin(np.deg2rad(beta))
                theta = np.rad2deg(np.rad2deg(np.arctan2(yi, xi)))

                # Normal shock
                _, _, _, p03 = nos.solve(m2)

                thetas.append(theta)
                p0.append(p01*p02*p03)
            b = np.append(b, np.arctan((xi-x1)/(yi-y1)))

        b = np.append(b, np.pi/2)
    
    h = ini.getfloat('Design', 'throat-height')
    
    aa, bb = gr(b, d0, d, h)

    # Solve Ax=b to get p1, p2, p_cowl (non-dimensionalized by h)
    x_ = np.linalg.solve(aa, bb).reshape(2, -1).T

    sum_d = abs(sum(delta))

    # If expansion angle is zero
    if sum_d == 0:
            sum_d = exp

    f = lambda y : (y - x_[-2, 1])/-np.tan(np.deg2rad(sum_d)) + x_[-2, 0]

    if arg.gtype == 'axi':
        y2 = x_[-1, 1] - h*np.cos(np.deg2rad(sum_d))*np.sqrt(ar)
        if y2 < 0:
            d2 = np.array([f(0), 0])
            d3 = np.array([f(0), h*np.sqrt(ar)])
        else:
            y2 = x_[-1, 1] - h*np.cos(np.deg2rad(sum_d))*np.sqrt(ar)
            d2 = np.array([f(y2), y2])
            d3 = np.array([f(y2), x_[-1, 1]])
    else:
        y2 = x_[-1, 1] - h*np.cos(np.deg2rad(sum_d))*ar
        d2 = np.array([f(y2), y2])
        d3 = np.array([f(y2), x_[-1, 1]])
        
    # diff = np.array([d1, d2, d3])
    diff = np.array([d2, d3])

    # Add origin
    x = np.vstack([[0, 0], x_[:-1]])
    x = np.vstack([x, diff])
    x = np.vstack([x, x_[-1]])            

    # Return position
    return x


def dat(ini, points, dmach, delta):
    '''
    Create .dat file from given parameters

    Parameter
    ---------
    ini : INIfile
        Design parameter
    points : ndarray
        Geometry coordinates
    dmach : float
        Design mach number
    delta : list
        Ramp's angle
    Return
    ------
    total : ndarray
    '''
    sect = 'Design'

    # moving the points to the right
    for i in range(len(points)):
        points[i, 0] += 2 

    ramp = points[:-3]
    diffuser = points[-3:-1]
    cowl = points[-1]

    # make cowl lip theta
    theta = ini.getfloat(sect, 'cowl-theta')

    # Check cowl theta to prevent cowl lip blunt shock
    m, _, _, _ = max_pbr(dmach, delta)
    cum_delta = np.cumsum(delta)
    rotate = ini.getfloat(sect, 'cowl-rotate')
    eff = (theta + rotate) - cum_delta[-2]
    _theta_max = obq.theta_max(m[-2]) - 10
    if eff > _theta_max:
        while eff > _theta_max:
            print(f'Cowl angle is too large. theta_max = {_theta_max:.4f} deg')
            theta = float(input('Cowl theta (deg): '))
            eff = (theta + rotate) - cum_delta[-2]
        theta = np.deg2rad(theta)
    else:
        print(f'Cowl angle is okay theta_max = {_theta_max:.4f} deg, cowl lip = {eff:.4f} deg')
        theta = np.deg2rad(theta)
    # _theta_max = obq.theta_max(dmach)
    # eff = theta + rotate
    # if eff > _theta_max:
    #     while eff > _theta_max:
    #         print(f'Cowl angle is too large. theta_max = {_theta_max:.4f} deg')
    #         theta = float(input('Cowl theta (deg): '))
    #         eff = theta + rotate
    #     theta = np.deg2rad(theta)
    # else:
    #     print(f'Cowl angle is okay theta_max = {_theta_max:.4f} deg, cowl lip = {eff:.4f} deg')
    #     theta = np.deg2rad(theta)

    inner_length = ini.getfloat(sect, 'inner-cowl-length')
    outer_length = ini.getfloat(sect, 'outer-cowl-length')

    cowl_in = np.array([cowl[0] + inner_length, cowl[1]])
    cowl_out = np.array([cowl[0] + outer_length*np.cos(theta), cowl[1] + outer_length*np.sin(theta)])

    # rotate the cowl lip
    if rotate := np.deg2rad(ini.getfloat(sect, 'cowl-rotate')):

        dx, dy = cowl_in[0] - cowl[0], cowl_in[1] - cowl[1]
        cowl_in = np.array([cowl[0] + dx*np.cos(rotate),
                            cowl[1] + dx*np.sin(rotate)])

        dx, dy = cowl_out[0] - cowl[0], cowl_out[1] - cowl[1]
        cowl_out = np.array([cowl[0] + dx*np.cos(rotate) - dy*np.sin(rotate),
                          cowl[1] + dx*np.sin(rotate) + dy*np.cos(rotate)])
    # Throat height
    throat = np.sqrt((ramp[-1, 0] - cowl[0])**2 + (ramp[-1, 1] - cowl[1])**2)
    diffuser = np.insert(diffuser, 0, [cowl_in[0], cowl_in[1] - throat], axis=0)

    # If diffuser is higher than outer
    # if cowl_out[1] < diffuser[-1, 1]:
    #     print("Reset your cowl theta")
        # diff = diffuser[-1, 1] - cowl_out[1]
        # cowl_out[1] += diff + 0.2

    # make far field
    far_ = ast.literal_eval(ini.get(sect, 'far'))
    far = np.array([[far_[0], cowl_out[1]], [far_[0], far_[1]], [0, far_[1]], [0, 0]])
    
    aip = ini.getfloat(sect, 'aip') 
    aip = np.array([[aip, diffuser[-2, 1]], [aip, diffuser[-1, 1]]])

    # make diffuser coordinates
    x = ini.getfloat(sect, 'intake-length')
    duct = np.array([[x, diffuser[-2, 1]], [x, diffuser[-1, 1]]])

    # ramp, throat, diffuser, cowl, far
    total = np.vstack([ramp, diffuser[:-1]])
    total = np.vstack([total, aip[0]])
    total = np.vstack([total, duct])
    total = np.vstack([total, aip[1]])
    total = np.vstack([total, diffuser[-1]])
    total = np.vstack([total, cowl_in])
    total = np.vstack([total, cowl])
    total = np.vstack([total, cowl_out])
    total = np.vstack([total, far])

    # plot
    if ini.get('Design', 'mode') == 'performance':
        plt.plot(total[:, 0], total[:, 1], marker='o')
        plt.show()

    path = f'./{ini.get('Design', 'mode')}/{ini.get('Design', 'name')}'
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/{ini.get('Design', 'name')}.dat", "w") as f:
        for x, y, in total:
            f.write(f"{x} {y}\n")

    return total


def geo_2d(ini, points, far=0.2, wall=0.01):
    ''' 
    Create .geo file from given parameters

    Parameter
    ---------
    points : list
        dat file points
    ini : object
        Include design parameters
    far : float
        Far field spacing
    wall : float
        Wall spacing
    '''
    path = f'./{ini.get('Design', 'mode')}/{ini.get('Design', 'name')}'
    os.makedirs(path, exist_ok=True)

    ramp = ini.getint('Design', 'ramp')

    with open(f"{path}/{ini.get('Design', 'name')}.geo", "w") as f:
        f.write(f'far = {far};\n')
        f.write(f'wall = {wall};\n')

        # Points Spacing
        for i, (x, y) in enumerate(points, start=1):
            if ramp+10 < i :  
                f.write(f"Point({i}) = {{{x}, {y}, 0, far}};\n")
            else:
                f.write(f"Point({i}) = {{{x}, {y}, 0, wall}};\n")
                 
        # Intake
        for i in range(1, ramp+15):
            f.write(f"Line({i}) = {{{i}, {i+1}}};\n")
        f.write(f"Line({ramp+15}) = {{{len(points)}, {1}}};\n")

        # Interface
        # entry l, r
        f.write(f"Line({ramp+16}) = {{{ramp+1}, {ramp+10}}};\n")
        f.write(f"Line({ramp+17}) = {{{ramp+10}, {ramp+1}}};\n")
       
        #aip l, r
        f.write(f"Line({ramp+18}) = {{{ramp+4}, {ramp+7}}};\n")
        f.write(f"Line({ramp+19}) =   {{{ramp+7}, {ramp+4}}};\n")

        # Boundary conditions
        numbers = list(range(1, ramp+16)) 

        # virtual surfaces
        virtual = [ramp+16, ramp+17, ramp+18, ramp+19]      
        ramp_ = numbers[0:ramp]
        far = list(range(ramp+12, ramp+16)) + [ramp+5]
        wall = [n for n in numbers if n not in far if n not in ramp_]

        if ini.get('Design', 'bezier') == 'O' :
            f.write(f"Bezier({ramp+25}) = {{{ramp+1}, {ramp+2}, {ramp+3},{ramp+4}}};\n")  # ramp+3, ramp+4 line 사라짐
            f.write(f"Bezier({ramp+26}) = {{{ramp+7}, {ramp+8}, {ramp+9}, {ramp+10}}};\n")
            f.write(f"Curve Loop(1) = {{{ramp+18}, {ramp+17}, {ramp+26}, {ramp+25}}};\n")
            wall.append(ramp+25)
            wall.append(ramp+26)
            for i in [1,2,3,7,8,9]:
                wall.remove(ramp+i)
        else:
            f.write(f"Curve Loop(1) = {{{ramp+17}, {ramp+18}, {ramp+1}, {ramp+2}, {ramp+3}, {ramp+7}, {ramp+8}, {ramp+9}}};\n")

        f.write(f"Curve Loop(2) = {{{ramp+4}, {ramp+5}, {ramp+6}, {ramp+19}}};\n")
        f.write("Plane Surface(2) = {2};\n")

        num = list(np.arange(ramp+10, ramp+17)) + ramp_
        f.write(f"Curve Loop(3) = {{{', '.join(map(str, num))}}};\n")
        f.write("Plane Surface(3) = {3};\n")

        f.write("Plane Surface(1) = {1};\n")

        f.write('Physical Surface("fluid", 50) = {1, 2, 3};\n')

        # Boundary conditions
        f.write(f'Physical Curve("wall", 101) = {{{', '.join(map(str, wall))}}};\n')
        f.write(f'Physical Curve("pout", 102) = {{{ramp+5}}};\n')
        f.write(f'Physical Curve("outflow", 103) = {{{ramp+12}}};\n')
        f.write(f'Physical Curve("far", 104) = {{{ramp+13}}};\n')
        f.write(f'Physical Curve("inflow", 105) = {{{ramp+14}}};\n')
        f.write(f'Physical Curve("sym", 106) = {{{ramp+15}}};\n')
        f.write(f'Physical Curve("ramp", 107) = {{{', '.join(map(str, ramp_))}}};\n')
        f.write(f'Physical Curve("periodic_aip_l", 208) = {{{virtual[2]}}};\n')
        f.write(f'Physical Curve("periodic_aip_r", 209) = {{{virtual[3]}}};\n')
        f.write(f'Physical Curve("periodic_entry_l", 210) = {{{virtual[0]}}};\n')
        f.write(f'Physical Curve("periodic_entry_r", 211) = {{{virtual[1]}}};\n')


def simple_dat(ini, points, h=1.0):
    '''
    Create .dat file from given parameters

    Parameter
    ---------
    ini : object
        Include design parameters

    Return
    ------
    total : list of tuple
    '''
    sect = 'Design'
    # moving the points to the right
    for i in range(len(points)):
        points[i] = (points[i, 0]+2, points[i, 1])

    ramp = points[0:-3]
    cowl = points[-1]

    theta = np.deg2rad(ini.getfloat(sect, 'cowl-theta'))
    outer_length = ini.getfloat(sect, 'outer-cowl-length')

    cowl_out = np.array([cowl[0] + outer_length*np.cos(theta)*h, cowl[1] + outer_length*np.sin(theta)])

    # make far field
    far_ = ast.literal_eval(ini.get(sect, 'far'))
    far = np.array([[far_[0]*h, cowl_out[1]], [far_[0]*h, far_[1]*h], [0, far_[1]*h], [0, 0]])

    # ramp, throat, cowl, far
    total = np.vstack([ramp, cowl])
    total = np.vstack([total, cowl_out])
    total = np.vstack([total, far])

    path = f'./{ini.get('Design', 'mode')}/{ini.get('Design', 'name')}'
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/{ini.get('Design', 'name')}.dat", "w") as f:
        for x, y, in total:
            f.write(f"{x} {y}\n")

    return total


def simple_2d(ini, points, far=0.1, wall=0.01):
    ''' 
    Create .geo file from given parameters

    Parameter
    ---------
    points : list
        dat file points
    ini : object
        Include design parameters
    far : float
        Far field spacing
    wall : float
        Wall spacing
    '''
    path = f'./{ini.get('Design', 'mode')}/{ini.get('Design', 'name')}'
    os.makedirs(path, exist_ok=True)

    ramp = ini.getint('Design', 'ramp')
    with open(f"{path}/{ini.get('Design', 'name')}.geo", "w") as f:
        f.write(f'far = {far};\n')
        f.write(f'wall = {wall};\n')

        # Points Spacing
        for i, (x, y) in enumerate(points, start=1):
            if ramp+2 < i :  
                f.write(f"Point({i}) = {{{x}, {y}, 0, far}};\n")
            else:
                f.write(f"Point({i}) = {{{x}, {y}, 0, wall}};\n")
                 
        # Intake
        for i in range(1, len(points)):
            f.write(f"Line({i}) = {{{i}, {i+1}}};\n")
        f.write(f"Line({len(points)}) = {{{len(points)}, {1}}};\n")

        f.write(f"Curve Loop(1) = {{{', '.join(map(str, range(1, len(points)+1)))}}};\n")
        f.write("Plane Surface(1) = {1};\n")

        f.write('Physical Surface("fluid", 50) = {1};\n')

        # Boundary conditions
        f.write(f'Physical Curve("ramp", 101) = {{{', '.join(map(str, range(1, ramp+1)))}}};\n')
        f.write(f'Physical Curve("pout", 102) = {{{ramp+1}}};\n')
        f.write(f'Physical Curve("wall", 103) = {{{ramp+2}, {ramp+3}}};\n')
        f.write(f'Physical Curve("outflow", 104) = {{{ramp+4}}};\n')
        f.write(f'Physical Curve("far", 105) = {{{ramp+5}}};\n')
        f.write(f'Physical Curve("inflow", 106) = {{{ramp+6}}};\n')
        f.write(f'Physical Curve("sym", 107) = {{{ramp+7}}};\n')


def gmsh(ini):
    ''' 
    Create .gmsh file

    Parameter
    ---------
    ini : object
    '''
    mode = ini.get('Design', 'mode')
    name = ini.get('Design', 'name')

    gmsh_path = ini.get('Design', 'gmsh')

    path = f'./{mode}/{name}'

    try:
        os.system(f"gmsh ./{path}/{name}.geo -2 -o ./{path}/{name}.msh -v 0")
    except:
        gmsh = input("Input your gmsh path: ")
        os.system(f"{gmsh} ./{path}/{name}.geo -2 -o ./{path}/{name}.msh -v 0")


def gr(b, d0, d, h):
    '''
    Geometric relations

    Parameter
    ---------
    b : ndarray
        Beta
    d0 : ndarray
        Cumulative delta
    d : ndarray
        Delta
    h : float
        Throat height

    Return
    ------
    aa : ndarray

    bb : ndarray

    '''
    # Geometric relations
    n = len(b)
    n1 = n - 1
    aa = np.zeros((2*n, 2*n))

    # Make matrix
    aa[0, 0] = -np.tan(d0[0] + d[0])
    aa[0, n] = 1
    aa[2*n1-1, n-1] = -np.tan(d0[0] + b[0])
    aa[2*n1-1, 2*n-1] = 1
    aa[2*n-2, n-2] = 1
    aa[2*n-2, n-1] = -1
    aa[2*n-1, 2*n-2] = -1
    aa[2*n-1, 2*n-1] = 1

    bb = np.zeros(2*n)
    # bb[2*n-2] = np.sin(sum(d))*h
    # bb[2*n-1] = np.cos(sum(d))*h
    bb[2*n-2] = 0
    bb[2*n-1] = h

    for i in range(1, n1):
        aa[i, i-1] = -np.tan(d0[i] + d[i])
        aa[i, i] = -aa[i, i-1]
        aa[i, n+i-1] = 1
        aa[i, n+i] = -1

        aa[2*n1-1-i, i-1] = -np.tan(d0[i] + b[i])
        aa[2*n1-1-i, n-1] = -aa[2*n1-1-i, i-1]
        aa[2*n1-1-i, n+i-1] = 1
        aa[2*n1-1-i, 2*n-1] = -1

    return aa, bb
