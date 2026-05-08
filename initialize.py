# -*- coding: utf-8 -*-
from inlet.make import coord, dat, geo_2d, simple_2d, simple_dat, gmsh
from inlet.pbr import find_pbr
import numpy as np
from pybaram.inifile import INIFile
import ast

class Mesh:
    
    def __init__(self, arg):
        """
        Parameters
        ----------
        name : string
               File name
        arg : namespace
            mode, geometry type
        coords : tuples 
                 Geometry coordinates(ramp, diffuser, cowl)
        """

        self.arg = arg
        self.mode = arg.cmd
        self.gtype = arg.gtype

        self.ini = ini = INIFile('ini.ini')

        self.mach = ini.getfloat('Design', 'mach')
        self.dmach = ini.getfloat('Design', 'design-mach')
        self.name = ini.get('Design', 'name')


    def make_gmsh(self, delta):
        '''
        Create .dat, .geo, .gmsh file

        delta : list
            Ramp's angle and expansion angle
        '''
        if delta != None:
            self.delta = delta
            self.coords = coord(self.dmach, delta, self.arg, self.ini)
            self.ramp = len(delta) - 1
        else:
            # read .ini file
            self.coords = np.array(ast.literal_eval(self.ini.get('Design', 'coords')), dtype=float)
            self.ramp, self.delta = self.coord_theta(self.coords)


        self.ini._cfg['Design'].update({
            'delta' : ', '.join(map(str, self.delta)),
            'ramp' : f'{self.ramp}',
        })

        if self.mode == "performance":
            total = dat(self.ini, self.coords, self.dmach, self.delta)
            geo_2d(self.ini, total)
            gmsh(self.ini)
        # Optimize
        else:
            total = simple_dat(self.ini, self.coords)
            simple_2d(self.ini, total)
            gmsh(self.ini)


    def pbr(self, ratio=0.8, stage=3):
        '''
        Find back pressure ratio range

        Parameters
        ----------
        ratio : float
            Range ratio
        stage : int
            Number of step
        
        Return
        ------
        pbr_max : float
            When throat normal shock
        pbr_min : float
            When diffuser normal shock

        '''
        stage = self.ini.getint('Design', 'stage')
        if self.arg.pbr == None:
            delta = self.ini.getlist('Design', 'delta')

            pbr_max, pbr_min = find_pbr(self.mach, delta[:-1])
            pbr_max -= 0.5

            range = ratio * (pbr_max - pbr_min)
            pbr_min += 0.5*(1-ratio)*range
            pbr_max -= 0.5*(1-ratio)*range
            step = range / stage
        else:
            pbr_min = self.arg.pbr[0]
            pbr_max = self.arg.pbr[1]
            step = (pbr_max - pbr_min) / stage
            
        return pbr_min, pbr_max, step
    

    def coord_theta(self, coords, exp=6):
        '''
        Convert coordinates to ramp's angle

        Parameters
        ----------
        coords : ndarray
            Intake's coordinates
        '''

        ramp = len(coords) - 4

        delta = []
        _delta = 0
        for i in range(0, ramp):
            _delta = np.arctan(coords[i+1, 1] - coords[i, 1])/(coords[i+1, 0] - coords[i, 0]) - _delta
            delta.append(np.rad2deg(_delta))

        delta.append(-(sum(delta) + exp))

        return ramp, delta


def make_cfg(obj, pbr=1.0, xd=7.0, cfl=10, max_iter=20000):
    """
    Custom Config file generator

    Parameters
    ----------
    obj : object
        Design parameters
    mach : float
        Mach number
    pbr : float
        Back pressure ratio w.r.t free-stream pressure
    xd : float
        Breaking point of free-stream condition
    cfl : float
        CFL number
    max_iter : int
        Max iteration

    Return
    ------
    ini : object
        INIFIle object
    """
    # Construct INI object
    ini = INIFile()
    cfg = ini._cfg

    # Constant variable
    gamma = 1.4
    rho = 1.0
    u = obj.mach
    p = 1.0/gamma
    dmach = obj.dmach

    # Configuration
    cfg['constants'] = {'gamma' : gamma, 'uf' : u, 'design-uf' : dmach}
    cfg['solver'] = {
        'system' : 'euler', 'order' : 2, 'limiter' : 'mlp-u2', 'u2k' : 1.0, 
        'riemann-solver' : 'rotated-roem'
    }
    cfg['solver-time-integrator'] = {
        'mode' : 'steady', 'cfl' : cfl, 'stepper' : 'lu-sgs',
        'max-iter' : max_iter, 'tolerance' : 1e-8
    }

    # If axi-symmetric case add source term
    if obj.gtype == 'axi':
        cfg['solver-source-terms'] = ({
            'rho' : '-rhov/y',
            'rhou' : '-rhou*rhov/rho/y',
            'rhov' : '-rhov*rhov/rho/y',
            'E' : '-(gamma*E + (1-gamma)/2*(rhou**2 + rhov**2)/rho)*rhov/rho/y'
            })
    
    cfg['soln-ics'] = {
        'hl' : '0.5*(1 - tanh(1e8*(x-{})))'.format(xd),
        'hr' : '1 - %(hl)s',
        'rho' : rho, 'p' : p, 'v' : 0.0,
        'u' : '{}*%(hl)s + 0.2*{}%(hr)s'.format(u, u),
    }
    cfg['soln-bcs-inflow'] = {
        'type' : 'sup-in', 
        'rho' : rho, 'u' : u, 'v' : 0, 'p' : p
    }
    cfg['soln-bcs-outflow'] = {'type' : 'sup-out'}
    cfg['soln-bcs-wall'] = {'type' : 'slip-wall'}
    cfg['soln-bcs-far'] = {'type' : 'far',
                        'rho' : rho, 'u' : u, 'v' : 0, 'p' : p}
    cfg['soln-bcs-ramp'] = {'type' : 'slip-wall'}
    cfg['soln-bcs-sym'] = {'type' : 'slip-wall'}
    
    # If optimization, no need back pressure ratio
    if obj.mode == 'performance':
        cfg['soln-bcs-pout'] = {'type' : 'sub-outp', 'p' : pbr*p}
        if obj.gtype == 'axi':
            cfg['soln-plugin-surface-aip_l'] = { 'iter-out' : 100,
                                            'items' : 'p0, mdot',
                                            'p0' : 'p*(1+ (gamma-1)/2*(u**2 + v**2)/(gamma*p/rho))**(gamma/(gamma-1))',
                                            'mdot' : 'rho*(u*nx+v*ny)*y'}
            cfg['soln-plugin-surface-aip_r'] = { 'iter-out' : 100,
                                                'items' : 'p0, mdot',
                                                'p0' : 'p*(1+ (gamma-1)/2*(u**2 + v**2)/(gamma*p/rho))**(gamma/(gamma-1))',
                                                'mdot' : 'rho*(u*nx+v*ny)*y'}
            cfg['soln-plugin-surface-entry_l'] = { 'iter-out' : 100,
                                            'items' : 'p0, mdot',
                                            'p0' : 'p*(1+ (gamma-1)/2*(u**2 + v**2)/(gamma*p/rho))**(gamma/(gamma-1))',
                                            'mdot' : 'rho*(u*nx+v*ny)*y'}
            cfg['soln-plugin-surface-entry_r'] = { 'iter-out' : 100,
                                                'items' : 'p0, mdot',
                                                'p0' : 'p*(1+ (gamma-1)/2*(u**2 + v**2)/(gamma*p/rho))**(gamma/(gamma-1))',
                                                'mdot' : 'rho*(u*nx+v*ny)*y'}
            cfg['soln-plugin-surface-pout'] = { 'iter-out' : 100,
                                        'items' : 'p0, mdot',
                                        'p0' : 'p*(1+ (gamma-1)/2*(u**2 + v**2)/(gamma*p/rho))**(gamma/(gamma-1))',
                                        'mdot' : 'rho*(u*nx+v*ny)*y'}
        else:
            cfg['soln-plugin-surface-aip_l'] = { 'iter-out' : 100,
                                            'items' : 'p0, mdot',
                                            'p0' : 'p*(1+ (gamma-1)/2*(u**2 + v**2)/(gamma*p/rho))**(gamma/(gamma-1))',
                                            'mdot' : 'rho*(u*nx+v*ny)'}
            cfg['soln-plugin-surface-aip_r'] = { 'iter-out' : 100,
                                                'items' : 'p0, mdot',
                                                'p0' : 'p*(1+ (gamma-1)/2*(u**2 + v**2)/(gamma*p/rho))**(gamma/(gamma-1))',
                                                'mdot' : 'rho*(u*nx+v*ny)'}
            cfg['soln-plugin-surface-entry_l'] = { 'iter-out' : 100,
                                            'items' : 'p0, mdot',
                                            'p0' : 'p*(1+ (gamma-1)/2*(u**2 + v**2)/(gamma*p/rho))**(gamma/(gamma-1))',
                                            'mdot' : 'rho*(u*nx+v*ny)'}
            cfg['soln-plugin-surface-entry_r'] = { 'iter-out' : 100,
                                                'items' : 'p0, mdot',
                                                'p0' : 'p*(1+ (gamma-1)/2*(u**2 + v**2)/(gamma*p/rho))**(gamma/(gamma-1))',
                                                'mdot' : 'rho*(u*nx+v*ny)'}
            cfg['soln-plugin-surface-pout'] = { 'iter-out' : 100,
                                        'items' : 'p0, mdot',
                                        'p0' : 'p*(1+ (gamma-1)/2*(u**2 + v**2)/(gamma*p/rho))**(gamma/(gamma-1))',
                                        'mdot' : 'rho*(u*nx+v*ny)'}
    else:
        cfg['soln-bcs-pout'] = {'type' : 'sup-out'}

    cfg['soln-plugin-writer'] = {'name' : f'./out-{{n}}', 'iter-out' : 5000}
    cfg['soln-plugin-stats'] = {'iter-out' : 100}

    return ini
