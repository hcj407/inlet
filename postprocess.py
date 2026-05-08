# -*- coding: utf-8 -*-
from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from compair import normal_shock as nos

class Post:

    hist = {
            "mdot_aip": [],   
            "pt_aip":   [],   
            "mfr":      [],   
            "pr":       [],   
            "cf_x":     [],
            "pbr":     [],   
}

    def __init__(self, intg, cfg):
        self.cfg = cfg
        self.sys = intg.sys
        self.curr_idx = intg._curr_idx

        self.eles = list(self.sys.eles)
        self.bcmap = {k.bctype: k for k in self.sys.bint}
        self.vcmap = {k.bctype: k for k in self.sys.vint}


    def step(self, pbr, obj) -> Tuple[float, float, float, float, float]:
        '''
        Accumulate and save properties
        '''

        mdot, pt, mfr, pr = self._vsurfint("aip", obj)
        cf_x, _ = self._surfpfx("ramp", obj)

        # 누적 저장(스칼라로 변환 보장)
        self.hist["mdot_aip"].append(float(mdot))
        self.hist["pt_aip"].append(float(pt))
        self.hist["mfr"].append(float(mfr))
        self.hist["pr"].append(float(pr))
        self.hist["cf_x"].append(float(cf_x))
        self.hist["pbr"].append(float(pbr))

        print("At AIP")
        print(f"mdot={mdot:.3f}, pt={pt:.3f}")   
        print(f"pr={pr*100.0:.3f}%, MFR={mfr*100.0:.3f}%")
        print("cf_x={}\n".format(cf_x))


    def _vsurfint(self, name, obj):
        '''
        Calculate virtual surface's properties

        Parameter
        ---------
        name : string
            Virtual surface name
        obj : object
            Design parameters

        Return
        ------
        mdot : float
            Mass flow
        pt : float
            Total pressure
        mfr : float
            Mass flow rate
        pr : float
            Pressure recovery ratio
        '''
        vl = self.vcmap[f"{name}_l"]
        vr = self.vcmap[f"{name}_r"]

        # Gamma
        gamma = self.cfg.getfloat('constants', 'gamma')
        mach  = self.cfg.getfloat('constants', 'uf')

        # Noraml and length of edge
        nvec = vl._vec_snorm           # shape: (ndims, nfpts)
        nmag = vl._mag_snorm           # shape: (nfpts,)

        if np.linalg.norm(vl.xf - vr.xf) > 1e-10:
            print("Re-ordering is needed (vl.xf != vr.xf)")

        # Parse element index
        fidxl, eidxl, _ = vl._lidx
        fidxr, eidxr, _ = vr._lidx

        # Parse conservative variables at both sides
        ul = self.eles[0].upts[self.curr_idx][:, eidxl]  # shape: (nvars, nfpts)
        ur = self.eles[0].upts[self.curr_idx][:, eidxr]
        
        # Calculate mdot and total pressure        
        um = 0.5 * (ul + ur)
        rho, p, u, v = self.eles[0].conv_to_prim(um, self.cfg)

        a = np.sqrt(gamma * p / rho)
        M = np.sqrt(u**2 + v**2) / a

        if obj.gtype == 'axi':
            y = self.eles[0].xf[fidxl, eidxl].T[1]
            mdot = np.sum((rho*u*nvec*nmag*y)[0])*2*np.pi
            mfr = mdot/(mach*np.pi*obj.coords[-1, -1]**2)
        else:
            mdot = np.sum((rho*u*nvec*nmag)[0])
            mfr = mdot/(mach*obj.coords[-1, -1])
        
        # Area average
        pt = np.sum(p * (1 + (gamma-1)*M**2/2)**(gamma/(gamma-1))*nmag) / np.sum(nmag)
        pt_inf = (1 + 0.5*(gamma - 1)*mach**2)**(gamma/(gamma-1))/gamma

        pr = pt/pt_inf

        return mdot, pt, mfr, pr


    def _surfpfx(self, name, obj, p0=1/1.4):
        '''
        x-direction surface pressure force

        Parameter
        ---------
        name : string
            Boundary surface name
        p0 : float
            Total pressure (Nondimensional)

        Return
        ------
        cx_f : float
            surface pressure force
        pt : float
        '''
        ref_len = obj.ini.getfloat('Design', 'reference-length')
        bc = self.bcmap[f"{name}"]

        if bc is None:
            return 0.0

        gamma = self.cfg.getfloat('constants', 'gamma')

        # Noraml and length of edge
        norm = bc._vec_snorm * bc._mag_snorm
        nmag = bc._mag_snorm

        # Parse element index
        _, eidxl, _  = bc._lidx

        # Parse conservative variables
        um = self.eles[0].upts[self.curr_idx][:, eidxl]

        rho, p, u, v = self.eles[0].conv_to_prim(um, self.cfg)
        
        pt = np.sum(p * (1 + (gamma-1)*u**2/2)**(gamma/(gamma-1))*nmag) / np.sum(nmag)
        cx_f = float(np.sum((p - p0) * norm[0])) / ref_len

        # If optimize mode, compute normal shock(Theory)
        if obj.mode == 'optimize':
            M = np.sqrt((u**2+v**2)/(gamma*p/rho))
            m, rho, p, pt = nos.solve(M)
            p0_inf = 1/1.4*((1+gamma-1)*M**2/2)**(gamma/(gamma-1))
            pr = np.sum(p * (1 + (gamma-1)*m**2/2)**(gamma/(gamma-1))*nmag/p0_inf) / np.sum(nmag) * 100
            pt = np.sum(pt*nmag)/np.sum(nmag)

            return cx_f, pt, pr

        return cx_f, pt


    def csv(self, name, obj, gamma=1.4, idx=-20):
        '''
        Calculate virtual surface's properties

        Parameter
        ---------
        name : string
            Virtual surface name
        obj : object
            Design parameters
        gamma : float
            Specific heat ratio
        idx : int
            Iteration average

        Return
        ------
        mdot : float
            Mass flow
        pt : float
            Total pressure
        mfr : float
            Mass flow rate
        pr : float
            Pressure recovery ratio
        '''
        aip_l = pd.read_csv(f"./surface_{name}_l.csv")
        aip_r = pd.read_csv(f"./surface_{name}_r.csv")

        mdot = abs(0.5*(aip_l["sum_mdot"] - aip_r["sum_mdot"]))[idx:].mean()
        pt = abs(0.5*(aip_l["avg_p0"] + aip_r["avg_p0"]))[idx:].mean()
        mach = obj.mach

        if obj.mode == 'axi':
            mdot = abs(0.5*(aip_l["sum_mdot"] - aip_r["sum_mdot"]))[idx:].mean()*2*np.pi
            mfr = mdot/(mach*np.pi*obj.coords[-1, -1]**2)
        else:
            mfr = mdot/(mach*obj.coords[-1, -1])
        
        pt_inf = (1 + 0.5*(gamma - 1)*mach**2)**(gamma/(gamma-1))/gamma

        pr = pt/pt_inf

        return mdot, pt, mfr, pr


    def plot(self):
        '''
        Plot pressure ratio recovery curve
        '''
        x = self.hist["mfr"]
        y = self.hist["pr"]

        total = np.vstack((self.hist["pbr"], x))
        total = np.vstack((total, y)).T

        np.savetxt("./PR-curve.csv", total, fmt="%.10f", header='Back preesure ratio,MFR,PR', delimiter=',', comments='')
        
        plt.style.use('ggplot')
        plt.rcParams['figure.dpi'] = 150

        plt.plot(x, y)
    
        for i in range(len(x)):
                plt.plot(x[i], y[i], marker='o', label=f"pbr={self.hist["pbr"][i]:.3f}")
        
        plt.xlabel("MFR")
        plt.ylabel("PR")
        plt.ylim(0.3, 1.1)
        plt.xlim(0.2, 1.1)
        plt.legend()
        plt.savefig(f"./PR-curve.png")
        plt.show(block=True)


    def pt_scatter(self, pt, obj):
        '''
        Plot pressure ratio recovery curve
        '''
        idx = pt.index(max(pt))
        x = range(1, len(pt)+1)
        
        plt.style.use('ggplot')
        plt.rcParams['figure.dpi'] = 150

        plt.close('all')
        fig, ax = plt.subplots(figsize=(10,7))
        ax.scatter(x, pt, marker='o', color='b') 
        ax.plot([x[idx]], [pt[idx]], marker='o', color='b', linestyle='none')
        ax.set_ylabel(r"$P_{t,max}$") 
        fig.tight_layout()
        fig.savefig(f"./{obj.mode}/{obj.name}/Pt_max.png", dpi=200, bbox_inches="tight")
        plt.show() 