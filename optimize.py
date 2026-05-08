# -*- coding: utf-8 -*-
from pybaram.api.io import import_mesh, export_soln
from pybaram.backends import get_backend
from pybaram.integrators import get_integrator
from pybaram.readers.native import NativeReader
from pybaram.utils.mpi import mpi_init
from inlet.initialize import Mesh
from inlet.postprocess import Post
from inlet.initialize import make_cfg
import itertools
import numpy as np
import os

def run(arg):
    '''
    Paraemter
    ---------
    arg : namespace
    '''
    # Get design condition
    obj = Mesh(arg)
    
    # Solve
    solve(arg, obj)
    
def solve(arg, obj, iter=5, gap=1.0):
    #TODO: Input initial theta and change
    # Initial theta
    iter = obj.ini.getint('Design', 'iter')
    x = arg.delta
    x0 = np.empty((len(x)-1, iter))
    eps = 0.001

    for i in range(len(x)-1):
        x0[i] = [x[i] + a for a in np.arange(-gap, gap+eps, gap*2/(iter-1))]

    pairs = list(itertools.product(*x0))
    exp = np.ones(iter**(len(x[:-1])))*x[-1]
    delta = np.column_stack((pairs, exp))

    pt = []
    angle = []
    i = 0 

    for d in delta:
        i += 1

        d = list(d)

        # Make gmsh
        obj.make_gmsh(d)

        # Make and change directory
        os.makedirs(f"./{obj.mode}/{obj.name}/{i}", exist_ok=True)
        os.chdir(f"./{obj.mode}/{obj.name}/{i}")
        _d = [float(x) for x in d]
        # Import mesh
        import_mesh(f"../{obj.name}.msh", f"./{_d}.pbrm")
        print("\nImported GMSH to pyBaram mesh")

        # Read mesh
        msh = NativeReader(f"{_d}.pbrm")

        # Construct ini file from condition
        cfg = make_cfg(obj)

        # Build MPI comm and backend
        comm = mpi_init()
        backend = get_backend('cpu', cfg)

        # Get integrator
        integrator = get_integrator(backend, cfg, msh, None, comm)
        
        # Run Integrator
        try:
            integrator.run()
        except ZeroDivisionError:
            print("ZeroDivisionError, Please retry")

        # Pybaram export
        try:
            export_soln(f'{_d}.pbrm', f'out-{integrator.iter}.pbrs', f'out_{i}.vtu')
        except FileNotFoundError:
            print("No output file")

        # Postprocessing
        post = Post(integrator, cfg)
        _, pt_, pr = post._surfpfx("pout", obj)

        #TODO: calculate normal shock 
        print(f"Pressure ratio is {pt_:.3f}")

        pt.append(pt_)
        angle.append(d)

        os.chdir("../../../")

    idx = pt.index(max(pt))
    post.pt_scatter(pt, obj)

    print(f"Total pressure max:{pt[idx]:.3f}")
    print(f"Ramp angle:{angle[idx][0]:.3f}, {angle[idx][1]:.3f}")
    print(f"Look {idx+1} directory")
    
    return pt
