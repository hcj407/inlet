# -*- coding: utf-8 -*-
from pybaram.api.io import import_mesh, export_soln
from pybaram.backends import get_backend
from pybaram.integrators import get_integrator
from pybaram.readers.native import NativeReader
from pybaram.utils.mpi import mpi_init
from inlet.initialize import Mesh
from inlet.postprocess import Post
from inlet.initialize import make_cfg
import os
import numpy as np


def run(arg):
    '''
    Paraemter
    ---------
    arg : namespace
    '''
    # Get design condition
    mode = arg.cmd
    
    # Create grid
    obj = Mesh(arg)
    obj.make_gmsh(arg.delta)
    name = obj.name

    # Import mesh
    print("Imported GMSH to pyBaram mesh")
    import_mesh(f"./{mode}/{name}/{name}.msh", f"./{mode}/{name}/{name}.pbrm")

    # Solve
    solve(f"./{mode}/{name}/{name}.pbrm", obj)


def solve(mshf, obj):
    # Read mesh
    msh = NativeReader(mshf)

    pbr_min, pbr_max, step = obj.pbr()

    os.chdir(f"./{obj.mode}/{obj.name}")
    eps = 1e-7

    for pbr in np.arange(pbr_min, pbr_max+eps, step):
       
        os.makedirs(f"./pbr{pbr:.3f}", exist_ok=True)
        os.chdir(f"./pbr{pbr:.3f}")

        # Construct ini file from condition
        cfg = make_cfg(obj, pbr=pbr)

        # Build MPI comm and backend
        comm = mpi_init()
        backend = get_backend('cpu', cfg)

        # Get integrator
        integrator = get_integrator(backend, cfg, msh, None, comm)
        
        try:
            # Run Integrator
            integrator.run()
        except ZeroDivisionError:
            print("ZeroDivisionError, Please retry")
        
        # Pybaram export
        try:
            export_soln(f'../{obj.name}.pbrm', f'./out-{integrator.iter}.pbrs', 'out.vtu')
        except FileNotFoundError:
            print("No output file")

        # Post
        post = Post(integrator, cfg)
        post.step(pbr, obj)
        os.chdir(f"../")

    # Plot PR curve
    if post is not None:
        post.plot()
    else:
        print("No valid 'post' created; skip plotting.")