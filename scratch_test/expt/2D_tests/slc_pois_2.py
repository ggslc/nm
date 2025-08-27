import petsc4py
import sys
import jax.numpy as jnp
petsc4py.init(sys.argv)
from petsc4py import PETSc

comm = PETSc.COMM_WORLD

opt_db = PETSc.Options()

#grid dims
nx = opt_db.getInt('nx',4)
ny = opt_db.getInt('ny',nx)
dx,dy = 1.0/nx, 1.0/ny
dxdy = dx*dy

# global matrix size
nDoF = nx * ny

def grid_ij(row):
    return (row % nx, row // nx)

def grid_xy(row):
    i,j = grid_ij(row)
    return i*dx - 0.5, j*dy - 0.5


A = PETSc.Mat()
A.create(comm=comm)

# leave the row decomposition up to PETSc 
A.setSizes(((PETSc.DECIDE, nDoF), (PETSc.DECIDE,nDoF)))
A.setType(PETSc.Mat.Type.AIJ)
A.setPreallocationNNZ(5) # 9 point stencil - 5 in this case but I have my reasons...

x, b = A.createVecs()
print (f'xv: size {x.size}, local size {x.local_size}, {x.owner_range}')


def beta(x,y):

    r = (x**2 + y**2)**0.5
    return 1.0e+4* (1.0 + jnp.cos(r));

def rhs(x,y):
    return 1.0+4;


#matrix assembly
Al = None
row_start, row_end = A.getOwnershipRange()
for row in range(row_start, row_end):

    i, j = grid_ij(row)
    
    P = row
    E = row + 1
    W = row - 1
    S = row - nx
    N = row + nx

    
#    print (row, i, j)
    ap = 0.0
    if j > 0:
        A[P,S] = -1.0 # south
        ap += 1.0
    if j < ny - 1:
        A[P,N] = -1.0 # north
        ap += 1.0
    if i > 0:
        A[P,W] = -1.0 # west
        ap += 1.0
    if i < nx - 1:
        A[P,E] = -1.0 # east
        ap += 1.0 
    
    A[P,P] = ap + 1.0 * dxdy

    b[row] = rhs(*grid_xy(row)) * dxdy



#inter-rank stuff
A.assemblyBegin()
A.assemblyEnd()
b.assemblyBegin()
b.assemblyEnd()



A.viewFromOptions('-view_mat')

ksp = PETSc.KSP()
ksp.create(comm = A.getComm())
ksp.setType('bcgs')
ksp.getPC().setType('hypre')

ksp.setOperators(A)
ksp.setFromOptions()
ksp.solve(b,x)

x.viewFromOptions('-view_sol')







