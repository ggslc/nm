import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc

OptDB = PETSc.Options()


n = OptDB.getInt('n', 5)
h = 1.0 / (n + 1)


A = PETSc.Mat()

A.create(comm=PETSc.COMM_WORLD)

A.setSizes((n * n, n * n))
A.setType(PETSc.Mat.Type.AIJ)
A.setFromOptions()

A.setPreallocationNNZ(5)


def index_to_grid(r):
    """Convert a row number into a grid point."""
    return (r // n, r % n)


rstart, rend = A.getOwnershipRange()
for row in range(rstart, rend):
    i, j = index_to_grid(row)
    A[row, row] = 4.0 / h**2
    if i > 0:
        column = row - n
        A[row, column] = -1.0 / h**2
    if i < n - 1:
        column = row + n
        A[row, column] = -1.0 / h**2
    if j > 0:
        column = row - 1
        A[row, column] = -1.0 / h**2
    if j < n - 1:
        column = row + 1
        A[row, column] = -1.0 / h**2


A.assemblyBegin()
A.assemblyEnd()

A.viewFromOptions('-view_mat')





ksp = PETSc.KSP()
ksp.create(comm=A.getComm())
ksp.setType(PETSc.KSP.Type.CG)
ksp.getPC().setType(PETSc.PC.Type.GAMG)



ksp.setOperators(A)
ksp.setFromOptions()


x, b = A.createVecs()
b.set(1.0)
ksp.solve(b, x)

x.viewFromOptions('-view_sol')
x.view()




