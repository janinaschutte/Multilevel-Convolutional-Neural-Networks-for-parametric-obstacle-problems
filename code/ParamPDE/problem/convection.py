from dolfin import *
set_log_level(WARNING)
from ..field.testfield import TestField
from .parallel import ParallelizableProblem


class Problem(ParallelizableProblem):
    def __init__(self, info):
        assert info['problem']['name'] == "nonlinear"
        self.info = info
        self.space = FunctionSpace(self.mesh, 'CG', self.degree)

        self.kappa = Constant(info['problem']['kappa'])

        # define variational problem
        self.forcing = Constant(1)

        # define boundary condition
        self.bc = DirichletBC(self.space, Constant(0.0), 'on_boundary')

        try: h = CellDiameter(V.mesh())
        except NameError: h = CellSize(V.mesh())
        self.h = h

        # setup random field
        M = self.info['expansion']['size']
        mean = self.info['expansion']['mean']
        scale = self.info['expansion']['scale']
        decay = -self.info['expansion']['decay rate']
        expfield = self.info['sampling']['distribution'] == 'normal'
        self.field = TestField(M, mean=mean, scale=scale, decay=decay, expfield=expfield)


    def solution(self, y):
        V = self.space
        kappa = self.kappa
        f = self.forcing
        bc = self.bc
        h = self.h

        def vel(fy):
            vel = as_vector([0.99-fy,(0.99-abs(fy))])
            return vel

        # define discretisation
        fy = self.field.realisation(y, V)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = (inner(vel(fy), grad(u)) * v + kappa * inner(grad(u), grad(v)) +
                        (h / (2.0 * sqrt(inner(vel(fy), vel(fy))))) * inner(vel(fy), grad(v)) * (inner(vel(fy), grad(u)) - kappa * div(grad(u)))) * dx
        L = ((f * v) + (h / (2.0 * sqrt(inner(vel(fy), vel(fy))))) * inner(vel(fy), grad(v)) * f) * dx

        u = Function(V)
        solve(a == L, u, bc)

        return u.vector().get_local(), y
