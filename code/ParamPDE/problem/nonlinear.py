import numpy as np
from dolfin import *
set_log_level(WARNING)
from ..field.testfield import TestField
from .parallel import ParallelizableProblem


class Problem(ParallelizableProblem):
    def __init__(self, info):
        assert info['problem']['name'] == "nonlinear"
        self.info = info
        self.space = FunctionSpace(self.mesh, 'CG', self.degree)

        self.fac = Constant(info['problem']['fac'])      # factor in nonlinearity (TODO: might have to be smaller than 0.1)
        self.kappa = Constant(info['problem']['kappa'])

        # define boundary condition
        self.bc = DirichletBC(self.space, Constant(0.0), 'on_boundary')

        # define variational problem
        self.forcing = Constant(1)

        # setup random field
        M = self.info['expansion']['size']
        mean = self.info['expansion']['mean']
        scale = self.info['expansion']['scale']
        decay = -self.info['expansion']['decay rate']
        expfield = self.info['sampling']['distribution'] == 'normal'
        self.field = TestField(M, mean=mean, scale=scale, decay=decay, expfield=expfield)

        # compute initial guess
        self.initial_guess = 0
        self.initial_guess = self.solution(np.zeros(M))

    def solution(self, y):
        V = self.space
        fac = self.fac
        kappa = self.kappa
        f = self.forcing
        bc = self.bc

        def q(a, u):
            # nonlinear coefficient
            return (kappa + fac*a + u)**2
            # return (1 + u)**2

        # construct coefficient
        a = self.field.realisation(y, V)

        u = Function(V)
        u.vector()[:] = self.initial_guess

        v = TestFunction(V)

        # define nonlinear functional and solve
        F = (q(a, u) * inner(grad(u), grad(v)) - f*v) * dx
        solve(F == 0, u, bc)
        return u.vector().get_local(), y
