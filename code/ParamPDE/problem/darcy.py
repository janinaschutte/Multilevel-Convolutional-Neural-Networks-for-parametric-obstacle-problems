import numpy as np
from dolfin import *
set_log_level(LogLevel.WARNING)
from ..field.testfield import TestField
from ..field.cookiefield import CookieField
from .parallel import ParallelizableProblem



class Problem(ParallelizableProblem):
    def __init__(self, info):
        assert info['problem']['name'] == "darcy"
        self.info = info

        # setup FE spaces (note that meshes are setup in ParallelizableProblem)
        self.V = [FunctionSpace(mesh, 'CG', self.degree) for mesh in self.mesh]

        # setup random field
        if info['expansion']['type'] == "karhunen_loeve":
            M = self.info['expansion']['size']
            mean = self.info['expansion']['mean']
            scale = self.info['expansion']['scale']
            decay = -self.info['expansion']['decay rate']
            expfield = self.info['sampling']['distribution'] == 'normal'
            self.field = TestField(M, mean=mean, scale=scale, decay=decay, expfield=expfield)
        elif info['expansion']['type'] == "cookie":
            M = self.info['expansion']['size']
            mean = self.info['expansion']['mean']
            fixed_radii = self.info['expansion']['fixed_radii']

            if fixed_radii:
                num_cookies_per_dim = np.sqrt(M)
                min_rad = self.info['expansion']['fixed_radius']
                max_rad = self.info['expansion']['fixed_radius']
            else:
                num_cookies_per_dim = np.sqrt(M * 0.5)
                min_rad = self.info['expansion']['min_radius']
                max_rad = self.info['expansion']['max_radius']
            assert np.abs(num_cookies_per_dim % 1) < 1e-13, 'Number of cookie dimension not admissible'
            num_cookies_per_dim = int(np.round(num_cookies_per_dim))

            self.field = CookieField(num_cookies_per_dim,
                                     mean=mean, radius_range=(min_rad, max_rad), fixed_radii=fixed_radii)

        # define forcing term
        self.forcing = Constant(1)

        # define boundary condition
        self.bc = [DirichletBC(V, Constant(0.0), 'on_boundary') for V in self.V]

    def solution(self, y):
        """
        Return solution of Darcy problem for given parameter realization y.

        Parameter
        ---------
        y   :   array_like
                Sample for realization of the problem.

        Returns
        -------
        u   :   solution vector (numpy array)
        """
        f = self.forcing

        # iterate levels and solve problems
        U = []
        for l, V in enumerate(self.V):
            # get field discretization and BC
            kappa = self.field.realisation(y, V)
            bc = self.bc[l]

            # setup variational problem
            u = TrialFunction(V)
            v = TestFunction(V)
            a = kappa * inner(nabla_grad(u), nabla_grad(v)) * dx
            L = f * v * dx

            # solve for residual correction
            if l > 0:
                #u1 = interpolate(U[-1], V)
                #L = L - kappa * inner(nabla_grad(u1), nabla_grad(v)) * dx
                u1 = sum(interpolate(u0, V) for u0 in U)
                L = f * v * dx - kappa * inner(nabla_grad(u1), nabla_grad(v)) * dx

            u = Function(V)
            solve(a == L, u, bc)
            U.append(u)

        return [u.vector().get_local() for u in U]

    def plain_solution(self, y, level):
        """
        :param y: array-like
            sample for the realization of diffusion parameter
        :param level: int
            mesh fineness level to compute refinement on
        :return: array-like
            array encoding solution of problem for given parameter
        """
        f = self.forcing

        # get vector space
        V = self.V[level]

        # compute parameter function
        kappa = self.field.realisation(y, V)

        # get boundary condition for mesh of this level
        bc = self.bc[level]

        # build variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        a = kappa * inner(nabla_grad(u), nabla_grad(v)) * dx
        L = f * v * dx

        # solve system
        u_sol = Function(V)
        solve(a == L, u_sol, bc)
        return u_sol.vector().get_local()

    def assemble_Gramian_matrix(self, y, level):
        """
        :param y: array-like
            sample for the realization of diffusion parameter
        :param level: int
            mesh fineness level to compute refinement on
        :return: array-like
            array encoding matrix from variational Darcy problem
        """
        # get vector space
        V = self.V[level]
        # compute parameter function
        kappa = self.field.realisation(y, V)
        # build variational problem
        u = TestFunction(V)
        v = TestFunction(V)
        a = kappa * inner(nabla_grad(u), nabla_grad(v)) * dx
        return assemble(a).get_local()


    def refinement(self, y, u_estimate, level):
        """
        :param y: array-like
            sample for the realization of diffusion parameter
        :param u_estimate: dolfin.Function
            estimate for the solution to compute refinement of
        :param level: int
            mesh fineness level to compute refinement on
        :return: dolfin.Function
            refinement function defined on resolution given by level
        """
        f = self.forcing

        # get vector space
        V = self.V[level]

        # compute parameter function
        kappa = self.field.realisation(y, V)

        # get boundary condition for mesh of this level
        bc = self.bc[level]

        # interpolate u_estimate on V
        u0 = interpolate(u_estimate, V)

        # build variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        a = kappa * inner(nabla_grad(u), nabla_grad(v)) * dx
        L = f * v * dx - kappa * inner(nabla_grad(u0), nabla_grad(v)) * dx

        # solve system
        u_sol = Function(V)
        solve(a == L, u_sol, bc)

        return u_sol

    def application(self, y_u, level):
        y,u_vec = y_u
        M = self.info['expansion']['size']
        assert y.shape == (M,)

        V = self.V[level]
        f = self.forcing
        kappa = self.field.realisation(y, V)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = kappa * inner(grad(u), grad(v)) * dx
        L = f * v * dx

        bc = DirichletBC(V, Constant(0.0), 'on_boundary')

        A, b = assemble_system(a, L, bc)
        u = Function(V).vector()
        u[:] = u_vec
        return (A*u).get_local()

    def residual(self, y_u, level):
        y, u_vec = y_u
        M = self.info['expansion']['size']
        assert y.shape == (M,)

        V = self.V[level]
        f = self.forcing
        kappa = self.field.realisation(y, V)

        u = TrialFunction(V)
        v = TestFunction(V)
        a = kappa * inner(grad(u), grad(v)) * dx
        L = f * v * dx

        bc = DirichletBC(V, Constant(0.0), 'on_boundary')

        A, b = assemble_system(a, L, bc)
        u = Function(V).vector()
        u[:] = u_vec
        res = (A*u - b).get_local()
        assert res.shape == u_vec.shape
        return res

    def residual_estimator(self, y_u, level):
        y,u_vec = y_u
        V = self.V[level]
        u = Function(V)
        u.vector()[:] = u_vec
        f = self.forcing
        kappa = self.field.realisation(y, V)

        # setup indicator
        mesh = V.mesh()
        h = CellDiameter(mesh)
        DG0 = FunctionSpace(mesh, 'DG', 0)
        dg0 = TestFunction(DG0)

        kappa = self.field.realisation(y, V)
        R_T = -(f + div(kappa * grad(u)))
        R_dT = kappa * grad(u)
        J = jump(R_dT)
        indicator = h ** 2 * (1 / kappa) * R_T ** 2 * dg0 * dx + avg(h) * avg(1 / kappa) * J **2 * 2 * avg(dg0) * dS

        # prepare indicators
        eta_res_local = assemble(indicator, form_compiler_parameters={'quadrature_degree': -1})
        return eta_res_local.get_local()

    # def refine_mesh(self, marked_cells):
    #     marker = MeshFunction("bool", self.mesh, self.mesh.topology().dim())
    #     marker.set_all(False)
    #     marker[marked_cells] = True  # for idx in marked_cells: marker[idx] = True
    #     self.mesh = refine(self.mesh, marker)
    #     self.space = FunctionSpace(self.mesh, 'CG', self.degree)
