import numpy as np
from dolfin import *
from petsc4py import PETSc
set_log_level(LogLevel.WARNING)
from ..field.testfield import TestField
from ..field.cookiefield import CookieField
from ..field.rough import RoughField
from .parallel import ParallelizableProblem


class Problem(ParallelizableProblem):
    def __init__(self, info):
        assert info['problem']['name'] == "obstacle-rough"
        self.info = info
        self.num_levels = self.info['fe']['levels']

        # setup FE spaces (note that meshes are setup in ParallelizableProblem)
        self.V = [FunctionSpace(mesh, 'CG', self.degree) for mesh in self.mesh]

        self.obstacle_param = info['expansion']['obstacle_parameters']
        
        # setup random field for obstacle
        self.field = RoughField()

        #set up diffusivity
        class ConstantField():
            def __init__(self):
                pass
            def realisation(self,y,V):
                return Constant(1) 
        self.diffusivity = ConstantField()

        # define forcing term
        # *4 due to transormation of the domain from [-1,1]^2 to [0,1]^2
        # and on smaller domain mor force is needed to such that the solution can arrive at obstacle
        self.forcing = -Constant(5*5)

        # define boundary condition
        self.bc = [DirichletBC(V, Constant(0.0), 'on_boundary') for V in self.V]
        self.bc_indices, self.bc_zeros = self.build_bc_indices()

        # get vector objects to hold coefficients for every level
        self.u_vecs = self.build_u_vecs()

        # build prolongation matrices for multigrid
        self.prolongation_matrices = self.build_prolongation_matrices()

        # initialize a matrix list
        self.a_mats = [None for i in range(self.num_levels)]

    def build_image_indices(self):
        image_indices_list = []
        for level in range(self.num_levels):
            img_shape = (self.info['fe']['mesh'] * 2 ** (level + 1) + 1,
                         self.info['fe']['mesh'] * 2 ** (level + 1) + 1)

            mesh = self.mesh[level]
            mesh_coords = mesh.coordinates()[dof_to_vertex_map(self.V[level])]

            x_indices = (mesh_coords[:, 0] * (img_shape[0] - 1) + 1e-5).astype(np.int32)
            y_indices = (mesh_coords[:, 1] * (img_shape[1] - 1) + 1e-5).astype(np.int32)

            image_indices_list.append((x_indices, y_indices))
        return image_indices_list

    def build_prolongation_matrices(self):
        prolongation_matrices = []
        image_indices = self.build_image_indices()
        for level in range(self.num_levels - 1):
            index_hash_table = {}
            n, m = len(image_indices[level][0]), len(image_indices[level + 1][0])
            for i, (x, y) in enumerate(zip(*image_indices[level + 1])):
                index_hash_table[(x, y)] = i

            x_indices, y_indices = image_indices[level]
            overlapping_dofs = [index_hash_table[(x * 2, y * 2)] for x, y in zip(x_indices, y_indices)]

            mesh_connectivity = {i: set() for i in range(n)}
            vertex_to_dof_map_saved = vertex_to_dof_map(self.V[level])
            for cell in self.mesh[level].cells():
                for i in range(3):
                    mesh_connectivity[vertex_to_dof_map_saved[cell[i]]].add(vertex_to_dof_map_saved[cell[(i + 1) % 3]])
                    mesh_connectivity[vertex_to_dof_map_saved[cell[i]]].add(vertex_to_dof_map_saved[cell[(i + 2) % 3]])

            P_backend = PETSc.Mat().createAIJ((n, m)).setUp()
            for i in range(n):
                P_backend.setValue(i, overlapping_dofs[i], 1.0)
                for k in mesh_connectivity[i]:
                    a0, b0 = x_indices[i] * 2, y_indices[i] * 2
                    a2, b2 = x_indices[k] * 2, y_indices[k] * 2
                    a1, b1 = (a2 - a0) // 2 + a0, (b2 - b0) // 2 + b0
                    j = index_hash_table[(a1, b1)]
                    P_backend.setValue(i, j, 0.5)
            P_backend.assemble()
            prolongation_matrices.append(P_backend)

        return prolongation_matrices

    def build_u_vecs(self):
        out = []
        for V in self.V:
            func = Function(V)
            vector = as_backend_type(func.vector()).vec()
            out.append(vector.copy())

        return out

    def build_bc_indices(self):
        bc_indices = []
        bc_zeros = []
        for level in range(self.num_levels):
            bc = self.bc[level]
            inds = np.sort(np.array(list(bc.get_boundary_values().keys())))

            bc_indices.append(inds.astype(np.int32))
            bc_zeros.append(PETSc.Vec().createWithArray(np.zeros(len(inds))))
        return bc_indices, bc_zeros

    def plain_solution(self, y, level, verbose=False):
        # get vector space
        V = self.V[level]

        # build variational problem
        u = TrialFunction(V)
        v = TestFunction(V)

        # compute rhs vector
        f_backend = as_backend_type(assemble(self.forcing * v * dx)).vec()

        # build operator
        obstacle_backend = as_backend_type(self.field.realisation(y,V).vector()).vec()

        kappa = self.diffusivity.realisation(y, V)
        a = kappa * inner(nabla_grad(u), nabla_grad(v)) * dx
        a_backend = as_backend_type(assemble(a)).mat()

        if verbose:
            u_sol, h1_errors, l2_errors = self.solve_multigrid(a_backend, f_backend,
                                                               obstacle_backend, level, verbose=True)
            return u_sol.getArray(), h1_errors, l2_errors
        else:
            u_sol = self.solve_multigrid(a_backend, f_backend, obstacle_backend, level)
            return u_sol.getArray()

    def build_mass_mat(self, level, norm='h1'):
        V = self.V[level]
        u = TrialFunction(V)
        v = TestFunction(V)
        if norm == 'h1':
            a = u * v * dx + inner(nabla_grad(u), nabla_grad(v)) * dx
        elif norm == 'l2':
            a = u * v * dx
        else:
            raise NotImplementedError()

        a_backend = as_backend_type(assemble(a)).mat()
        return a_backend

    def solve_multigrid(self, a_backend, f_backend, obstacle_backend,
                        level, tol=1e-6, smoothing_iters=3, max_iter=1000, verbose=False):
        u0 = self.u_vecs[level].copy()
        u0 += obstacle_backend
        u0.setValues(self.bc_indices[level], self.bc_zeros[level])

        self.a_mats[level] = a_backend
        for i in range(level - 1, -1, -1):
            P = self.prolongation_matrices[i]
            self.a_mats[i] = P.matMult(self.a_mats[i + 1]).matTransposeMult(P)

        for i in range(level):
            self.a_mats[i].zeroRowsColumns(self.bc_indices[i])

        r = u0.copy()
        if verbose:
            h1_mat = self.build_mass_mat(level, norm='h1')
            l2_mat = self.build_mass_mat(level, norm='l2')
            approximations = []
        for i in range(max_iter):
            self.a_mats[level].multAdd(-u0, f_backend, r)
            e = self.recursive_smoother(r, level, smoothing_iters, distance_to_psi=obstacle_backend - u0)

            # e.pointwiseMax(u0 - self.obstacle, e)
            if verbose:
                # print(i, np.max(np.abs(e)))
                approximations.append(u0.copy())

            if np.max(np.abs(e)) <= tol:
                if verbose:
                    print('Converged after {} Iterations'.format(i))
                break
            u0 += e
        else:
            print('Did not converge after {} Iterations'.format(max_iter))
            print("Maximal e:", np.max(np.abs(e)))

        if verbose:
            dists = [x - u0 for x in approximations]
            h1_errors = []
            l2_errors = []
            for x in dists:
                y_h1 = x.copy()
                y_l2 = x.copy()
                h1_mat.mult(x, y_h1)
                l2_mat.mult(x, y_l2)
                h1_errors.append(y_h1.dot(x))
                l2_errors.append(y_l2.dot(x))
            return u0, h1_errors[:-1], l2_errors[:-1]
        else:
            return u0

    def recursive_smoother(self, f, level, smoothing_iters, distance_to_psi=None):
        if distance_to_psi is None:
            distance_to_psi = self.u_vecs[level].copy()
            distance_to_psi.zeroEntries()
        f.setValues(self.bc_indices[level], self.bc_zeros[level])

        u = self.u_vecs[level].copy()
        u.zeroEntries()
        for i in range(smoothing_iters):
            self.a_mats[level].SOR(f, u, sortype=3, omega=0.5)
            u.setValues(self.bc_indices[level], self.bc_zeros[level])
            u.pointwiseMax(distance_to_psi, u)

        r = self.u_vecs[level].copy()
        if level > 0:
            self.a_mats[level].multAdd(-u, f, r)
            r2 = self.u_vecs[level - 1].copy()
            self.prolongation_matrices[level - 1].mult(r, r2)

            e = self.recursive_smoother(r2, level - 1, smoothing_iters)
            self.prolongation_matrices[level - 1].multTransposeAdd(e, u, u)

            u.pointwiseMax(distance_to_psi, u)

        for i in range(smoothing_iters):
            self.a_mats[level].SOR(f, u, sortype=3, omega=0.5)
            u.setValues(self.bc_indices[level], self.bc_zeros[level])
            u.pointwiseMax(distance_to_psi, u)
        return u