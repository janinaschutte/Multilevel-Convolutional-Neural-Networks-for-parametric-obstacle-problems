import os, time, json
import numpy as np
from scipy.linalg import eigh
import torch
from dolfin import *
from petsc4py import PETSc
from code.ParamPDE.compute_samples import load_problem_and_sampler, Sampler


def load_problem_info_dict(problemDir):
    """
    :param problem_dir: str
        path of problem json file to load problem parameters from
    :return: dict
        dictionary containing all the loaded problem parameters
    """
    if not os.path.isdir(problemDir):
        raise IOError(f"PROBLEM '{problemDir}' is not a directory")

    problemFile = f"{problemDir}/parameters.json"
    try:
        with open(problemFile, 'r') as f:
            problemInfo = json.load(f)
    except FileNotFoundError:
        raise IOError(f"Cannot read file '{problemFile}'")
    except json.JSONDecodeError:
        raise IOError(f"'{problemFile}' is not a valid JSON file")

    problemInfo['problemDir'] = problemDir
    return problemInfo


class MultiLevelRefinementSampler():

    def __init__(self, problemInfo="./code/ParamPDE/samples-darcy-1", overwrite_levels=None):
        if type(problemInfo) == str:
            try:
                problemInfo = load_problem_info_dict(problemInfo)
            except Exception as e:
                raise Exception('Tried to interpret input as path to problem info dir but failed due to \n' + str(e))

        if problemInfo['problem']['name'] not in ['darcy', 'obstacle', 'stefan', 'obstacle-rough']:
            raise NotImplementedError()
        if overwrite_levels is not None:
            problemInfo['fe']['levels'] = overwrite_levels

        if problemInfo['problem']['name'] == 'obstacle':
            self.is_obstacle_parameter = problemInfo['expansion']['obstacle_parameters']
        else:
            self.is_obstacle_parameter = False
        self.mesh_size = problemInfo['fe']['mesh']

        self.problemInfo = problemInfo
        self.problem, self.sampler = load_problem_and_sampler(self.problemInfo)

        self.vec_to_img_order = []
        self.img_to_vec_order = []
        self.img_shape = []
        self.build_reordering()
        
        self.mass_matrices = {"h1":[], "l2":[]}

        ### Self check that prolongation matrix is correct
        def get_transform_matrix(level0, level1, func):
            row_inds, column_inds, values, shape = func(level0, level1)
            inds_pt = torch.as_tensor(np.stack((row_inds, column_inds), axis=0).astype(np.int64))
            a_pt_sparse = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float32)), size=shape)
            a = a_pt_sparse.to_dense().numpy()
            return a

        a = get_transform_matrix(0, 1, self.get_transformation_matrix_coo_format)
        b = get_transform_matrix(0, 1, self.get_transformation_matrix_coo_format_naive)
        assert np.all(np.isclose(a, b))
        a = get_transform_matrix(1, 0, self.get_transformation_matrix_coo_format)
        b = get_transform_matrix(1, 0, self.get_transformation_matrix_coo_format_naive)
        assert np.all(np.isclose(a, b))

    def reinit_sampler(self):
        self.sampler = Sampler(self.problemInfo)

    def build_reordering(self, eps=1e-5):
        for level in range(self.problemInfo['fe']['levels']):
            self.img_shape.append((self.mesh_size * 2**level + 1,
                                   self.mesh_size * 2**level + 1))

            mesh_coords = self.problem.mesh[level].coordinates()[dof_to_vertex_map(self.problem.V[level])]
            x_indices = (mesh_coords[:, 0] * (self.img_shape[-1][0] - 1) + eps).astype(np.longlong)
            y_indices = (mesh_coords[:, 1] * (self.img_shape[-1][1] - 1) + eps).astype(np.longlong)
            indices = np.ravel_multi_index((y_indices, x_indices), dims=self.img_shape[-1])
            self.img_to_vec_order.append(indices)
            inv_indices = np.zeros_like(indices)
            inv_indices[indices] = np.arange(len(indices), dtype=indices.dtype)
            self.vec_to_img_order.append(inv_indices)

    def get_solutions_images_from_ys(self, ys, level, verbose=False):
        solutions = []

        for i, y in enumerate(ys):
            if verbose:
                print(i)
                print("H",y[0])
            if verbose:
                u_vec, _,_ = self.problem.plain_solution(y, level, verbose=verbose)
            else:
                u_vec = self.problem.plain_solution(y, level, verbose=verbose)
            solutions.append(u_vec[self.vec_to_img_order[level]].reshape(*self.img_shape[level]))

        return np.array(solutions).astype(np.float64)
    
    def get_Gramian_mats_from_ys(self, level, verbose=False):
        A_matrices = []

        for i, y in enumerate(ys):
            a_mat = self.problem.assemble_Gramian_matrix(y, level)
            A_matrices.append(a_mat)
            if verbose:
                print(i)

        return np.array(A_matrices).astype(np.float64)

    def draw_ys(self, num_samples):
        return np.array(self.sampler(num_samples)).astype(np.float64)

    def get_ys_images(self, ys, level, verbose=False):
        out = np.zeros((len(ys), *self.img_shape[level]))
        for i, y in enumerate(ys):
            if self.is_obstacle_parameter:
                y = y[:-1]
            y_vec = self.problem.field.realisation(y, self.problem.V[level]).vector().get_local()
            out[i] = y_vec[self.vec_to_img_order[level]].reshape(self.img_shape[level])
            if verbose:
                print(i)
        return out

    def get_ys_obstacle_images(self, ys, level, verbose=False):
        assert self.is_obstacle_parameter
        out = np.ones((len(ys), *self.img_shape[level])) * ys[:, -1, None, None]
        return out

    def build_matrices(self, up_to):
        self.mass_matrices = {"h1":[], "l2":[]}
        for level in range(up_to):
            self.mass_matrices["h1"].append(self.build_base_mass_matrix(level, norm='h1'))
            self.mass_matrices["l2"].append(self.build_base_mass_matrix(level, norm='l2'))

    def build_base_mass_matrix(self, level, norm='l2'):
        V = self.problem.V[level]

        phi1 = TrialFunction(V)
        phi2 = TestFunction(V)

        if norm == 'l2':
            a = phi1 * phi2 * dx
        elif norm == 'h1':
            a = phi1 * phi2 * dx + inner(grad(phi1), grad(phi2)) * dx
        else:
            raise Exception('No norm named {} implemented!'.format(norm))

        A = assemble(a)
        am = as_backend_type(A).mat()

        return am

    def image_to_function(self, image, level):
        u_vec = image.flatten()[self.img_to_vec_order[level]]
        u_func = Function(self.problem.V[level])
        u_func.vector()[:] = u_vec.copy()
        return u_func

    def function_to_image(self, u_func, level):
        img = u_func.vector().get_local()[self.vec_to_img_order[level]].reshape(*self.img_shape[level])
        return img

    def get_higher_interpolation(self, u_func, level):
        return interpolate(u_func, self.problem.V[level])

    def get_mass_matrix_coo_format(self, level, norm='l2'):
        am = self.build_base_mass_matrix(level, norm=norm)

        row_inds, column_inds, values = am.getValuesCSR()
        shape = am.getSize()
        slice_lengths = np.diff(row_inds)
        uncomp_row_inds = np.repeat(np.arange(len(slice_lengths), dtype=column_inds.dtype), slice_lengths)

        return uncomp_row_inds, column_inds, values, shape

    def get_transformation_matrix_coo_format_naive(self, level0, level1):
        V0 = self.problem.V[level0]
        V1 = self.problem.V[level1]
        u0 = Function(V0)
        u1 = Function(V1)
        n, m = len(u1.vector().get_local()), len(u0.vector().get_local())
        row_inds = []
        column_inds = []
        values = []
        for i in range(m):
            u0.vector()[:] = 0
            u0.vector()[i] = 1
            u1.assign(interpolate(u0, V1))
            u1_vec = u1.vector().get_local()
            inds = u1_vec.nonzero()[0].astype(np.int64)
            vals = u1_vec[inds]
            column_inds.append(np.zeros(len(inds), dtype=np.int64) + i)
            row_inds.append(inds)
            values.append(vals)
        row_inds = np.concatenate(row_inds, axis=0)
        column_inds = np.concatenate(column_inds, axis=0)
        values = np.concatenate(values, axis=0).astype(np.float32)
        return row_inds, column_inds, values, (n, m)

    def get_transformation_matrix_coo_format(self, level0, level1):
        assert abs(level0 - level1) == 1

        turned = False
        if level0 > level1:
            level0, level1 = level1, level0
            turned = True

        V0, V1 = self.problem.V[level0], self.problem.V[level1]

        n, m = V1.dim(), V0.dim()

        overlapping_dofs = self.img_to_vec_order[level0]
        overlapping_dofs = 2 * (overlapping_dofs // self.img_shape[level0][0]) * self.img_shape[level1][0] \
                           + 2 * (overlapping_dofs % self.img_shape[level0][1])
        overlapping_dofs = self.vec_to_img_order[level1][overlapping_dofs]

        if turned:
            return np.arange(m).astype(np.int64), overlapping_dofs.astype(np.int64), np.ones(m, dtype=np.float32), (m, n)

        mesh_connectivity = {i: set() for i in range(n)}
        vertex_to_dof_map_saved = vertex_to_dof_map(V0)
        for cell in self.problem.mesh[level0].cells():
            for i in range(3):
                mesh_connectivity[vertex_to_dof_map_saved[cell[i]]].add(vertex_to_dof_map_saved[cell[(i + 1) % 3]])
                mesh_connectivity[vertex_to_dof_map_saved[cell[i]]].add(vertex_to_dof_map_saved[cell[(i + 2) % 3]])

        row_inds = []
        column_inds = []
        values = []
        for i in range(m):
            column_inds.append(i)
            row_inds.append(overlapping_dofs[i])
            values.append(1)
            a, b = np.unravel_index(self.img_to_vec_order[level1][overlapping_dofs[i]], self.img_shape[level1])

            for k in mesh_connectivity[i]:
                c, d = np.unravel_index(self.img_to_vec_order[level1][overlapping_dofs[k]], self.img_shape[level1])
                x, y = a + np.sign(c - a) * (np.abs(c - a) // 2), b + np.sign(d - b) * (np.abs(d - b) // 2)
                j = self.vec_to_img_order[level1][np.ravel_multi_index((x, y), self.img_shape[level1])]
                column_inds.append(i)
                row_inds.append(j)
                values.append(0.5)

        row_inds = np.asarray(row_inds, dtype=np.int64)
        column_inds = np.asarray(column_inds, dtype=np.int64)
        values = np.asarray(values, dtype=np.float32)

        return row_inds, column_inds, values, (n, m)

    def compute_MRh1error(self, u1_imgs, u0_imgs, level, return_all = False, operator=False):
        # print("build mass matrix")
        am = self.build_base_mass_matrix(level, norm='h1')

        # print("reshape vectors")
        u1_vectors = u1_imgs.reshape(u1_imgs.shape[0], -1)[:, self.img_to_vec_order[level]]
        u0_vectors = u0_imgs.reshape(u0_imgs.shape[0], -1)[:, self.img_to_vec_order[level]]

        # print("setup")
        u0_m = PETSc.Mat()
        u0_m.createDense(u0_vectors.shape, array=u0_vectors)
        u0_m.setUp()

        own_norms = np.sum(PETScMatrix(u0_m.matMult(am)).array() * u0_vectors, axis=1)
        u0_norms_sq = np.mean(own_norms)

        # print("set up")
        u1_m = PETSc.Mat()
        u1_m.createDense(u1_vectors.shape, array=u1_vectors)
        u1_m.setUp()

        # print("diff")
        d_m = u1_m - u0_m

        # print("diff norm")
        losses = np.sum(PETScMatrix(d_m.matMult(am)).array() * (u1_vectors - u0_vectors), axis=1)
        d_norms_sq = np.mean(losses)

        if return_all:
            return np.sqrt(losses/own_norms)
        if operator:
            return np.sqrt(d_norms_sq / u0_norms_sq)
        return np.mean(np.sqrt(losses/own_norms))
    
    def compute_MRl2error(self, u1_imgs, u0_imgs, level, return_all = False, operator=False):
        am = self.build_base_mass_matrix(level, norm='l2')

        u1_vectors = u1_imgs.reshape(u1_imgs.shape[0], -1)[:, self.img_to_vec_order[level]]
        u0_vectors = u0_imgs.reshape(u0_imgs.shape[0], -1)[:, self.img_to_vec_order[level]]

        u0_m = PETSc.Mat()
        u0_m.createDense(u0_vectors.shape, array=u0_vectors)
        u0_m.setUp()

        own_norms = np.sum(PETScMatrix(u0_m.matMult(am)).array() * u0_vectors, axis=1)
        u0_norms_sq = np.mean(own_norms)

        u1_m = PETSc.Mat()
        u1_m.createDense(u1_vectors.shape, array=u1_vectors)
        u1_m.setUp()

        d_m = u1_m - u0_m

        losses = np.sum(PETScMatrix(d_m.matMult(am)).array() * (u1_vectors - u0_vectors), axis=1)
        d_norms_sq = np.mean(losses)

        if return_all:
            return np.sqrt(losses/own_norms)
        if operator:
            return np.sqrt(d_norms_sq / u0_norms_sq)
        return np.mean(np.sqrt(losses/own_norms))

    def load_ys_train(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'train_ys.npy'))[:num_samples]

    def load_ys_images_train(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'train_ys_imgs.npy'))[:num_samples]

    def load_ys_obstacle_images_train(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'train_ys_imgs_obstacle.npy'))[:num_samples]

    def load_solutions_images_train(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'train_u_imgs.npy'))[:num_samples]

    def load_ys_val(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'val_ys.npy'))[:num_samples]

    def load_ys_images_val(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'val_ys_imgs.npy'))[:num_samples]

    def load_ys_obstacle_images_val(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'val_ys_imgs_obstacle.npy'))[:num_samples]

    def load_solutions_images_val(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'val_u_imgs.npy'))[:num_samples]

    def load_ys_test(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'test_ys.npy'))[:num_samples]

    def load_ys_images_test(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'test_ys_imgs.npy'))[:num_samples]

    def load_ys_obstacle_images_test(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'test_ys_imgs_obstacle.npy'))[:num_samples]

    def load_solutions_images_test(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'test_u_imgs.npy'))[:num_samples]

    def load_solutions_images_test_fine(self, num_samples, saving_path):
        return np.load(os.path.join(saving_path, 'test_u_imgs_fine.npy'))[:num_samples]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    level = 5
    sampler = MultiLevelRefinementSampler(problemInfo="./code/ParamPDE/samples-obstacle", overwrite_levels=level)

    ys = sampler.draw_ys(1)
    print(ys[-1])
    y_imgs = sampler.get_ys_obstacle_images(ys, level-1)
    print(y_imgs.min(), y_imgs.max())
    plt.imshow(y_imgs[0])
    plt.savefig('./code/test1.png')


    '''
    for i in range(level):
        row_inds, column_inds, values, shape = sampler.get_mass_matrix_coo_format(i, norm='l2')
        inds_pt = torch.as_tensor(np.stack((row_inds, column_inds), axis=0).astype(np.int64))
        a_pt_sparse = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float32)), size=shape)
        torch.save(a_pt_sparse, './code/test/Data/l2_mat_{}.pt'.format(i))


    for i in range(level):
        row_inds, column_inds, values, shape = sampler.get_mass_matrix_coo_format(i, norm='h1')
        inds_pt = torch.as_tensor(np.stack((row_inds, column_inds), axis=0).astype(np.int64))
        a_pt_sparse = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float32)), size=shape)
        torch.save(a_pt_sparse, './code/test/Data/h1_mat_{}.pt'.format(i))


    ys = sampler.draw_ys(NUM_SAMPLES)
    y_imgs = sampler.get_ys_images(ys, level - 1, verbose=False)
    u_imgs = sampler.get_solutions_images_from_ys(ys, level - 1, verbose=True)

    y_imgs_pt = torch.as_tensor(y_imgs)
    u_imgs_pt = torch.as_tensor(u_imgs)
    torch.save(y_imgs_pt, './code/test/Data/ys.pt')
    torch.save(u_imgs_pt, './code/test/Data/us.pt')

    for i in range(level):
        order0 = torch.as_tensor(sampler.vec_to_img_order[i].astype(np.long))
        order1 = torch.as_tensor(sampler.img_to_vec_order[i].astype(np.long))
        torch.save(order0, './code/test/Data/vec_to_img_{}.pt'.format(i))
        torch.save(order1, './code/test/Data/img_to_vec_{}.pt'.format(i))

    for i in range(level - 1):
        row_inds, column_inds, values, shape = sampler.get_transformation_matrix_coo_format(i, i+1)
        inds_pt = torch.as_tensor(np.stack((row_inds, column_inds), axis=0).astype(np.int64))
        a_pt_sparse = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float32)), size=shape)
        torch.save(a_pt_sparse, './code/test/Data/upsampling_mat_{}_{}.pt'.format(i, i+1))

    for i in range(level - 1):
        row_inds, column_inds, values, shape = sampler.get_transformation_matrix_coo_format(i + 1, i)
        inds_pt = torch.as_tensor(np.stack((row_inds, column_inds), axis=0).astype(np.int64))
        a_pt_sparse = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float32)), size=shape)
        torch.save(a_pt_sparse, './code/test/Data/downsampling_mat_{}_{}.pt'.format(i + 1, i))

    '''
    '''

    def get_transform_matrix(level0, level1):
        row_inds, column_inds, values, shape = sampler.get_transformation_matrix_coo_format_naive(level0, level1)
        inds_pt = torch.as_tensor(np.stack((row_inds, column_inds), axis=0).astype(np.int64))
        a_pt_sparse = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float32)), size=shape)
        a = a_pt_sparse.to_dense().numpy()
        return a


    def get_transform_matrix2(level0, level1):
        row_inds, column_inds, values, shape = sampler.get_transformation_matrix_coo_format(level0, level1)
        inds_pt = torch.as_tensor(np.stack((row_inds, column_inds), axis=0).astype(np.int64))
        a_pt_sparse = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float32)), size=shape)
        a = a_pt_sparse.to_dense().numpy()
        return a


    level0 = 5
    level1 = 6

    V0 = sampler.problem.V[level0]
    V1 = sampler.problem.V[level1]

    #print(sampler.problem.mesh[1].coordinates()[dof_to_vertex_map(sampler.problem.V[1])][38])

    #print(func1.vector()[:][sampler.vec_to_img_order[0]].reshape(sampler.img_shape[0]))
    #print(func2.vector()[:][sampler.vec_to_img_order[1]].reshape(sampler.img_shape[1]))

    # vec1 = func1.vector().get_local()
    # vec2 = func2.vector().get_local()

    #row_inds0, column_inds0, values0, shape0 = sampler.get_transformation_matrix_coo_format_naive(level0, level1)
    #row_inds1, column_inds1, values1, shape1 = sampler.get_transformation_matrix_coo_format(level0, level1)
    #order0 = np.argsort(column_inds0)
    #order1 = np.argsort(column_inds1)
    #row_inds0, column_inds0, values0 = row_inds0[order0], column_inds0[order0], values0[order0]
    #row_inds1, column_inds1, values1 = row_inds1[order1], column_inds1[order1], values1[order1]

    #print(row_inds0)
    #print(row_inds1)

    #t1 = time.time()
    #a = get_transform_matrix(level0, level1).T
    #print('NOW:')
    #t2 = time.time()
    #b = get_transform_matrix2(level0, level1).T
    #t3 = time.time()
    #print(t2 - t1, t3 - t2)

    #print(np.max(np.abs(a - b)))

    #plt.figure(figsize=(12, 6))
    #plt.imshow(a)
    #plt.savefig('./code/test1.png')
    #plt.figure(figsize=(12, 6))
    #plt.imshow(b)
    #plt.savefig('./code/test2.png')
    '''
