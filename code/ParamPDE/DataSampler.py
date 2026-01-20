import numpy as np
import os, json, time
from dolfin import *
import multiprocessing as mp

from .compute_samples import load_problem_and_sampler
from .Workers import solution_worker_2d_images, solution_worker_fenics_vectors


class DataSampler():
    def __init__(self, problemInfo, overwrite_level=None):
        """
        :param problemInfo: dict
            info dict to get problem parameters. Is passed to load_problem_and_sampler
            usually loaded from json file
        """
        if type(problemInfo) == str:
            try:
                problemInfo = self.load_problem_info_dict(problemInfo)
            except Exception as e:
                raise Exception('Tried to interpret input as path to problem info dir but failed due to \n' + str(e))

        if problemInfo['problem']['name'] != 'darcy':
            raise NotImplementedError()

        if overwrite_level is not None:
            problemInfo['fe']['levels'] = overwrite_level

        # get sampler and problem object
        self.problem, self.sampler = load_problem_and_sampler(problemInfo)

    def get_pixel_coords(self, image_size):
        """
        :param image_size: tuple(2,) of int
            image resolution
        :return: np.array(image_size[0], image_size[1], 2) of float
            coordinate array for positions of pixels in [0-1] square
        """
        # assumes we are on the 0-1 square
        coords = np.stack(np.meshgrid(np.linspace(0, 1, image_size[0], endpoint=True),
                                      np.linspace(0, 1, image_size[1], endpoint=True)), axis=2)
        return coords

    def compute_2d_image_batch(self, batch_size, image_sizes):
        """
        :param batch_size: int
            how many samples to compute
        :param image_sizes: list of tuple(2,) of int
            list of resolutions to compute solutions and parameter functions on
            should be the same length as the number of refinements in problem
        :return: list of list of np.ndarray(image_sizes[i][0], image_sizes[i][1], 2)
            for each element in the batch there is a list of output image arrays.
            For each image size, there is an element in this list
            containing the solution and the parameter function of the corresponding refinement
            interpolated on the coordinates given by the image size
            Additionally, the last element is the full solution (all refinements added together)
            in the finest mesh using fenics and interpolated on the pixel coordinates given by the
            last element in image sizes (basically the best approx of the solution and the paramter function)

        if the meshes in self.problem are unit square and the image sizes coincide with the number of nodes
        in the meshes along each axis (for each refinement respectively), then the mapping between fenics
        function and 2d array is lossless.
        """
        assert len(image_sizes) == len(self.problem.mesh)

        # compute coordinates of each pixel from resolution to pass to workers
        pixel_coords = [self.get_pixel_coords(image_size) for image_size in image_sizes]

        # get worker parameters for each element in batch
        params = [(y, self.problem, pixel_coords) for y in self.sampler(batch_size)]

        # distribute to a pool of workers
        pool = mp.Pool()
        results = pool.map_async(solution_worker_2d_images, params)
        pool.close()
        pool.join()

        return results

    def compute_fenics_vectors_batch(self, batch_size):
        """
        :param batch_size: int
            number of samples to compute as one batch.
            For this, batch_size many subprocesses will be opened
        :return: list of tuple(2,)
            a list of all the results from the batch.
            Each element itself is a touple containing the high fidelity solution together with the
            high fidelity parameter function vector
        """
        # get worker parameters into list to pass to pool
        params = [(x, self.problem) for x in self.sampler(batch_size)]

        # do the whole pool thing
        pool = mp.Pool()
        results = pool.map_async(solution_worker_fenics_vectors, params).get()
        pool.close()
        pool.join()

        return results

    def load_problem_info_dict(self, problem_dir):
        """
        :param problem_dir: str
            path of problem json file to load problem parameters from
        :return: dict
            dictionary containing all the loaded problem parameters
        """
        if not os.path.isdir(problem_dir):
            raise IOError(f"PROBLEM '{problem_dir}' is not a directory")

        problemFile = f"{problem_dir}/parameters.json"
        try:
            with open(problemFile, 'r') as f:
                problemInfo = json.load(f)
        except FileNotFoundError:
            raise IOError(f"Cannot read file '{problemFile}'")
        except json.JSONDecodeError:
            raise IOError(f"'{problemFile}' is not a valid JSON file")

        problemInfo['problemDir'] = problem_dir
        return problemInfo

    def compute_data_set1(self, data_path, num_samples, file_name, batch_size=32, verbose=False):
        """
        :param data_path: str
            path to save .npy files and mesh.xml file to
        :param num_samples: int
            number of samples to compute
        :param file_name: str
            name prefix of the saved .npy files
        :param batch_size: int
            batch size for the computation. Dictates how many subprocesses are opened
        :param verbose: bool
            if True, progress is printed

        Computed and saves the high fidelity solutions in the data path.
        For each sample, the solution u and the parameters y are saved as fem vectors
        in a tuple(2,) for the highest mesh resolution of self.problem
        """
        # computing and saving mesh to put in the vectors if needed
        mesh = self.problem.mesh[-1]
        File(os.path.join(data_path, file_name + 'mesh.xml')) << mesh

        N = num_samples
        i = 0
        while N > 0:
            batch = self.compute_fenics_vectors_batch(min(N, batch_size))
            for result in batch:
                path = os.path.join(data_path, file_name + '_{:06d}'.format(i))
                np.save(path, np.array(result, dtype=np.object))
                i += 1
                if verbose and i % 100 == 0:
                    print('Done with computing [{}/{}] datapoints'.format(i, num_samples))
            N -= batch_size

    def compute_data_set2(self, data_path, num_samples, file_name, batch_size=32, verbose=False):
        """
        :param data_path: str
            path to save .npy files and mesh.xml file to
        :param num_samples: int
            number of samples to compute
        :param file_name: str
            name prefix of the saved .npy files
        :param batch_size: int
            batch size for the computation. Dictates how many subprocesses are opened
        :param verbose: bool
            if True, progress is printed

        Computed and saves the high fidelity and its interpolated reduced counterpart
        solutions in the data path.
        For each sample, the solution u and the parameters y are saved as fem vectors
        in a tuple(2,) for the all mesh resolutions of self.problem
        """
        # computing and saving mesh to put in the vectors if needed
        mesh = self.problem.mesh[-1]
        max_level = len(self.problem.V) - 1
        File(os.path.join(data_path, file_name + '_{}_'.format(max_level) + 'mesh.xml')) << mesh

        N = num_samples
        i = 0
        us = [Function(V) for V in self.problem.V]
        while N > 0:
            batch = self.compute_fenics_vectors_batch(min(N, batch_size))

            for result in batch:
                u_vector_fine, y = result

                us[-1].vector()[:] = u_vector_fine
                path = os.path.join(data_path, file_name + '_{}'.format(max_level) + '_{:06d}.npy'.format(i))
                np.save(path, u_vector_fine.astype(np.float64))

                for level in list(range(max_level))[::-1]:
                    us[level].assign(interpolate(us[level + 1], self.problem.V[level]))
                    path = os.path.join(data_path, file_name + '_{}'.format(level) + '_{:06d}.npy'.format(i))
                    np.save(path, us[level].vector().get_local().astype(np.float64))

                path = os.path.join(data_path, file_name + '_{}'.format('y') + '_{:06d}.npy'.format(i))
                np.save(path, y.astype(np.float64))

                i += 1
                if verbose and i % 100 == 0:
                    print('Done with computing [{}/{}] datapoints'.format(i, num_samples))
            N -= batch_size




if __name__ == "__main__": 
    problemDir = "./code/ParamPDE/samples-darcy-1"

    sampler = DataSampler(problemDir)
    sampler.compute_data_set1('./data/TrainData', 10000, 'full_vectors_darcy')