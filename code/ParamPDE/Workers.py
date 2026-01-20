import numpy as np
from dolfin import *
import multiprocessing as mp


def solution_worker_2d_images(params):
    """
    :param params: list of tuple(3,)
        list containing the y-parameters, the problem object and the pixel coordinates
        ys: np.ndarray() sampled parameter vector
        problem: ParallelizableProblem class defining mesh, solution operator, etc.
        pixel_coordinates: list of np.ndarray(n, m, 2) coordinates of pixel samplings
            for each refinement steps one element in this list
    :return: list of np.ndarray()
        list of len(pixel_coords) + 1 elements containing the interpolations of the
        problem refinement solution for each refinement on the corresponding pixel coordinates
        and one extra summation of all the refinements to a full solution on the finest grid
        interpolated on the last pixel_coords
    """
    # unpack input (could be done with * in the function but yeah whatev..)
    ys, problem, pixel_coords = params

    # compute the solutions (or better their refinements) on each scale into a list
    uls = problem.solution(ys)

    # some initialization and new variables
    result_list = []
    high_fidelity_image_shape = pixel_coords[-1].shape[:2]
    full_u_interpolation = np.zeros(high_fidelity_image_shape)
    high_fidelity_flattened_coords = pixel_coords[-1].reshape(-1, 2)

    # for each refinement solution, interpolate it on the corresponding coordinates
    # and add it to the big high fidelity solution
    for l in range(len(uls)):
        # get shape and flattaned coordinates
        image_shape = pixel_coords[l].shape[:2]
        flattened_coords = pixel_coords[l].reshape(-1, 2)

        # build fenics function
        V = problem.V[l]
        u_function = Function(V)
        # hacky way to get vector into function
        u_function.vector()[:] = uls[l]
        ys_function = problem.field.realisation(ys, V)

        # compute interpolation by just evaluating the fenics function
        # if there is a faster way to do this on a whole array at once, please tell me
        u_interpolation = np.array([u_function(x, y) for x, y in flattened_coords])
        u_interpolation = u_interpolation.reshape(image_shape)

        # add interpolation onto high fidelity solution
        full_u_interpolation += np.array([u_function(x, y) for x, y in high_fidelity_flattened_coords]).reshape(high_fidelity_image_shape)

        # evaluate parameter function on pixel coordinates
        ys_interpolation = np.array([ys_function(x, y) for x, y in flattened_coords])
        ys_interpolation = ys_interpolation.reshape(image_shape)

        result_list.append(np.stack((u_interpolation, ys_interpolation), axis=2))

    # also add high fidelity solution and last parameter interpolation
    result_list.append(np.stack((full_u_interpolation,  ys_interpolation), axis=2))

    return result_list


def solution_worker_fenics_vectors(params):
    """
    :param params: list of tuple(2,)
        list containing the y-parameters and the problem
        ys: np.ndarray() sampled parameter vector
        problem: ParallelizableProblem class defining mesh, solution operator, etc.
    :return: touple(2,) of array-like
        function vectors of the high fidelity solution together with the
        high fidelity parameter function vector (not the same as the parameter vector of the input)
    """
    # unpack parameters
    ys, problem = params

    # set level as max level
    level = len(problem.V) - 1

    # compute high fidelity solution directly
    u_vector = problem.plain_solution(ys, level)

    return u_vector, ys