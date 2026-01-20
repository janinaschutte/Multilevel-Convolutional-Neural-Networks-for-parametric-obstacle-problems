import numpy as np
import torch
from dolfin import *
import time
from tqdm import tqdm
import os
import pickle

from .DataHandler import MultiLevelRefinementSampler, load_problem_info_dict


def build_problem(problem, param_dimension, num_corrections, obstacle_parameter, fixed_radii):
    if problem == "uniform":
        problem_info = "./code/ParamPDE/samples-darcy-1"
        problem_info_dict = load_problem_info_dict(problem_info)
        problem_info_dict['sampling']['distribution'] = "uniform"
        problem_info_dict['expansion']['size'] = param_dimension
        problem_info_dict['expansion']['mean'] = 1
    elif problem == "log-normal":
        problem_info = "./code/ParamPDE/samples-darcy-1"
        problem_info_dict = load_problem_info_dict(problem_info)
        problem_info_dict['sampling']['distribution'] = "normal"
        problem_info_dict['expansion']['size'] = param_dimension
        problem_info_dict['expansion']['mean'] = 0
    elif problem == "cookie":
        problem_info = "./code/ParamPDE/samples-darcy-cookie"
        problem_info_dict = load_problem_info_dict(problem_info)
        problem_info_dict['expansion']['size'] = param_dimension
        problem_info_dict["expansion"]["fixed_radii"] = fixed_radii
    elif problem == "obstacle":
        problem_info = "./code/ParamPDE/samples-obstacle"
        problem_info_dict = load_problem_info_dict(problem_info)
        problem_info_dict['sampling']['distribution'] = "uniform"
        problem_info_dict['expansion']['size'] = param_dimension
        problem_info_dict['expansion']['mean'] = 1
        problem_info_dict['expansion']['obstacle_parameters'] = obstacle_parameter
    elif problem == "stefan":
        problem_info = "./code/ParamPDE/samples-stefan"
        problem_info_dict = load_problem_info_dict(problem_info)
        problem_info_dict['sampling']['distribution'] = "uniform"
        problem_info_dict['expansion']['size'] = param_dimension
        problem_info_dict['expansion']['mean'] = 0.0
    else:
        raise NotImplementedError()

    if problem_info_dict['problem']['name'] not in ['darcy', 'obstacle','stefan']:
        raise NotImplementedError()

    problem_info_dict['fe']['levels'] = num_corrections
    
    return problem_info_dict


def build_data_set(problem, param_dimension, num_corrections, num_samples, num_val_samples, num_test_samples,
                   obstacle_parameter, fixed_radii, saving_path):

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    problem_info_dict = build_problem(problem, param_dimension, num_corrections, obstacle_parameter, fixed_radii)
    print("almost starting")
    DataSampler = MultiLevelRefinementSampler(problemInfo=problem_info_dict, overwrite_levels=num_corrections + 2)

    print("starting training samples")
    ys = DataSampler.draw_ys(num_samples)
    y_imgs = DataSampler.get_ys_images(ys, num_corrections - 1, verbose=False)
    if obstacle_parameter:
        y_imgs_obstacle = DataSampler.get_ys_obstacle_images(ys, num_corrections - 1, verbose=False)
    G_mats = DataSampler.get_Gramian_mats_from_ys(ys, num_corrections - 1, verbose=True)

    np.save(os.path.join(saving_path, 'train_ys.npy'), ys)
    np.save(os.path.join(saving_path, 'train_ys_imgs.npy'), y_imgs)
    if obstacle_parameter:
        np.save(os.path.join(saving_path, 'train_ys_imgs_obstacle.npy'), y_imgs_obstacle)
    np.save(os.path.join(saving_path, 'train_G_mats.npy'), G_mats)

    print("starting validation samples")
    ys_val = DataSampler.draw_ys(num_val_samples)
    y_imgs_val = DataSampler.get_ys_images(ys_val, num_corrections - 1, verbose=False)
    if obstacle_parameter:
        y_imgs_obstacle_val = DataSampler.get_ys_obstacle_images(ys_val, num_corrections - 1, verbose=False)
    # u_imgs_val = DataSampler.get_solutions_images_from_ys(ys_val, num_corrections - 1, verbose=True)
    G_mats_val = DataSampler.get_Gramian_mats_from_ys(ys, num_corrections - 1, verbose=True)

    np.save(os.path.join(saving_path, 'val_ys.npy'), ys_val)
    np.save(os.path.join(saving_path, 'val_ys_imgs.npy'), y_imgs_val)
    if obstacle_parameter:
        np.save(os.path.join(saving_path, 'val_ys_imgs_obstacle.npy'), y_imgs_obstacle_val)
    np.save(os.path.join(saving_path, 'val_G_mats.npy'), G_mats_val)

    print("starting test samples")
    ys_test = DataSampler.draw_ys(num_test_samples)
    y_imgs_test = DataSampler.get_ys_images(ys_test, num_corrections - 1, verbose=False)
    if obstacle_parameter:
        y_imgs_obstacle_test = DataSampler.get_ys_obstacle_images(ys_test, num_corrections - 1, verbose=False)
    u_imgs_test = DataSampler.get_solutions_images_from_ys(ys_test, num_corrections - 1, verbose=True)
    fine_u_imgs_test = DataSampler.get_solutions_images_from_ys(ys_test, num_corrections + 1, verbose=True)
    G_mats_test = DataSampler.get_Gramian_mats_from_ys(ys, num_corrections - 1, verbose=True)

    np.save(os.path.join(saving_path, 'test_ys.npy'), ys_test)
    np.save(os.path.join(saving_path, 'test_ys_imgs.npy'), y_imgs_test)
    if obstacle_parameter:
        np.save(os.path.join(saving_path, 'test_ys_imgs_obstacle.npy'), y_imgs_obstacle_test)
    np.save(os.path.join(saving_path, 'test_u_imgs.npy'), u_imgs_test)
    np.save(os.path.join(saving_path, 'test_u_imgs_fine.npy'), fine_u_imgs_test)
    np.save(os.path.join(saving_path, 'test_G_mats.npy'), G_mats_test)

    del DataSampler.sampler
    pickle.dump(DataSampler, open(os.path.join(saving_path, 'data_sampler.p'), 'wb'))


if __name__ == "__main__":
    #build_data_set('uniform', 10, 4, 4000, 256, 256, False, True, './code/FinalConvModel/Data/testuniform')
    #build_data_set('cookie', 16, 4, 4000, 256, 256, False, True, './code/FinalConvModel/Data/testcookie')
    '''
    try:
        build_data_set('cookie', 16, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/cookie16fixed')
    except Exception as e:
        print('Generating cookie data failed due to', e)

    try:
        build_data_set('obstacle', 10, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/obstacle10')
    except Exception as e:
        print('Generating obstacle data failed due to', e)
    
    try:
        build_data_set('obstacle', 11, 7, 10000, 1024, 1024, True, True, './code/FinalConvModel/Data/obstacle11variable')#10000, 1024.1024
    except Exception as e:
        print('Generating obstacle variable data failed due to', e)
    
    #try:
    build_data_set('stefan', 4, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/stefan4variable')#10000, 1024.1024
    #except Exception as e:
    #    print('Generating stefan variable data failed due to', e)
    '''
    try:
        build_data_set('obstacle', 11, 3, 100, 10, 10, True, True, './code/FinalConvModel/Data/obstacle11variable_test_data')
    except Exception as e:
        print('Generating obstacle variable data failed due to', e)
    
    '''
    build_data_set('uniform', 10, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/uniform10')
    build_data_set('uniform', 50, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/uniform50')
    build_data_set('uniform', 100, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/uniform100')
    build_data_set('uniform', 200, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/uniform200')
    
    
    build_data_set('log-normal', 10, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/lognormal10')
    build_data_set('log-normal', 50, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/lognormal50')
    build_data_set('log-normal', 100, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/lognormal100')
    '''
    #build_data_set('log-normal', 200, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/lognormal200')

    #build_data_set('cookie', 16, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/cookie16fixed')
    #build_data_set('cookie', 32, 7, 10000, 1024, 1024, False, False, './code/FinalConvModel/Data/cookie32variable')
    #build_data_set('cookie', 64, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/cookie64fixed')
    #build_data_set('cookie', 128, 7, 10000, 1024, 1024, False, False, './code/FinalConvModel/Data/cookie128variable')


    '''
    try:
        build_data_set('cookie', 32, 7, 10000, 1024, 1024, False, False, './code/FinalConvModel/Data/cookie32variable')
    except Exception as e:
        print('Generating cookie variable data failed due to', e)

    print('Done!')
    '''

    #build_data_set('cookie', 64, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/cookie64fixed')
    #build_data_set('cookie', 128, 7, 10000, 1024, 1024, False, False, './code/FinalConvModel/Data/cookie128variable')

    #build_data_set('obstacle', 50, 7, 10000, 1024, 1024, False, True, './code/FinalConvModel/Data/obstacle50')

    # leonhard-09 process 54203
    #build_data_set('obstacle', 51, 7, 10000, 1024, 1024, True, True, './code/FinalConvModel/Data/obstacle51variable')
    # leonhard-07 process 46118
    #build_data_set('obstacle', 11, 7, 10000, 1024, 1024, True, True, './code/FinalConvModel/Data/obstacle11variable')
    print('Done!')

