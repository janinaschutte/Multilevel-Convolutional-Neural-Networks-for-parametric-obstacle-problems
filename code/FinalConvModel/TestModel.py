# on leonhard-13 id 94591

import os, time
import numpy as np
import torch
from dolfin import *

from .Model import BigConvModel
from .DataHandler import MultiLevelRefinementSampler

NUM_SAMPLES = 1024 # 2000s
BATCH_SIZE = 32
NUM_REFINEMENTS = 7
FINE_REFINEMENT_IND = 9

device = torch.device("cpu")#torch.device('cuda:2')

DataSampler = MultiLevelRefinementSampler(problemInfo="./code/ParamPDE/samples-darcy-1",
                                          overwrite_levels=FINE_REFINEMENT_IND)

from .TrainModel import build_pt_dataset,build_pt_obstacle_dataset, get_transforms_and_orderings

num_corrections = NUM_REFINEMENTS
model_size_per_level = [8e+5 for _ in range(num_corrections)]
model_size_per_level = [10e+5 for _ in range(2)] + [8e+5 for i in range(num_corrections-2)]
upsampling_matrices, upsampling_img_to_vec, upsampling_vec_to_img, img_sizes = get_transforms_and_orderings(DataSampler, num_corrections, device)
orderings2 = upsampling_vec_to_img
orderings = []
for level in range(num_corrections):
    order = torch.as_tensor(DataSampler.img_to_vec_order[level].astype(np.longlong)).to(device)
    orderings.append(order)

def apply_model_to_data_set(model, inputs, inputs_obstacle, means, stds):
    out_list = [[] for i in range(NUM_REFINEMENTS)]
    for i in range(0, NUM_SAMPLES, BATCH_SIZE):
        input_batch = [x[i: i + BATCH_SIZE].to(device) for x in inputs]
        input_batch_obstacle = [x[i: i + BATCH_SIZE].to(device) for x in inputs_obstacle]
        out = model.forward(input_batch, obstacle_levels=input_batch_obstacle)#model(input_batch)
        out = [(out[j]*stds[j] + means[j]).detach() for j in range(len(out))]
        for j in range(NUM_REFINEMENTS):
            out_list[j].append(out[j])
    return [torch.cat(x, dim=0) for x in out_list]

def add_out(out_list):
    added_list = [out_list[0].clone().squeeze()]
    for i in range(len(out_list)-1):
        added_list.append(upsample(added_list[i], i).squeeze() + out_list[i+1].clone().squeeze())
    return added_list

# def interpolate_to_higher_level_(u_imgs, level0, level1):
#     out_imgs = []
#     for u_img in u_imgs:
#         print("img to func")
#         u_function = DataSampler.image_to_function(u_img, level0)
#         print("func up")
#         new_u_function = DataSampler.get_higher_interpolation(u_function, level1)
#         print("func to img")
#         out_imgs.append(DataSampler.function_to_image(new_u_function, level1))
#     return np.array(out_imgs)

def interpolate_to_higher_level(u_imgs, level0, level1):
    imgs_interp = []
    for u_img in u_imgs:
        u_function = DataSampler.image_to_function(u_img.squeeze(), level0)
        new_u_function = DataSampler.get_higher_interpolation(u_function, level1)
        imgs_interp.append(np.array(DataSampler.function_to_image(new_u_function, level1)))
    return np.stack(imgs_interp, axis=0)



def upsample(img_batch, level):
    order0 = orderings[level]
    order1 = orderings2[level]
    a_pt_sparse = upsampling_matrices[level]
    img_shape = img_sizes[level]
    img_batch = img_batch.view(img_batch.shape[0], -1)[:, order0].permute(1, 0)
    img_batch = a_pt_sparse.mm(img_batch).permute(1, 0)
    img_batch = img_batch[:, order1].view(img_batch.shape[0], 1, *img_shape)
    return img_batch

error_dict = {'l2_errors': [], 'h1_errors': [],
              'l2_errors_partial': [], 'h1_errors_partial': [],
              'l2_error_to_fine_reference': [], 'h1_error_to_fine_reference': [], 
              'l2_error_true_to_fine_reference': [], 'h1_error_true_to_fine_reference': []}
error_dict['l2_errors_opertor'] = []
error_dict['h1_errors_opertor'] = []
error_dict['l2_error_to_fine_reference_opertor'] = []
error_dict['h1_error_to_fine_reference_opertor'] = []

# error_dict = np.load('./code/FinalConvModel/ModelSaves/TestMetricsDict.npy', allow_pickle=True)#.tolist()

def test_solutions(outs, outs_added, grid_sols, grid_diffs, fine_sols, level):
    print("get l2 same grid partial")
    error_dict['l2_errors_partial'].append(DataSampler.compute_MRl2error(outs, grid_diffs, level, return_all =True))
    print("get h1")
    error_dict['h1_errors_partial'].append(DataSampler.compute_MRh1error(outs, grid_diffs, level, return_all = True))

    print("get l2 same grid")
    error_dict['l2_errors'].append(DataSampler.compute_MRl2error(outs_added, grid_sols, level, return_all =True))
    print("get h1")
    error_dict['h1_errors'].append(DataSampler.compute_MRh1error(outs_added, grid_sols, level, return_all = True))
    
    print("interpolate out")
    fine_total_outs = interpolate_to_higher_level(outs_added, level, FINE_REFINEMENT_IND - 1)
    print("get l2 fine grid")
    error_dict['l2_error_to_fine_reference'].append(DataSampler.compute_MRl2error(fine_total_outs, fine_sols, FINE_REFINEMENT_IND - 1, return_all =True))
    print("get h1")
    error_dict['h1_error_to_fine_reference'].append(DataSampler.compute_MRh1error(fine_total_outs, fine_sols, FINE_REFINEMENT_IND - 1, return_all =True))

    print("interpolate true")
    fine_total_grid_sols = interpolate_to_higher_level(grid_sols, level, FINE_REFINEMENT_IND - 1)
    print("get l2 true to fine")
    error_dict['l2_error_true_to_fine_reference'].append(DataSampler.compute_MRl2error(fine_total_grid_sols, fine_sols, FINE_REFINEMENT_IND - 1, return_all =True))
    print("get h1")
    error_dict['h1_error_true_to_fine_reference'].append(DataSampler.compute_MRh1error(fine_total_grid_sols, fine_sols, FINE_REFINEMENT_IND - 1, return_all =True))
    
    print("get l2 operator same grid")
    error_dict['l2_errors_opertor'].append(DataSampler.compute_MRl2error(outs_added, grid_sols, level, operator =True))
    print("get h1")
    error_dict['h1_errors_opertor'].append(DataSampler.compute_MRh1error(outs_added, grid_sols, level, operator = True))
    
    print("get l2 operator fine grid")
    error_dict['l2_error_to_fine_reference_opertor'].append(DataSampler.compute_MRl2error(fine_total_outs, fine_sols, FINE_REFINEMENT_IND - 1, operator =True))
    print("get h1")
    error_dict['h1_error_to_fine_reference_opertor'].append(DataSampler.compute_MRh1error(fine_total_outs, fine_sols, FINE_REFINEMENT_IND - 1, operator =True))


print("loading training dict")
saving_path = './code/FinalConvModel/ModelSaves'
run_name = 'obstacle11variable_big_1'#'obstacle11level7'
training_log_dict_saving_path = os.path.join(saving_path, 'TrainingMetricsDict_{}.npy'.format(run_name))

training_log_dict = np.load(training_log_dict_saving_path,  allow_pickle=True).tolist()
print(training_log_dict.keys())
print('test_mrh1_per_level',training_log_dict['test_mrh1_per_level'])
print('test_partial_mrh1_per_level',training_log_dict['test_partial_mrh1_per_level'])
print('test_mrh1_reference',training_log_dict['test_mrh1_reference'])
print('test_mrl2_per_level', training_log_dict['test_mrl2_per_level'])
print('test_partial_mrl2_per_level', training_log_dict['test_partial_mrl2_per_level'])
print('test_mrl2_reference',training_log_dict['test_mrl2_reference'])

y_img_mean = training_log_dict['y_img_mean']
y_img_std = training_log_dict['y_img_std']
target_stds = training_log_dict['target_stds'] 
target_means = training_log_dict['target_means']


import matplotlib.pyplot as plt
plot_saving_path = f'code/FinalConvModel/Tests/{run_name}/'
plt.plot(training_log_dict['test_mrh1_per_level'])
plt.savefig(f'{plot_saving_path}test_mrh1_per_level.png')
plt.clf()
plt.plot(training_log_dict['test_partial_mrh1_per_level'])
plt.title(r"Mean relative $H^1$ error on each level")
plt.xlabel("individual FE spaces")
numbs = len(training_log_dict['test_partial_mrh1_per_level'])
plt.xticks(range(numbs), [fr"$V_{i}$" for i in range(1, numbs+1)])
plt.ylabel(r"mean relative $H^1$ error")
plt.yscale('log')
plt.savefig(f'{plot_saving_path}test_partial_mrh1_per_level.png')
plt.clf()
plt.plot(training_log_dict['test_mrl2_per_level'])
plt.savefig(f'{plot_saving_path}test_mrl2_per_level.png')
plt.clf()
plt.plot(training_log_dict['test_partial_mrl2_per_level'])
plt.title(r"Mean relative $L^2$ error on each level")
plt.xlabel("individual FE spaces")
numbs = len(training_log_dict['test_partial_mrl2_per_level'])
plt.xticks(range(numbs), [fr"$V_{i}$" for i in range(1, numbs+1)])
plt.ylabel(r"mean relative $L^2$ error")
plt.yscale('log')
plt.savefig(f'{plot_saving_path}test_partial_mrl2_per_level.png')
plt.clf()




print("loading test data")
num_samples = NUM_SAMPLES#1024
data_loading_path = './code/FinalConvModel/Data/obstacle11variable'

obstacle_parameter=True
y_imgs = DataSampler.load_ys_images_test(num_samples, data_loading_path)
if obstacle_parameter:
    y_imgs_obstacle = DataSampler.load_ys_obstacle_images_test(num_samples, data_loading_path)
u_imgs = DataSampler.load_solutions_images_test(num_samples, data_loading_path)
u_imgs_fine = DataSampler.load_solutions_images_test_fine(num_samples, data_loading_path)
print("u fine", u_imgs_fine.shape)



print("preparing test set")
test_data_ys = build_pt_dataset(DataSampler, y_imgs, num_corrections, device)
test_data_ys = [(x - y_img_mean) / y_img_std for x in test_data_ys]
if obstacle_parameter:
    test_data_ys_obstacle = build_pt_obstacle_dataset(DataSampler, y_imgs_obstacle, num_corrections, device)
else:
    test_data_ys_obstacle = None
target_list = build_pt_dataset(DataSampler, u_imgs, num_corrections, device)
test_data_us = [target_list[0].clone()]
for i in range(1, NUM_REFINEMENTS):
    test_data_us.append(target_list[i].clone() - upsample(target_list[i - 1].clone(), i - 1))

model = BigConvModel(NUM_REFINEMENTS, model_size_per_level, True, upsampling_matrices, upsampling_img_to_vec, upsampling_vec_to_img, img_sizes)#*get_transforms_and_orderings(DataSampler, num_corrections, device))
model.load_state_dict(torch.load(os.path.join(saving_path, 'model_{}.pt'.format(run_name)), map_location=torch.device('cpu')))#'./code/MultilevelBigConvModel/ModelSaves/model.pt'))
model.to(device)
model.eval()

outs = apply_model_to_data_set(model, test_data_ys, test_data_ys_obstacle, target_means, target_stds)
outs_added = add_out(outs)

# np.save(f'./code/FinalConvModel/Tests/{run_name}/TestMetricsDict.npy', error_dict, allow_pickle=True)
error_dict = np.load(f'./code/FinalConvModel/Tests/{run_name}/TestMetricsDict.npy', allow_pickle=True).tolist()
x_axis = []
print(f"Start testing on samples: {[out.shape for out in outs_added]}")
for i in range(NUM_REFINEMENTS):
    print('Level {}:'.format(i))
    x_axis.append(outs[i].shape[-1]*outs[i].shape[-2])
    # test_solutions(outs[i].cpu().numpy(), outs_added[i].cpu().numpy(), target_list[i].cpu().numpy(), test_data_us[i].cpu().numpy(), u_imgs_fine, i)#data_us[i].cpu().numpy(), data_us[-1].cpu().numpy(), i)

    for key in error_dict.keys():
        print(key, np.mean(error_dict[key][i]), np.var(error_dict[key][i]), np.max(np.abs(error_dict[key][i])), error_dict[key][i].shape)

_mean = [np.mean(e) for e in error_dict['l2_error_to_fine_reference']]
plt.plot(x_axis, _mean, label= "Model")
_std = [np.std(e) for e in error_dict['l2_error_to_fine_reference']]
plt.fill_between(x_axis, [m-s for m,s in zip(_mean, _std)], [m+s for m,s in zip(_mean, _std)], alpha = 0.3)
_mean = [np.mean(e) for e in error_dict['l2_error_true_to_fine_reference']]
plt.plot(x_axis, _mean, label="FE")
_std = [np.std(e) for e in error_dict['l2_error_true_to_fine_reference']]
plt.fill_between(x_axis, [m-s for m,s in zip(_mean, _std)], [m+s for m,s in zip(_mean, _std)], alpha = 0.3)
plt.yscale('log')
plt.xscale('log')
plt.title(r'$L^2$ reference error')
plt.xlabel("number of FE coefficients")
plt.legend()
plt.savefig(f'{plot_saving_path}test_reference_l2_per_level.png')
plt.clf()

_mean = [np.mean(e) for e in error_dict['h1_error_to_fine_reference']]
plt.plot(x_axis, _mean, label= "Model")
_std = [np.std(e) for e in error_dict['h1_error_to_fine_reference']]
plt.fill_between(x_axis, [m-s for m,s in zip(_mean, _std)], [m+s for m,s in zip(_mean, _std)], alpha = 0.3)
_mean = [np.mean(e) for e in error_dict['h1_error_true_to_fine_reference']]
plt.plot(x_axis, _mean, label="FE")
_std = [np.std(e) for e in error_dict['h1_error_true_to_fine_reference']]
plt.fill_between(x_axis, [m-s for m,s in zip(_mean, _std)], [m+s for m,s in zip(_mean, _std)], alpha = 0.3)
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.title(r'$H^1$ reference error')
plt.xlabel("number of FE coefficients")
plt.savefig(f'{plot_saving_path}test_reference_h1_per_level.png')
plt.clf()


np.save(f'./code/FinalConvModel/Tests/{run_name}/TestMetricsDict.npy', error_dict, allow_pickle=True)

