import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
# from dolfin import *
import time
# from tqdm import tqdm
import os
import pickle

from .Model import BigConvModel
# from .DataHandler import MultiLevelRefinementSampler, load_problem_info_dict


def get_interpolations(sampler, u_imgs, level0, level1, device):
    row_inds, column_inds, values, shape = sampler.get_transformation_matrix_coo_format(level0, level1)
    inds_pt = torch.as_tensor(np.stack((row_inds, column_inds), axis=0).astype(np.int64))
    a_pt_sparse = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float64)), size=shape)

    order0 = torch.as_tensor(sampler.img_to_vec_order[level0].astype(np.longlong))
    order1 = torch.as_tensor(sampler.vec_to_img_order[level1].astype(np.longlong))

    u_vecs = u_imgs.view(u_imgs.shape[0], -1)[:, order0].permute(1, 0)
    u_vecs = a_pt_sparse.mm(u_vecs).permute(1, 0)

    u_imgs = u_vecs[:, order1].view(u_vecs.shape[0], *sampler.img_shape[level1])
    return u_imgs


def get_obstacle_interpolations(sampler, u_imgs, level0, level1, device):
    assert level1 + 1 == level0
    column_inds, row_inds, _, shape = sampler.get_transformation_matrix_coo_format(level1, level0)
    shape = shape[::-1]
    row_column_list = [[] for i in range(shape[0])]
    for row_ind, column_ind in zip(row_inds, column_inds):
        row_column_list[row_ind].append(column_ind)

    order0 = torch.as_tensor(sampler.img_to_vec_order[level0].astype(np.longlong))
    order1 = torch.as_tensor(sampler.vec_to_img_order[level1].astype(np.longlong))

    u_vecs = u_imgs.view(u_imgs.shape[0], -1)[:, order0].permute(1, 0)
    new_u_vecs = torch.zeros((shape[0], u_vecs.shape[1]), dtype=u_vecs.dtype)
    for i, row_column_inds in enumerate(row_column_list):
        new_u_vecs[i, :] = torch.max(u_vecs[row_column_inds, :], dim=0)[0]
    u_vecs = new_u_vecs.permute(1, 0)

    u_imgs = u_vecs[:, order1].view(u_vecs.shape[0], *sampler.img_shape[level1])
    return u_imgs


def build_pt_dataset(sampler, imgs_numpy, num_corrections, device):
    out_list = [torch.as_tensor(imgs_numpy)]
    for i in range(1, num_corrections):
        out_list.append(get_interpolations(sampler, out_list[-1], num_corrections - i, num_corrections - i - 1, device))
    out_list.reverse()

    for i in range(num_corrections):
        out_list[i] = out_list[i].to(torch.float32)[:, None, :, :]

    return out_list


def build_pt_obstacle_dataset(sampler, imgs_numpy, num_corrections, device):
    out_list = [torch.as_tensor(imgs_numpy)]
    for i in range(1, num_corrections):
        out_list.append(get_obstacle_interpolations(sampler, out_list[-1], num_corrections - i,
                                                    num_corrections - i - 1, device))
    out_list.reverse()

    for i in range(num_corrections):
        out_list[i] = out_list[i].to(torch.float32)[:, None, :, :]

    return out_list


def get_transforms_and_orderings(DataSampler, num_corrections, device):
    out_matrices = []
    out_orders_img_to_vec = []
    out_orders_vec_to_img = []
    img_sizes = []
    for i in range(num_corrections - 1):
        row_inds, column_inds, values, shape = DataSampler.get_transformation_matrix_coo_format(i, i + 1)
        inds_pt = torch.as_tensor(np.stack((row_inds, column_inds), axis=0).astype(np.int64))
        a_pt_sparse = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float32)), size=shape)
        order0 = torch.as_tensor(DataSampler.img_to_vec_order[i].astype(np.longlong)).to(device)
        order1 = torch.as_tensor(DataSampler.vec_to_img_order[i+1].astype(np.longlong)).to(device)

        out_matrices.append(a_pt_sparse.to(device))
        out_orders_img_to_vec.append(order0)
        out_orders_vec_to_img.append(order1)
        img_sizes.append(DataSampler.img_shape[i + 1])

    return out_matrices, out_orders_img_to_vec, out_orders_vec_to_img, img_sizes


class LossFunct():
    def __init__(self, sampler, num_corrections, device, train_data_us, norm='h1', small_mode=False):
        self.num_corrections = num_corrections
        if not small_mode:
            num_corrections = num_corrections + 2
        self.a_matrices = []
        self.a_matrices64 = []
        self.orderings = []
        for level in range(num_corrections):
            row_inds, column_inds, values, shape = sampler.get_mass_matrix_coo_format(level, norm=norm)

            inds_pt = torch.as_tensor(np.stack((row_inds, column_inds), axis=0).astype(np.int64))
            a_pt_sparse = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float32)), size=shape)
            a_pt_sparse64 = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float64)), size=shape)
            self.a_matrices.append(a_pt_sparse.to(device))
            self.a_matrices64.append(a_pt_sparse64.cpu())

            order = torch.as_tensor(sampler.img_to_vec_order[level].astype(np.longlong)).to(device)
            self.orderings.append(order)

        self.upsampling_matrices, _, self.orderings2, self.img_sizes = get_transforms_and_orderings(
            sampler, num_corrections, device)

        if train_data_us is not None:
            targets = [x[:100].clone() for x in train_data_us]
            self.target_means = [torch.mean(targets[0], dim=0)[None, ...].to(device)]
            self.target_means.extend([torch.zeros_like(x[:1], device=device) for x in targets[1:]])
            self.target_stds = [torch.sqrt(torch.mean((targets[0] - self.target_means[0].cpu())**2)).item()]
            for i in range(1, len(targets)):
                var = torch.mean((targets[i].to(device) - self.upsample(targets[i - 1].to(device), i - 1))**2)
                self.target_stds.append(torch.sqrt(var).item())
            print(self.target_stds)

    def get_single_loss(self, preds, targets, level):
        x = (preds - targets)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)[:, self.orderings[level]].permute(1, 0)
        return torch.sum(torch.mm(self.a_matrices[level], x) * x) / x.shape[0]

    def upsample(self, img_batch, level):
        order0 = self.orderings[level]
        order1 = self.orderings2[level]
        a_pt_sparse = self.upsampling_matrices[level]
        img_shape = self.img_sizes[level]

        img_batch = img_batch.view(img_batch.shape[0], -1)[:, order0].permute(1, 0)
        img_batch = a_pt_sparse.mm(img_batch).permute(1, 0)
        img_batch = img_batch[:, order1].view(img_batch.shape[0], 1, *img_shape)
        return img_batch

    def get_overall_loss(self, pred_list, target_list):
        #summed_solution = None
        level = len(pred_list)
        new_target_list = [target_list[0]]
        for i in range(1, level):
            new_target_list.append(target_list[i] - self.upsample(target_list[i - 1], i - 1))

        target_list = [(x - m) / std for x, std, m in zip(new_target_list, self.target_stds[:level], self.target_means[:level])]

        losses = [self.get_single_loss(x, y, i) for i, (x, y) in enumerate(zip(pred_list, target_list))]
        loss = sum(losses)
        return loss, [x.data.item() for x in losses]

    def get_norm_sq(self, x, level):
        x = x.clone().cpu().to(torch.float64)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)[:, self.orderings[level]].permute(1, 0)
        return torch.sum(torch.mm(self.a_matrices64[level], x) * x, dim=0)


def train_epoch(model, optimizer, loss_function, inputs, obstacle_inputs, targets, num_corrections,
                num_samples_per_level, device, batch_size):
    cumulative_loss = 0
    cumulative_intermediate_losses = [0 for i in range(num_corrections)]
    cumulative_intermediate_losses_counter = [0 for i in range(num_corrections)]
    print("in training")
    level_batches = []
    for level in range(num_corrections):
        start_point = 0 if level == num_corrections - 1 else num_samples_per_level[level + 1]
        inds = np.random.permutation(np.arange(start_point, num_samples_per_level[level]))
        for i in range(0, len(inds), batch_size):
            level_batches.append((level, inds[i:i + batch_size]))
    num_batches = len(level_batches)
    level_batches = [level_batches[k] for k in np.random.permutation(np.arange(num_batches))]

    if not np.all(np.diff(num_samples_per_level) == 0):
        extra_level_batches = []
        for i in range(len(level_batches)):
            rand_batch_indices = np.random.choice(num_samples_per_level[-1], size=(batch_size,))
            extra_level_batches.append((num_corrections - 1, rand_batch_indices))
        level_batches = [x for y in zip(level_batches, extra_level_batches) for x in y]

    print(len(level_batches))
    # print(torch.cuda.get_device_properties(device).total_memory*9.53e-6, torch.cuda.memory_reserved(device)*9.53e-6,
    #       torch.cuda.memory_allocated(device)*9.53e-6, (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))*9.53e-6)
    for level, batch_indices in level_batches:
        input_batch = [x[batch_indices].to(device) for x in inputs]
        if obstacle_inputs is not None:
            input_batch_obstacle = [x[batch_indices].to(device) for x in obstacle_inputs]
        else:
            input_batch_obstacle = None
        #print(level)
        target_batch = [x[batch_indices].to(device) for x in targets]
        print(level, "attempting forward", [inp for inp in input_batch], [inp.max() for inp in input_batch])
        outs = model.forward(input_batch, obstacle_levels=input_batch_obstacle, num_levels=level + 1)
        print("getting loss", [inp for inp in outs], [inp.max() for inp in outs])
        loss, intermediate_losses = loss_function.get_overall_loss(outs, target_batch)
        cumulative_loss += loss.data.item() / num_batches
        for k in range(level + 1):
            cumulative_intermediate_losses[k] += intermediate_losses[k]
            cumulative_intermediate_losses_counter[k] += 1

        print("grad")
        optimizer.zero_grad()
        print("zeroed", loss)
        loss.backward()
        print("backwarded")
        optimizer.step()
        print("grad done")

    cumulative_intermediate_losses = [x / y for x, y in zip(cumulative_intermediate_losses,
                                                            cumulative_intermediate_losses_counter)]
    print("returning")
    return cumulative_loss, cumulative_intermediate_losses


def val_epoch(model, loss_function, target_stds, target_means, inputs, obstacle_inputs, targets, num_corrections,
              num_val_samples, batch_size, device):
    val_indices = np.arange(num_val_samples)
    val_loss = 0
    intermediate_val_losses = [0 for i in range(num_corrections)]

    target_norms_sq = [0 for i in range(num_corrections)]
    distance_norms_sq = [0 for i in range(num_corrections)]
    corr_distance_norms_sq = [0 for i in range(num_corrections)]

    for i in range(0, num_val_samples, batch_size):
        batch_indices = val_indices[i:i + batch_size]

        input_batch = [x[batch_indices].to(device) for x in inputs]

        if obstacle_inputs is not None:
            input_batch_obstacle = [x[batch_indices].to(device) for x in obstacle_inputs]
        else:
            input_batch_obstacle = None
        target_batch = [x[batch_indices].to(device) for x in targets]

        corrections_batch = [target_batch[0]]
        for i in range(1, num_corrections):
            corrections_batch.append(target_batch[i] - loss_function.upsample(target_batch[i - 1], i - 1))

        for i, x in enumerate(target_batch):
            target_norms_sq[i] += torch.mean(loss_function.get_norm_sq(x, i)) / np.ceil(num_val_samples / batch_size)

        with torch.no_grad():
            model_out = model(input_batch, obstacle_levels=input_batch_obstacle, num_levels=num_corrections)
            outs = [x.detach() for x in model_out]
            loss, intermediate_losses = loss_function.get_overall_loss(outs, target_batch)
            outs = [x * std + target_mean for x, std, target_mean in zip(outs, target_stds, target_means)]
        summed_outs = [outs[0]]
        for i in range(1, num_corrections):
            summed_outs.append(loss_function.upsample(summed_outs[-1], i - 1) + outs[i])

        val_loss += loss.data.item() / np.ceil(num_val_samples / batch_size)
        for k in range(num_corrections):
            intermediate_val_losses[k] += intermediate_losses[k] / np.ceil(num_val_samples / batch_size)

        for i, (x, y) in enumerate(zip(summed_outs, target_batch)):
            distance_norms_sq[i] += torch.mean(loss_function.get_norm_sq(x - y, i)) / np.ceil(num_val_samples / batch_size)

        for i, (x, y) in enumerate(zip(outs, corrections_batch)):
            corr_distance_norms_sq[i] += torch.mean(loss_function.get_norm_sq(x - y, i)) / np.ceil(num_val_samples / batch_size)

    val_mres = []
    val_partial_mres = []
    for level in range(num_corrections):
        mre = torch.sqrt(distance_norms_sq[level] / target_norms_sq[level]).data.item()
        val_mres.append(mre)
        corr_mre = torch.sqrt(corr_distance_norms_sq[level] / target_norms_sq[level]).data.item()
        val_partial_mres.append(corr_mre)

    return val_loss, intermediate_val_losses, val_mres, val_partial_mres


def test_epoch(model, loss_function, target_stds, target_means, inputs, obstacle_inputs, target_imgs, target_imgs_fine,
               num_corrections, num_test_samples, batch_size, device):

    test_indices = np.arange(num_test_samples)
    target_norms_sq = [0 for i in range(num_corrections)]
    distance_norms_sq = [0 for i in range(num_corrections)]
    corr_distance_norms_sq = [0 for i in range(num_corrections)]
    target_norms_sq_fine = 0
    distance_norms_sq_fine = 0

    for i in range(0, num_test_samples, batch_size):
        batch_indices = test_indices[i:i + batch_size]

        input_batch = [x[batch_indices].to(device) for x in inputs]
        if obstacle_inputs is not None:
            input_batch_obstacle = [x[batch_indices].to(device) for x in obstacle_inputs]
        else:
            input_batch_obstacle = None
        target_batch = [x[batch_indices].to(device) for x in target_imgs]
        target_batch_fine = target_imgs_fine[batch_indices].to(device)

        corrections_batch = [target_batch[0]]
        for i in range(1, num_corrections):
            corrections_batch.append(target_batch[i] - loss_function.upsample(target_batch[i - 1], i - 1))

        for i, x in enumerate(target_batch):
            target_norms_sq[i] += torch.mean(loss_function.get_norm_sq(x, i)) / np.ceil(num_test_samples / batch_size)
        target_norms_sq_fine += torch.mean(loss_function.get_norm_sq(target_batch_fine, num_corrections + 1)) \
                                / np.ceil(num_test_samples / batch_size)

        with torch.no_grad():
            model_out = model(input_batch, obstacle_levels=input_batch_obstacle, num_levels=num_corrections)
            outs = [x.detach() for x in model_out]
            outs = [x * std + target_mean for x, std, target_mean in zip(outs, target_stds, target_means)]
        summed_outs = [outs[0]]
        for i in range(1, num_corrections):
            summed_outs.append(loss_function.upsample(summed_outs[-1], i - 1) + outs[i])

        for i, (x, y) in enumerate(zip(outs, corrections_batch)):
            corr_distance_norms_sq[i] += torch.mean(loss_function.get_norm_sq(x - y, i)) / np.ceil(num_test_samples / batch_size)

        for i, (x, y) in enumerate(zip(summed_outs, target_batch)):
            distance_norms_sq[i] += torch.mean(loss_function.get_norm_sq(x - y, i)) / np.ceil(num_test_samples / batch_size)
        fine_outs = loss_function.upsample(summed_outs[-1], num_corrections - 1)
        fine_outs = loss_function.upsample(fine_outs, num_corrections)
        distance_norms_sq_fine += torch.mean(loss_function.get_norm_sq(fine_outs - target_batch_fine, num_corrections + 1)) \
                                  / np.ceil(num_test_samples / batch_size)


    test_mres = []
    test_partial_mres = []
    for level in range(num_corrections):
        mre = torch.sqrt(distance_norms_sq[level] / target_norms_sq[level]).data.item()
        test_mres.append(mre)
        corr_mre = torch.sqrt(corr_distance_norms_sq[level] / target_norms_sq[level]).data.item()
        test_partial_mres.append(corr_mre)

    test_mre_fine = torch.sqrt(distance_norms_sq_fine / target_norms_sq_fine).data.item()

    return test_mres, test_partial_mres, test_mre_fine


def test_epoch_l2(model, loss_function, target_stds, target_means, inputs, obstacle_inputs, target_imgs, target_imgs_fine,
               num_corrections, num_test_samples, batch_size, device):

    test_indices = np.arange(num_test_samples)
    target_norms_sq = [0 for i in range(num_corrections)]
    distance_norms_sq = [0 for i in range(num_corrections)]
    corr_distance_norms_sq = [0 for i in range(num_corrections)]
    target_norms_sq_fine = 0
    distance_norms_sq_fine = 0

    for i in range(0, num_test_samples, batch_size):
        batch_indices = test_indices[i:i + batch_size]

        input_batch = [x[batch_indices].to(device) for x in inputs]
        if obstacle_inputs is not None:
            input_batch_obstacle = [x[batch_indices].to(device) for x in obstacle_inputs]
        else:
            input_batch_obstacle = None
        target_batch = [x[batch_indices].to(device) for x in target_imgs]
        target_batch_fine = target_imgs_fine[batch_indices].to(device)

        corrections_batch = [target_batch[0]]
        for i in range(1, num_corrections):
            corrections_batch.append(target_batch[i] - loss_function.upsample(target_batch[i - 1], i - 1))

        for i, x in enumerate(target_batch):
            target_norms_sq[i] += torch.mean(loss_function.get_norm_sq(x, i)) / np.ceil(num_test_samples / batch_size)
        target_norms_sq_fine += torch.mean(loss_function.get_norm_sq(target_batch_fine, num_corrections + 1)) \
                                / np.ceil(num_test_samples / batch_size)

        with torch.no_grad():
            model_out = model(input_batch, obstacle_levels=input_batch_obstacle, num_levels=num_corrections)
            outs = [x.detach() for x in model_out]
            outs = [x * std + target_mean for x, std, target_mean in zip(outs, target_stds, target_means)]
        summed_outs = [outs[0]]
        for i in range(1, num_corrections):
            summed_outs.append(loss_function.upsample(summed_outs[-1], i - 1) + outs[i])

        for i, (x, y) in enumerate(zip(outs, corrections_batch)):
            corr_distance_norms_sq[i] += torch.mean(loss_function.get_norm_sq(x - y, i)) / np.ceil(num_test_samples / batch_size)

        for i, (x, y) in enumerate(zip(summed_outs, target_batch)):
            distance_norms_sq[i] += torch.mean(loss_function.get_norm_sq(x - y, i)) / np.ceil(num_test_samples / batch_size)
        fine_outs = loss_function.upsample(summed_outs[-1], num_corrections - 1)
        fine_outs = loss_function.upsample(fine_outs, num_corrections)
        distance_norms_sq_fine += torch.mean(loss_function.get_norm_sq(fine_outs - target_batch_fine, num_corrections + 1)) \
                                  / np.ceil(num_test_samples / batch_size)


    test_mres = []
    test_partial_mres = []
    for level in range(num_corrections):
        mre = torch.sqrt(distance_norms_sq[level] / target_norms_sq[level]).data.item()
        test_mres.append(mre)
        corr_mre = torch.sqrt(corr_distance_norms_sq[level] / target_norms_sq[level]).data.item()
        test_partial_mres.append(corr_mre)

    test_mre_fine = torch.sqrt(distance_norms_sq_fine / target_norms_sq_fine).data.item()

    return test_mres, test_partial_mres, test_mre_fine


def get_lr_schedule_func(num_epochs, a=0.3, b=0.8, lr1=1e-3, lr2=2e-5):
    def learning_rate_schedule(epoch):
        if epoch < int(a * num_epochs):
            return lr1
        if epoch < int(b * num_epochs):
            alpha = (epoch - int(a * num_epochs)) / (int(b * num_epochs) - int(a * num_epochs))
            return (1 - alpha) * lr1 + alpha * lr2
        return lr2

    return learning_rate_schedule


def compute_test_l2_error(metrics_dict_path, device):
    metrics_dict = np.load(metrics_dict_path, allow_pickle=True)[None][0]
    input_params = metrics_dict['input_params']
    data_loading_path = input_params['data_loading_path']
    num_test_samples = input_params['num_test_samples']
    num_corrections = input_params['num_corrections']
    y_img_mean = metrics_dict['y_img_mean']
    y_img_std = metrics_dict['y_img_std']
    obstacle_parameter = input_params['obstacle_parameter']
    saving_path = input_params['saving_path']
    run_name = input_params['run_name']
    target_stds = metrics_dict['target_stds']
    target_means = [x.to(device) for x in metrics_dict['target_means']]
    model_size_per_level = input_params['model_size_per_level']
    batch_size = input_params['batch_size']

    DataSampler = pickle.load(open(os.path.join(data_loading_path, 'data_sampler.p'), 'rb'))
    DataSampler.reinit_sampler()

    y_imgs_test = DataSampler.load_ys_images_test(num_test_samples, data_loading_path)
    if obstacle_parameter:
        y_imgs_obstacle = DataSampler.load_ys_obstacle_images_test(num_test_samples, data_loading_path)
    test_data_ys = build_pt_dataset(DataSampler, y_imgs_test, num_corrections, device)
    test_data_ys = [(x - y_img_mean) / y_img_std for x in test_data_ys]
    if obstacle_parameter:
        test_data_ys_obstacle = build_pt_obstacle_dataset(DataSampler, y_imgs_obstacle, num_corrections, device)
    else:
        test_data_ys_obstacle = None
    u_imgs_test = DataSampler.load_solutions_images_test(num_test_samples, data_loading_path)
    fine_u_imgs_test = DataSampler.load_solutions_images_test_fine(num_test_samples, data_loading_path)
    test_data_us = build_pt_dataset(DataSampler, u_imgs_test, num_corrections, device)
    test_data_us_fine = torch.as_tensor(fine_u_imgs_test).to(torch.float32)[:, None, :, :]

    loss_function = LossFunct(DataSampler, num_corrections, device, test_data_us, norm='l2')

    model = BigConvModel(num_corrections, model_size_per_level, obstacle_parameter,
                         *get_transforms_and_orderings(DataSampler, num_corrections, device))
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(saving_path, 'model_{}.pt'.format(run_name))))
    model.eval()

    test_mres, test_partial_mres, test_mre_fine = test_epoch_l2(model, loss_function, target_stds, target_means,
                                                             test_data_ys, test_data_ys_obstacle, test_data_us,
                                                             test_data_us_fine, num_corrections, num_test_samples,
                                                             batch_size, device)

    metrics_dict['test_mrl2_per_level'] = test_mres
    metrics_dict['test_partial_mrl2_per_level'] = test_partial_mres
    metrics_dict['test_mrl2_reference'] = test_mre_fine

    for level in range(num_corrections):
        print('Level {} Test partial MR-L2 Loss: {}'.format(level, test_partial_mres[level]))
        print('Level {} Test MR-L2 Loss: {}'.format(level, test_mres[level]))
    print('Reference Test MR-L2 Loss: {}'.format(test_mre_fine))

    np.save(metrics_dict_path, metrics_dict, allow_pickle=True)




def perform_run(problem, param_dimension, num_corrections, num_samples_per_level, num_epochs, batch_size,
                num_val_samples, num_test_samples, model_size_per_level, obstacle_parameter, fixed_radii, saving_path,
                run_name, device, data_loading_path):
    print('training on', device)
    input_params = {'problem': problem,
                    'param_dimension': param_dimension,
                    'num_corrections': num_corrections,
                    'num_samples_per_level': num_samples_per_level,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'num_val_samples': num_val_samples,
                    'num_test_samples': num_test_samples,
                    'model_size_per_level': model_size_per_level,
                    'obstacle_parameter': obstacle_parameter,
                    'fixed_radii': fixed_radii,
                    'saving_path': saving_path,
                    'run_name': run_name,
                    'data_loading_path': data_loading_path}

    assert np.all(np.diff(num_samples_per_level) <= 0)

    training_log_dict = {'input_params': input_params,
                         'train_errors': [],
                         'train_errors_per_level': [],
                         'val_errors': [],
                         'val_errors_per_level': [],
                         'val_mrh1_per_level': [],
                         'val_partial_mrh1_per_level': [],
                         'epoch': [],
                         'num_samples_per_lvl': num_samples_per_level}
    training_log_dict_saving_path = os.path.join(saving_path, 'TrainingMetricsDict_{}.npy'.format(run_name))

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    # build data sampler
    print("building data sampler")
    DataSampler = pickle.load(open(os.path.join(data_loading_path, 'data_sampler.p'), 'rb'))
    DataSampler.reinit_sampler()

    # sample ys and fine us for training dataset
    print("loading training set")
    max_num_samples = max(num_samples_per_level)
    # ys = DataSampler.load_ys_train(max_num_samples, data_loading_path)
    y_imgs = DataSampler.load_ys_images_train(max_num_samples, data_loading_path)
    if obstacle_parameter:
        y_imgs_obstacle = DataSampler.load_ys_obstacle_images_train(max_num_samples, data_loading_path)
    u_imgs = DataSampler.load_solutions_images_train(max_num_samples, data_loading_path)

    # normalize ys and generate multilevel decomposition
    print("normalizing training set")
    y_img_mean = np.mean(y_imgs)
    y_img_std = np.sqrt(np.mean((y_imgs - y_img_mean)**2))
    training_log_dict['y_img_mean'] = y_img_mean
    training_log_dict['y_img_std'] = y_img_std

    print("preparing training set")
    train_data_ys = build_pt_dataset(DataSampler, y_imgs, num_corrections, device)
    train_data_ys = [(x - y_img_mean) / y_img_std for x in train_data_ys]
    if obstacle_parameter:
        train_data_ys_obstacle = build_pt_obstacle_dataset(DataSampler, y_imgs_obstacle, num_corrections, device)
    else:
        train_data_ys_obstacle = None
    train_data_us = build_pt_dataset(DataSampler, u_imgs, num_corrections, device)

    # sample ys and fine us for validation dataset
    print("loading validation set")
    # ys_val = DataSampler.load_ys_val(num_val_samples, data_loading_path)
    y_imgs_val = DataSampler.load_ys_images_val(num_val_samples, data_loading_path)
    if obstacle_parameter:
        y_imgs_obstacle = DataSampler.load_ys_obstacle_images_val(num_val_samples, data_loading_path)
    u_imgs_val = DataSampler.load_solutions_images_val(num_val_samples, data_loading_path)

    # normalize validation dataset with params from training set
    val_data_ys = build_pt_dataset(DataSampler, y_imgs_val, num_corrections, device)
    val_data_ys = [(x - y_img_mean) / y_img_std for x in val_data_ys]
    if obstacle_parameter:
        val_data_ys_obstacle = build_pt_obstacle_dataset(DataSampler, y_imgs_obstacle, num_corrections, device)
    else:
        val_data_ys_obstacle = None
    val_data_us = build_pt_dataset(DataSampler, u_imgs_val, num_corrections, device)

    # build loss function and get target normalization parameters
    loss_function = LossFunct(DataSampler, num_corrections, device, train_data_us)
    target_stds = loss_function.target_stds
    target_means = loss_function.target_means
    training_log_dict['target_stds'] = target_stds
    training_log_dict['target_means'] = [x.to('cpu') for x in target_means]

    model = BigConvModel(num_corrections, model_size_per_level, obstacle_parameter,
                         *get_transforms_and_orderings(DataSampler, num_corrections, device))
    model.to(device)

    model_npar = [sum([x.numel() for x in model.level_layers[i].parameters() if x.requires_grad]) for i in
                  range(num_corrections)]
    for level in range(num_corrections):
        print('Level {} num params: {}'.format(level, model_npar[level]))
    training_log_dict['actual_num_param'] = model_npar

    optimizer = Adam(model.parameters(), lr=1)
    scheduler = LambdaLR(optimizer, get_lr_schedule_func(num_epochs))

    min_val_mre = float('inf')
    for i in range(1, num_epochs + 1):
        t_start = time.perf_counter()
        print(i,"training")
        epoch_training_loss, training_inter_losses = train_epoch(model, optimizer, loss_function, train_data_ys,
                                                                 train_data_ys_obstacle, train_data_us, num_corrections,
                                                                 num_samples_per_level, device, batch_size)
        epoch_val_loss, val_inter_losses, val_mres, val_partial_mres = val_epoch(model, loss_function, target_stds,
                                                                                 target_means, val_data_ys,
                                                                                 val_data_ys_obstacle, val_data_us,
                                                                                 num_corrections, num_val_samples,
                                                                                 batch_size, device)
        t_end = time.perf_counter()

        if val_mres[-1] < min_val_mre:
            min_val_mre = val_mres[-1]
            torch.save(model.state_dict(), os.path.join(saving_path, 'model_{}.pt'.format(run_name)))

        training_log_dict['epoch'].append(i)
        training_log_dict['train_errors'].append(epoch_training_loss)
        training_log_dict['train_errors_per_level'].append(training_inter_losses)
        training_log_dict['val_errors'].append(epoch_val_loss)
        training_log_dict['val_errors_per_level'].append(val_inter_losses)
        training_log_dict['val_mrh1_per_level'].append(val_mres)
        training_log_dict['val_partial_mrh1_per_level'].append(val_partial_mres)

        np.save(training_log_dict_saving_path, training_log_dict, allow_pickle=True)

        print("Epoch [{}/{}] finished in {:.2f}s".format(i, num_epochs, t_end - t_start))
        print("Training Loss: {}".format(epoch_training_loss))
        print("Validation Loss: {}".format(epoch_val_loss))
        print("Validation MRE: {} (best: {})".format(val_mres[-1], min_val_mre))
        print('------')
        for level in range(num_corrections):
            print('Level {} Training Loss: {}'.format(level, training_inter_losses[level]))
            print('Level {} Val Loss: {}'.format(level, val_inter_losses[level]))
        print('------')
        for level in range(num_corrections):
            print('Level {} Val partial MR-H1 Loss: {}'.format(level, val_partial_mres[level]))
            print('Level {} Val MR-H1 Loss: {}'.format(level, val_mres[level]))
        print('------')

        scheduler.step()

    del val_data_ys, val_data_us, u_imgs_val, y_imgs_val, train_data_us, train_data_ys, u_imgs, y_imgs
    if obstacle_parameter:
        del val_data_ys_obstacle, train_data_ys_obstacle, y_imgs_obstacle

    # ys_test = DataSampler.load_ys_test(num_test_samples, data_loading_path)
    y_imgs_test = DataSampler.load_ys_images_test(num_test_samples, data_loading_path)
    if obstacle_parameter:
        y_imgs_obstacle = DataSampler.load_ys_obstacle_images_test(num_test_samples, data_loading_path)
    test_data_ys = build_pt_dataset(DataSampler, y_imgs_test, num_corrections, device)
    test_data_ys = [(x - y_img_mean) / y_img_std for x in test_data_ys]
    if obstacle_parameter:
        test_data_ys_obstacle = build_pt_obstacle_dataset(DataSampler, y_imgs_obstacle, num_corrections, device)
    else:
        test_data_ys_obstacle = None
    u_imgs_test = DataSampler.load_solutions_images_test(num_test_samples, data_loading_path)
    fine_u_imgs_test = DataSampler.load_solutions_images_test_fine(num_test_samples, data_loading_path)
    test_data_us = build_pt_dataset(DataSampler, u_imgs_test, num_corrections, device)
    test_data_us_fine = torch.as_tensor(fine_u_imgs_test).to(torch.float32)[:, None, :, :]

    model.load_state_dict(torch.load(os.path.join(saving_path, 'model_{}.pt'.format(run_name))))
    model.eval()
    print('Performing Test Epoch on {} samples'.format(num_test_samples))
    test_mres, test_partial_mres, test_mre_fine = test_epoch(model, loss_function, target_stds, target_means,
                                                             test_data_ys, test_data_ys_obstacle, test_data_us,
                                                             test_data_us_fine, num_corrections, num_test_samples,
                                                             batch_size, device)

    training_log_dict['test_mrh1_per_level'] = test_mres
    training_log_dict['test_partial_mrh1_per_level'] = test_partial_mres
    training_log_dict['test_mrh1_reference'] = test_mre_fine

    for level in range(num_corrections):
        print('Level {} Test partial MR-H1 Loss: {}'.format(level, test_partial_mres[level]))
        print('Level {} Test MR-H1 Loss: {}'.format(level, test_mres[level]))
    print('Reference Test MR-H1 Loss: {}'.format(test_mre_fine))

    loss_function_l2 = LossFunct(DataSampler, num_corrections, device, test_data_us, norm='l2')

    test_mres_l2, test_partial_mres_l2, test_mre_fine_l2 = test_epoch_l2(model, loss_function_l2, target_stds, target_means,
                                                                test_data_ys, test_data_ys_obstacle, test_data_us,
                                                                test_data_us_fine, num_corrections, num_test_samples,
                                                                batch_size, device)

    training_log_dict['test_mrl2_per_level'] = test_mres_l2
    training_log_dict['test_partial_mrl2_per_level'] = test_partial_mres_l2
    training_log_dict['test_mrl2_reference'] = test_mre_fine_l2

    for level in range(num_corrections):
        print('Level {} Test partial MR-L2 Loss: {}'.format(level, test_partial_mres_l2[level]))
        print('Level {} Test MR-L2 Loss: {}'.format(level, test_mres_l2[level]))
    print('Reference Test MR-L2 Loss: {}'.format(test_mre_fine_l2))

    np.save(training_log_dict_saving_path, training_log_dict, allow_pickle=True)


if __name__ == "__main__":
    
    """print("3 levels")
    num_corrections = 3
    perform_run('obstacle', 11, num_corrections, num_corrections*[100], 200, 24,
                10, 10, [8e+5 for i in range(num_corrections)], True, True, './code/FinalConvModel/ModelSaves',
                'obstacle11level3', torch.device('cpu'), './code/FinalConvModel/Data/obstacle11variable_test_data')"""
    
    num_corrections = 7
    perform_run('obstacle', 11, num_corrections, num_corrections*[10000], 200, 24,
                1024, 1024, [8e+5 for i in range(num_corrections)], True, True, './code/FinalConvModel/ModelSaves',
                'obstacle11level7', torch.device('cuda:0'), './code/FinalConvModel/Data/obstacle11variable')
    """
    perform_run('uniform', 10, 3, [3000, 3000, 3000], 20, 24,
                128, 128, [1e+4 for i in range(3)], False, True, './code/FinalConvModel/ModelSaves',
                'testuniform', torch.device('cpu'), './code/FinalConvModel/Data/uniform10')"""
    '''
    perform_run('cookie', 16, 4, [4000 for i in range(4)], 20, 24,
                256, 256, [4e+5 for i in range(4)], False, True, './code/FinalConvModel/ModelSaves',
                'testcookie', torch.device('cuda:0'), './code/FinalConvModel/Data/testcookie')

    '''


