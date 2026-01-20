import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from dolfin import *
import time
from tqdm import tqdm
import os
import pickle

from .Model import BigVCycleModel, BigFlatModel
from .DataHandler import MultiLevelRefinementSampler, load_problem_info_dict


def get_interpolations(sampler, u_imgs, level0, level1, device):
    row_inds, column_inds, values, shape = sampler.get_transformation_matrix_coo_format(level0, level1)
    inds_pt = torch.as_tensor(np.stack((row_inds, column_inds), axis=0).astype(np.int64))
    a_pt_sparse = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float64)), size=shape)

    order0 = torch.as_tensor(sampler.img_to_vec_order[level0].astype(np.long))
    order1 = torch.as_tensor(sampler.vec_to_img_order[level1].astype(np.long))

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

    order0 = torch.as_tensor(sampler.img_to_vec_order[level0].astype(np.long))
    order1 = torch.as_tensor(sampler.vec_to_img_order[level1].astype(np.long))

    u_vecs = u_imgs.view(u_imgs.shape[0], -1)[:, order0].permute(1, 0)
    new_u_vecs = torch.zeros((shape[0], u_vecs.shape[1]), dtype=u_vecs.dtype)
    for i, row_column_inds in enumerate(row_column_list):
        new_u_vecs[i, :] = torch.max(u_vecs[row_column_inds, :], dim=0)[0]
    u_vecs = new_u_vecs.permute(1, 0)

    u_imgs = u_vecs[:, order1].view(u_vecs.shape[0], *sampler.img_shape[level1])
    return u_imgs


def build_pt_dataset(sampler, imgs_numpy, num_corrections, device):
    return torch.as_tensor(imgs_numpy).to(torch.float32)[:, None, :, :]


def get_transforms_and_orderings(DataSampler, num_corrections, device):
    out_matrices = []
    out_orders_img_to_vec = []
    out_orders_vec_to_img = []
    img_sizes = []
    for i in range(num_corrections - 1):
        row_inds, column_inds, values, shape = DataSampler.get_transformation_matrix_coo_format(i, i + 1)
        inds_pt = torch.as_tensor(np.stack((row_inds, column_inds), axis=0).astype(np.int64))
        a_pt_sparse = torch.sparse_coo_tensor(inds_pt, torch.as_tensor(values.astype(np.float32)), size=shape)
        order0 = torch.as_tensor(DataSampler.img_to_vec_order[i].astype(np.long)).to(device)
        order1 = torch.as_tensor(DataSampler.vec_to_img_order[i+1].astype(np.long)).to(device)

        out_matrices.append(a_pt_sparse.to(device))
        out_orders_img_to_vec.append(order0)
        out_orders_vec_to_img.append(order1)
        img_sizes.append(DataSampler.img_shape[i + 1])

    return out_matrices, out_orders_img_to_vec, out_orders_vec_to_img, img_sizes


class LossFunct():
    def __init__(self, sampler, num_corrections, device, train_data_us, norm='h1'):
        self.num_corrections = num_corrections
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

            order = torch.as_tensor(sampler.img_to_vec_order[level].astype(np.long)).to(device)
            self.orderings.append(order)

        self.upsampling_matrices, _, self.orderings2, self.img_sizes = get_transforms_and_orderings(
            sampler, num_corrections, device)

        targets = train_data_us[:100].clone()
        self.target_mean = torch.mean(targets, dim=0)[None, ...].to(device)
        self.target_std = torch.sqrt(torch.mean((targets - self.target_mean.cpu())**2)).item()
        print(self.target_std)

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

    def get_overall_loss(self, pred, target):
        target = (target - self.target_mean) / self.target_std
        return self.get_single_loss(pred, target, self.num_corrections - 1)

    def get_norm_sq(self, x, level):
        x = x.clone().cpu().to(torch.float64)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)[:, self.orderings[level]].permute(1, 0)
        return torch.sum(torch.mm(self.a_matrices64[level], x) * x, dim=0)


def train_epoch(model, optimizer, loss_function, inputs, obstacle_inputs, targets, num_corrections,
                num_samples, device, batch_size):
    cumulative_loss = 0

    batches = np.random.permutation(np.arange(num_samples))
    batches = [batches[i:i+batch_size] for i in range(0, num_samples, batch_size)]

    for batch_indices in batches:
        input_batch = inputs[batch_indices].to(device)
        if obstacle_inputs is not None:
            input_batch_obstacle = obstacle_inputs[batch_indices].to(device)
        else:
            input_batch_obstacle = None

        target_batch = targets[batch_indices].to(device)

        outs = model.forward(input_batch, obstacle_levels=input_batch_obstacle)

        loss = loss_function.get_overall_loss(outs, target_batch)

        cumulative_loss += loss.data.item() / len(batches)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return cumulative_loss


def val_epoch(model, loss_function, target_std, target_mean, inputs, obstacle_inputs, targets, num_corrections,
              num_val_samples, batch_size, device):
    val_indices = np.arange(num_val_samples)
    val_loss = 0

    target_norms_sq = 0
    distance_norms_sq = 0

    for i in range(0, num_val_samples, batch_size):
        batch_indices = val_indices[i:i + batch_size]

        input_batch = inputs[batch_indices].to(device)

        if obstacle_inputs is not None:
            input_batch_obstacle = obstacle_inputs[batch_indices].to(device)
        else:
            input_batch_obstacle = None
        target_batch = targets[batch_indices].to(device)

        target_norms_sq += torch.mean(loss_function.get_norm_sq(target_batch, num_corrections - 1)) / np.ceil(num_val_samples / batch_size)

        with torch.no_grad():
            model_out = model(input_batch, obstacle_levels=input_batch_obstacle).detach()
            loss = loss_function.get_overall_loss(model_out, target_batch)
            outs = model_out * target_std + target_mean

        val_loss += loss.data.item() / np.ceil(num_val_samples / batch_size)

        distance_norms_sq += torch.mean(loss_function.get_norm_sq(outs - target_batch, num_corrections - 1)) / np.ceil(num_val_samples / batch_size)

    mre = torch.sqrt(distance_norms_sq / target_norms_sq).data.item()

    return val_loss, mre


def test_epoch(model, loss_function, target_std, target_mean, inputs, obstacle_inputs, target_imgs, target_imgs_fine,
               num_corrections, num_test_samples, batch_size, device):

    test_indices = np.arange(num_test_samples)
    target_norms_sq = 0
    distance_norms_sq = 0

    target_norms_sq_fine = 0
    distance_norms_sq_fine = 0

    for i in range(0, num_test_samples, batch_size):
        batch_indices = test_indices[i:i + batch_size]

        input_batch = inputs[batch_indices].to(device)
        if obstacle_inputs is not None:
            input_batch_obstacle = obstacle_inputs[batch_indices].to(device)
        else:
            input_batch_obstacle = None
        target_batch = target_imgs[batch_indices].to(device)
        target_batch_fine = target_imgs_fine[batch_indices].to(device)


        target_norms_sq += torch.mean(loss_function.get_norm_sq(target_batch, num_corrections - 1)) / np.ceil(num_test_samples / batch_size)
        target_norms_sq_fine += torch.mean(loss_function.get_norm_sq(target_batch_fine, num_corrections + 1)) \
                                / np.ceil(num_test_samples / batch_size)

        with torch.no_grad():
            model_out = model(input_batch, obstacle_levels=input_batch_obstacle).detach()
            outs = model_out * target_std + target_mean

        distance_norms_sq += torch.mean(loss_function.get_norm_sq(outs - target_batch, num_corrections - 1)) / np.ceil(num_test_samples / batch_size)
        fine_outs = loss_function.upsample(outs, num_corrections - 1)
        fine_outs = loss_function.upsample(fine_outs, num_corrections)
        distance_norms_sq_fine += torch.mean(loss_function.get_norm_sq(fine_outs - target_batch_fine, num_corrections + 1)) \
                                  / np.ceil(num_test_samples / batch_size)



    mre = torch.sqrt(distance_norms_sq / target_norms_sq).data.item()

    test_mre_fine = torch.sqrt(distance_norms_sq_fine / target_norms_sq_fine).data.item()

    return mre, test_mre_fine


def test_epoch_l2(model, loss_function, target_std, target_mean, inputs, obstacle_inputs, target_imgs, target_imgs_fine,
               num_corrections, num_test_samples, batch_size, device):
    test_indices = np.arange(num_test_samples)
    target_norms_sq = 0
    distance_norms_sq = 0

    target_norms_sq_fine = 0
    distance_norms_sq_fine = 0

    for i in range(0, num_test_samples, batch_size):
        batch_indices = test_indices[i:i + batch_size]

        input_batch = inputs[batch_indices].to(device)
        if obstacle_inputs is not None:
            input_batch_obstacle = obstacle_inputs[batch_indices].to(device)
        else:
            input_batch_obstacle = None
        target_batch = target_imgs[batch_indices].to(device)
        target_batch_fine = target_imgs_fine[batch_indices].to(device)

        target_norms_sq += torch.mean(loss_function.get_norm_sq(target_batch, num_corrections - 1)) / np.ceil(
            num_test_samples / batch_size)
        target_norms_sq_fine += torch.mean(loss_function.get_norm_sq(target_batch_fine, num_corrections + 1)) \
                                / np.ceil(num_test_samples / batch_size)

        with torch.no_grad():
            model_out = model(input_batch, obstacle_levels=input_batch_obstacle).detach()
            outs = model_out * target_std + target_mean

        distance_norms_sq += torch.mean(loss_function.get_norm_sq(outs - target_batch, num_corrections - 1)) / np.ceil(
            num_test_samples / batch_size)
        fine_outs = loss_function.upsample(outs, num_corrections - 1)
        fine_outs = loss_function.upsample(fine_outs, num_corrections)
        distance_norms_sq_fine += torch.mean(
            loss_function.get_norm_sq(fine_outs - target_batch_fine, num_corrections + 1)) \
                                  / np.ceil(num_test_samples / batch_size)

    mre = torch.sqrt(distance_norms_sq / target_norms_sq).data.item()

    test_mre_fine = torch.sqrt(distance_norms_sq_fine / target_norms_sq_fine).data.item()

    return mre, test_mre_fine


def get_lr_schedule_func(num_epochs, a=0.3, b=0.8, lr1=1e-3, lr2=2e-5):
    def learning_rate_schedule(epoch):
        if epoch < int(a * num_epochs):
            return lr1
        if epoch < int(b * num_epochs):
            alpha = (epoch - int(a * num_epochs)) / (int(b * num_epochs) - int(a * num_epochs))
            return (1 - alpha) * lr1 + alpha * lr2
        return lr2

    return learning_rate_schedule


def perform_run(problem, param_dimension, num_corrections, num_samples, num_epochs, batch_size,
                num_val_samples, num_test_samples, model_size, obstacle_parameter, fixed_radii, saving_path,
                run_name, device, data_loading_path, model_type='flat'):
    assert model_type == 'flat' or model_type == 'vcycles'
    run_name = run_name + model_type
    print('training on', device)
    input_params = {'problem': problem,
                    'param_dimension': param_dimension,
                    'num_corrections': num_corrections,
                    'num_samples': num_samples,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'num_val_samples': num_val_samples,
                    'num_test_samples': num_test_samples,
                    'model_size': model_size,
                    'obstacle_parameter': obstacle_parameter,
                    'fixed_radii': fixed_radii,
                    'saving_path': saving_path,
                    'run_name': run_name,
                    'data_loading_path': data_loading_path}


    training_log_dict = {'input_params': input_params,
                         'train_errors': [],
                         'val_errors': [],
                         'val_mrh1': [],
                         'epoch': []}
    training_log_dict_saving_path = os.path.join(saving_path, 'TrainingMetricsDict_{}.npy'.format(run_name))

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    # build data sampler
    DataSampler = pickle.load(open(os.path.join(data_loading_path, 'data_sampler.p'), 'rb'))
    DataSampler.reinit_sampler()


    # ys = DataSampler.load_ys_train(max_num_samples, data_loading_path)
    y_imgs = DataSampler.load_ys_images_train(num_samples, data_loading_path)
    if obstacle_parameter:
        y_imgs_obstacle = DataSampler.load_ys_obstacle_images_train(num_samples, data_loading_path)
    u_imgs = DataSampler.load_solutions_images_train(num_samples, data_loading_path)

    # normalize ys and generate multilevel decomposition
    y_img_mean = np.mean(y_imgs)
    y_img_std = np.sqrt(np.mean((y_imgs - y_img_mean)**2))
    training_log_dict['y_img_mean'] = y_img_mean
    training_log_dict['y_img_std'] = y_img_std

    train_data_ys = build_pt_dataset(DataSampler, y_imgs, num_corrections, device)
    train_data_ys = (train_data_ys - y_img_mean) / y_img_std
    if obstacle_parameter:
        train_data_ys_obstacle = build_pt_dataset(DataSampler, y_imgs_obstacle, num_corrections, device)
    else:
        train_data_ys_obstacle = None
    train_data_us = build_pt_dataset(DataSampler, u_imgs, num_corrections, device)

    # sample ys and fine us for validation dataset
    # ys_val = DataSampler.load_ys_val(num_val_samples, data_loading_path)
    y_imgs_val = DataSampler.load_ys_images_val(num_val_samples, data_loading_path)
    if obstacle_parameter:
        y_imgs_obstacle = DataSampler.load_ys_obstacle_images_val(num_val_samples, data_loading_path)
    u_imgs_val = DataSampler.load_solutions_images_val(num_val_samples, data_loading_path)

    # normalize validation dataset with params from training set
    val_data_ys = build_pt_dataset(DataSampler, y_imgs_val, num_corrections, device)
    val_data_ys = (val_data_ys - y_img_mean) / y_img_std
    if obstacle_parameter:
        val_data_ys_obstacle = build_pt_dataset(DataSampler, y_imgs_obstacle, num_corrections, device)
    else:
        val_data_ys_obstacle = None
    val_data_us = build_pt_dataset(DataSampler, u_imgs_val, num_corrections, device)

    # build loss function and get target normalization parameters
    loss_function = LossFunct(DataSampler, num_corrections, device, train_data_us)
    target_std = loss_function.target_std
    target_mean = loss_function.target_mean
    training_log_dict['target_std'] = target_std
    training_log_dict['target_mean'] = target_mean.to('cpu')

    if model_type == 'flat':
        model = BigFlatModel(num_corrections, model_size, obstacle_parameter)
    else:
        model = BigVCycleModel(num_corrections, model_size, obstacle_parameter)
    model.to(device)

    model_npar = sum([x.numel() for x in model.parameters() if x.requires_grad])
    print('Num params: {}'.format(model_npar))
    training_log_dict['actual_num_param'] = model_npar

    optimizer = Adam(model.parameters(), lr=1)
    scheduler = LambdaLR(optimizer, get_lr_schedule_func(num_epochs))

    min_val_mre = float('inf')
    for i in range(1, num_epochs + 1):
        t_start = time.perf_counter()
        epoch_training_loss = train_epoch(model, optimizer, loss_function, train_data_ys,
                                          train_data_ys_obstacle, train_data_us, num_corrections,
                                          num_samples, device, batch_size)
        epoch_val_loss, val_mre = val_epoch(model, loss_function, target_std, target_mean, val_data_ys,
                                             val_data_ys_obstacle, val_data_us, num_corrections, num_val_samples,
                                             batch_size, device)
        t_end = time.perf_counter()

        if val_mre < min_val_mre:
            min_val_mre = val_mre
            torch.save(model.state_dict(), os.path.join(saving_path, 'model_{}.pt'.format(run_name)))

        training_log_dict['epoch'].append(i)
        training_log_dict['train_errors'].append(epoch_training_loss)
        training_log_dict['val_errors'].append(epoch_val_loss)
        training_log_dict['val_mrh1'].append(val_mre)

        np.save(training_log_dict_saving_path, training_log_dict, allow_pickle=True)

        print("Epoch [{}/{}] finished in {:.2f}s".format(i, num_epochs, t_end - t_start))
        print("Training Loss: {}".format(epoch_training_loss))
        print("Validation Loss: {}".format(epoch_val_loss))
        print("Validation MRE: {} (best: {})".format(val_mre, min_val_mre))
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
    test_data_ys = (test_data_ys - y_img_mean) / y_img_std
    if obstacle_parameter:
        test_data_ys_obstacle = build_pt_dataset(DataSampler, y_imgs_obstacle, num_corrections, device)
    else:
        test_data_ys_obstacle = None
    u_imgs_test = DataSampler.load_solutions_images_test(num_test_samples, data_loading_path)
    fine_u_imgs_test = DataSampler.load_solutions_images_test_fine(num_test_samples, data_loading_path)
    test_data_us = build_pt_dataset(DataSampler, u_imgs_test, num_corrections, device)
    test_data_us_fine = torch.as_tensor(fine_u_imgs_test).to(torch.float32)[:, None, :, :]

    model.load_state_dict(torch.load(os.path.join(saving_path, 'model_{}.pt'.format(run_name))))
    model.eval()
    print('Performing Test Epoch on {} samples'.format(num_test_samples))
    test_mre, test_mre_fine = test_epoch(model, loss_function, target_std, target_mean,
                                         test_data_ys, test_data_ys_obstacle, test_data_us,
                                         test_data_us_fine, num_corrections, num_test_samples,
                                         batch_size, device)

    training_log_dict['test_mrh1'] = test_mre
    training_log_dict['test_mrh1_reference'] = test_mre_fine

    print('Test MR-H1 Loss: {}'.format(test_mre))
    print('Reference Test MR-H1 Loss: {}'.format(test_mre_fine))

    loss_function_l2 = LossFunct(DataSampler, num_corrections, device, test_data_us, norm='l2')

    test_mre_l2, test_mre_fine_l2 = test_epoch_l2(model, loss_function_l2, target_std, target_mean,
                                                                test_data_ys, test_data_ys_obstacle, test_data_us,
                                                                test_data_us_fine, num_corrections, num_test_samples,
                                                                batch_size, device)

    training_log_dict['test_mrl2'] = test_mre_l2
    training_log_dict['test_mrl2_reference'] = test_mre_fine_l2

    print('Test MR-L2 Loss: {}'.format(test_mre_l2))
    print('Reference Test MR-L2 Loss: {}'.format(test_mre_fine_l2))

    np.save(training_log_dict_saving_path, training_log_dict, allow_pickle=True)


if __name__ == "__main__":

    '''
    # process 37348
    perform_run('uniform', 10, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'uniform10OnlyVcycles', torch.device('cuda:0'), './code/FinalConvModel/Data/uniform10',
                model_type='vcycles')
    

    # process 37520
    perform_run('uniform', 50, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'uniform50OnlyVcycles', torch.device('cuda:1'), './code/FinalConvModel/Data/uniform50',
                model_type='vcycles')
    
    # process 37740
    perform_run('uniform', 100, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'uniform100OnlyVcycles', torch.device('cuda:2'), './code/FinalConvModel/Data/uniform100',
                model_type='vcycles')
    
    # process 37843
    perform_run('uniform', 200, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'uniform200OnlyVcycles', torch.device('cuda:3'), './code/FinalConvModel/Data/uniform200',
                model_type='vcycles')
    '''
    '''
    
    
    # process 20261
    perform_run('log-normal', 10, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'lognormal10OnlyVcycles', torch.device('cuda:0'), './code/FinalConvModel/Data/lognormal10',
                model_type='vcycles')
    

    # process 20373
    perform_run('log-normal', 50, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'lognormal50OnlyVcycles', torch.device('cuda:1'), './code/FinalConvModel/Data/lognormal50',
                model_type='vcycles')
    
    # process 20489
    perform_run('log-normal', 100, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100OnlyVcycles', torch.device('cuda:2'), './code/FinalConvModel/Data/lognormal100',
                model_type='vcycles')
    
    # process 34892
    perform_run('log-normal', 200, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'lognormal200OnlyVcycles', torch.device('cuda:0'), './code/FinalConvModel/Data/lognormal200',
                model_type='vcycles')

    # same as above
    perform_run('cookie', 16, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'cookie16OnlyVcycles', torch.device('cuda:0'), './code/FinalConvModel/Data/cookie16fixed',
                model_type='vcycles')
    
    # process 35008
    perform_run('cookie', 32, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, False, './code/FinalConvModel/ModelSaves',
                'cookie32OnlyVcycles', torch.device('cuda:1'), './code/FinalConvModel/Data/cookie32variable',
                model_type='vcycles')
    
    # process 35114
    perform_run('cookie', 64, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'cookie64OnlyVcycles', torch.device('cuda:2'), './code/FinalConvModel/Data/cookie64fixed',
                model_type='vcycles')
    
    # process 35220
    perform_run('cookie', 128, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'cookie128OnlyVcycles', torch.device('cuda:3'), './code/FinalConvModel/Data/cookie128variable',
                model_type='vcycles')
    '''
    '''
    '''

    '''
    # process 41738
    perform_run('log-normal', 100, 7, 232, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100OnlyVcycles', torch.device('cuda:0'), './code/FinalConvModel/Data/lognormal100',
                model_type='vcycles')
    
    # process 41578
    perform_run('uniform', 100, 7, 232, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'uniform100OnlyVcycles', torch.device('cuda:1'), './code/FinalConvModel/Data/uniform100',
                model_type='vcycles')
    '''




    '''
    # process 2752
    for i in range(2):
        perform_run('uniform', 10, 7, 10000, 200, 8,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'uniform10OnlyVcycles_extra{}'.format(i), torch.device('cuda:0'), './code/FinalConvModel/Data/uniform10',
                    model_type='vcycles')
                    
        perform_run('log-normal', 10, 7, 10000, 200, 8,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'lognormal10OnlyVcycles_extra{}'.format(i), torch.device('cuda:0'), './code/FinalConvModel/Data/lognormal10',
                    model_type='vcycles')
                    
        perform_run('cookie', 16, 7, 10000, 200, 8,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'cookie16OnlyVcycles_extra{}'.format(i), torch.device('cuda:0'), './code/FinalConvModel/Data/cookie16fixed',
                    model_type='vcycles')
    

    # process 2867
    for i in range(2):
        perform_run('uniform', 50, 7, 10000, 200, 8,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'uniform50OnlyVcycles_extra{}'.format(i), torch.device('cuda:1'), './code/FinalConvModel/Data/uniform50',
                    model_type='vcycles')
                    
        perform_run('log-normal', 50, 7, 10000, 200, 8,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'lognormal50OnlyVcycles_extra{}'.format(i), torch.device('cuda:1'), './code/FinalConvModel/Data/lognormal50',
                    model_type='vcycles')
                    
        perform_run('cookie', 32, 7, 10000, 200, 8,
                    1024, 1024, 5.6e+6, False, False, './code/FinalConvModel/ModelSaves',
                    'cookie32OnlyVcycles_extra{}'.format(i), torch.device('cuda:1'), './code/FinalConvModel/Data/cookie32variable',
                    model_type='vcycles')
    
    # process 3000
    for i in range(2):
        perform_run('uniform', 100, 7, 10000, 200, 8,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'uniform100OnlyVcycles_extra{}'.format(i), torch.device('cuda:2'), './code/FinalConvModel/Data/uniform100',
                    model_type='vcycles')
                    
        perform_run('log-normal', 100, 7, 10000, 200, 8,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'lognormal100OnlyVcycles_extra{}'.format(i), torch.device('cuda:2'), './code/FinalConvModel/Data/lognormal100',
                    model_type='vcycles')
                    
        perform_run('cookie', 64, 7, 10000, 200, 8,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'cookie64OnlyVcycles_extra{}'.format(i), torch.device('cuda:2'), './code/FinalConvModel/Data/cookie64fixed',
                    model_type='vcycles')
    
    # process 3128
    for i in range(2):
        perform_run('uniform', 200, 7, 10000, 200, 8,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'uniform200OnlyVcycles_extra{}'.format(i), torch.device('cuda:3'), './code/FinalConvModel/Data/uniform200',
                    model_type='vcycles')
                    
        perform_run('log-normal', 200, 7, 10000, 200, 8,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'lognormal200OnlyVcycles_extra{}'.format(i), torch.device('cuda:3'), './code/FinalConvModel/Data/lognormal200',
                    model_type='vcycles')
                    
        perform_run('cookie', 128, 7, 10000, 200, 8,
                    1024, 1024, 5.6e+6, False, False, './code/FinalConvModel/ModelSaves',
                    'cookie128OnlyVcycles_extra{}'.format(i), torch.device('cuda:3'), './code/FinalConvModel/Data/cookie128variable',
                    model_type='vcycles')
    
    perform_run('cookie', 128, 7, 10000, 200, 8,
                1024, 1024, 5.6e+6, False, False, './code/FinalConvModel/ModelSaves',
                'cookie128OnlyVcycles', torch.device('cuda:3'), './code/FinalConvModel/Data/cookie128variable',
                model_type='vcycles')
    '''
    '''
    

    # process 34970
    perform_run('log-normal', 100, 7, 232, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100OnlyVcycles_extra1', torch.device('cuda:3'), './code/FinalConvModel/Data/lognormal100',
                model_type='vcycles')

    perform_run('log-normal', 100, 7, 232, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100OnlyVcycles_extra2', torch.device('cuda:3'), './code/FinalConvModel/Data/lognormal100',
                model_type='vcycles')

    
    # process 35501
    perform_run('uniform', 100, 7, 232, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'uniform100OnlyVcycles_extra1', torch.device('cuda:2'), './code/FinalConvModel/Data/uniform100',
                model_type='vcycles')

    perform_run('uniform', 100, 7, 232, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'uniform100OnlyVcycles_extra2', torch.device('cuda:2'), './code/FinalConvModel/Data/uniform100',
                model_type='vcycles')
    '''
    '''
    
    #process 44605
    perform_run('uniform', 100, 7, 232, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'uniform100OnlyVcycles_extra3_232', torch.device('cuda:1'), './code/FinalConvModel/Data/uniform100',
                model_type='vcycles')

    perform_run('log-normal', 100, 7, 232, 200, 8,
                1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100OnlyVcycles_extra3_232', torch.device('cuda:1'), './code/FinalConvModel/Data/lognormal100',
                model_type='vcycles')
    
    # process 24306
    for i in range(1, 4):
        perform_run('uniform', 100, 7, 10000, 200, 2,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'uniform100_flat_extra{}'.format(i), torch.device('cuda:0'), './code/FinalConvModel/Data/uniform100',
                    model_type='flat')
    
    # process 24405
    for i in range(1, 4):
        perform_run('uniform', 200, 7, 10000, 200, 2,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'uniform200_flat_extra{}'.format(i), torch.device('cuda:1'), './code/FinalConvModel/Data/uniform200',
                    model_type='flat')
    
    # process 
    for i in range(1, 4):
        perform_run('log-normal', 100, 7, 232, 200, 2,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'lognormal100_flat_extra{}'.format(i), torch.device('cuda:2'),
                    './code/FinalConvModel/Data/lognormal100',
                    model_type='flat')

    # process
    for i in range(1, 4):
        perform_run('log-normal', 200, 7, 232, 200, 2,
                    1024, 1024, 5.6e+6, False, True, './code/FinalConvModel/ModelSaves',
                    'lognormal200_flat_extra{}'.format(i), torch.device('cuda:3'),
                    './code/FinalConvModel/Data/lognormal200',
                    model_type='flat')
    '''
    '''
    '''
    # process 6939
    for i in range(1, 4):
        perform_run('cookie', 32, 7, 232, 200, 2,
                    1024, 1024, 5.6e+6, False, False, './code/FinalConvModel/ModelSaves',
                    'cookie32_decay_vcycle_extra{}'.format(i), torch.device('cuda:2'),
                    './code/FinalConvModel/Data/cookie32variable',
                    model_type='vcycles')

