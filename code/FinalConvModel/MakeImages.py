import os, time
import numpy as np
import torch
from dolfin import *

from .Model import BigConvModel
from .DataHandler import MultiLevelRefinementSampler
from .TrainModel import build_pt_obstacle_dataset, LossFunct
from .BuildDataSets import build_problem

from matplotlib import cm
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['figure.figsize'] = 3.5,3.5#3.5*cm, 3.5*cm
import matplotlib.pyplot as plt

prob = "obstacle11variable"#"obstacle10rough"#"obstacle100rough"#"obstacle220rough"#"obstacle50rough"#
pro = "obstacle-rough"#"obstacle"#
data_loading_path = './code/FinalConvModel/Data/'+prob
obstacle_parameter = False#True#


NUM_REFINEMENTS = 7
FINE_REFINEMENT_IND = 7

fixed_radii = None
problem_info_dict = build_problem(pro, NUM_REFINEMENTS, NUM_REFINEMENTS, obstacle_parameter, fixed_radii)
# problem_info_dict = "./code/ParamPDE/samples-darcy-1"

device = torch.device("cpu")#torch.device('cuda:2')

DataSampler = MultiLevelRefinementSampler(problemInfo=problem_info_dict,
                                          overwrite_levels=FINE_REFINEMENT_IND)

max_num_samples = 100
ind = 0
y_imgs = DataSampler.load_ys_images_train(max_num_samples, data_loading_path)
print("y min max", y_imgs.min(), y_imgs.max())
ys = DataSampler.load_ys_train( max_num_samples, data_loading_path)
print("y min max", ys.min(), ys.max())
print("y samples loaded:", y_imgs.shape)
if obstacle_parameter:
    y_imgs_obstacle = DataSampler.load_ys_obstacle_images_train(max_num_samples, data_loading_path)
    print("y obs samples loaded:", y_imgs_obstacle.shape, y_imgs_obstacle[ind,10,10])
    print("y obs range:", np.min(y_imgs_obstacle[:,10,10]), np.max(y_imgs_obstacle[:,10,10]))
if pro == "obstacle-rough":
    y_imgs_obstacle = y_imgs
u_imgs = DataSampler.load_solutions_images_train(max_num_samples, data_loading_path)
print("u samples loaded:", u_imgs.shape)

h = 1/(u_imgs.shape[-1])
X = np.arange(0, 1, h)
Y = np.arange(0, 1, h)
X, Y = np.meshgrid(X, Y)
for ind in range(10):
    print("X,Y", X.shape, Y.shape)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(elev=10, azim=45)
    c = ax.plot_surface(X, Y, u_imgs[ind], cmap=cm.coolwarm, linewidth=0.1, antialiased=False)
    ax.plot_surface(X,Y, y_imgs_obstacle[ind])
    # plt.colorbar(c, shrink=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.locator_params(axis='y', nbins=1)
    plt.locator_params(axis='x', nbins=1)
    plt.title(r'Solution $u$')
    plt.tight_layout()
    plt.savefig("code/FinalConvModel/Images/"+prob+"/u_3d_"+str(ind)+".png")
    plt.clf()


target_list = build_pt_obstacle_dataset(DataSampler, u_imgs, NUM_REFINEMENTS, "cpu")
loss = LossFunct(DataSampler, NUM_REFINEMENTS, "cpu", target_list, small_mode=True)
print("shapes", [l.shape for l in target_list])
for ref in range(4):
    c = plt.imshow(target_list[ref][ind][0], origin="lower")#[1:-1,1:-1]
    plt.colorbar(c, shrink=0.7)
    plt.title(fr'Solution on level ${ref+1}$')
    plt.tight_layout()
    plt.savefig("code/FinalConvModel/Images/"+prob+"/Solution_level_"+str(ref)+".png")
    plt.clf()
    f = DataSampler.image_to_function(target_list[ref][ind][0].detach().cpu().numpy(), ref)
    c = plot(f, extend='max')
    plt.colorbar(c, shrink=0.7)
    plt.title(fr'Solution on level ${ref+1}$')
    plt.tight_layout()
    plt.savefig("code/FinalConvModel/Images/"+prob+"/Solution_function_level_"+str(ref)+".png")
    plt.clf()
ref=4

kappa = DataSampler.image_to_function(y_imgs[ind], NUM_REFINEMENTS-1)
c = plot(kappa, extend='max')
d = plt.colorbar(c, shrink=0.7)
d.ax.locator_params(nbins=3)
d.ax.tick_params(labelsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.locator_params(axis='y', nbins=1)
plt.locator_params(axis='x', nbins=1)
plt.title(r'Parameter field $\kappa$')
plt.tight_layout()
plt.savefig("code/FinalConvModel/Images/"+prob+"/kappa.png")
plt.clf()

u = DataSampler.image_to_function((u_imgs[ind]>y_imgs_obstacle[ind])*1, NUM_REFINEMENTS-1)
c = plot(u, extend='max')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.locator_params(axis='y', nbins=1)
plt.locator_params(axis='x', nbins=1)
plt.title(r"Contact set")
plt.tight_layout()
plt.savefig("code/FinalConvModel/Images/"+prob+"/domain.png")
plt.clf()


u = DataSampler.image_to_function(y_imgs_obstacle[ind], NUM_REFINEMENTS-1)
c = plot(u, extend='max')
# d = plt.colorbar(c, shrink=0.7)
# d.ax.locator_params(nbins=3)
# d.ax.tick_params(labelsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.locator_params(axis='y', nbins=1)
plt.locator_params(axis='x', nbins=1)
plt.title(fr"Obstacle $\pi\equiv {np.round(y_imgs_obstacle[ind][5,5], decimals=3)}$")
plt.tight_layout()
plt.savefig("code/FinalConvModel/Images/"+prob+"/obstacle.png")
plt.clf()


new_target_list = [target_list[0]]
for i in range(1, NUM_REFINEMENTS):
    new_target_list.append(target_list[i].detach().clone() - loss.upsample(target_list[i - 1].detach().clone(), i - 1))
for i,cor in enumerate(new_target_list[:ref]):
    print("shape", cor.shape)
    c = plt.imshow(cor[ind][0], origin="lower")#[1:-1,1:-1]
    plt.colorbar(c, shrink=0.7)
    plt.title(fr'Correction ${i+1}$')
    plt.tight_layout()
    plt.savefig("code/FinalConvModel/Images/"+prob+"/Correction_"+str(i)+".png")
    plt.clf()

    f = DataSampler.image_to_function(cor[ind][0].detach().cpu().numpy(), i)
    c = plot(f, extend='max')
    # if i<4:
    #     plot(DataSampler.problem.mesh[i])
    plt.colorbar(c, shrink=0.7)
    plt.title(fr'Correction ${i+1}$')
    plt.tight_layout()
    plt.savefig("code/FinalConvModel/Images/"+prob+"/Correction_function_"+str(i)+".png")
    plt.clf()

u = DataSampler.image_to_function(u_imgs[ind], NUM_REFINEMENTS-1)
c = plot(u, extend='max')
plt.colorbar(c)
plt.title(r'Solution $u$')
plt.tight_layout()
plt.savefig("code/FinalConvModel/Images/"+prob+"/u.png")
plt.clf()


u = DataSampler.image_to_function(np.mean((u_imgs>y_imgs_obstacle)*1, axis=0), NUM_REFINEMENTS-1)
c = plot(u, extend='max')
plt.colorbar(c)
plt.title(r"Contact set mean")
plt.tight_layout()
plt.savefig("code/FinalConvModel/Images/"+prob+"/domain_mean.png")
plt.clf()
