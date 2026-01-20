import numpy as np
import torch
from torch.nn import Conv2d, ReLU, Module, ModuleList, ConvTranspose2d, Softplus, Sequential, PixelShuffle


class VCycleLayer(Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        self.depth = depth

        if depth > -1:#0:
            self.pre_smoothing = Sequential(Conv2d(in_channels, out_channels, 3, padding=1), ReLU())
            self.down_sampling = Sequential(Conv2d(out_channels, out_channels, 5, stride=2, padding=2), ReLU())
            self.recursive_layer = VCycleLayer(out_channels, out_channels, depth - 1)
            self.up_sampling = Sequential(ConvTranspose2d(out_channels, out_channels, 5, stride=2, padding=1), ReLU())
            self.post_smoothing = Sequential(Conv2d(2 * out_channels, out_channels, 3, padding=1), ReLU())
        else:
            self.pre_smoothing = Sequential(Conv2d(in_channels, out_channels, 3, padding=1), ReLU(),
                                            Conv2d(out_channels, out_channels, 3, padding=1), ReLU())

    def forward(self, x):
        x = self.pre_smoothing(x)
        if self.depth > -1:#0:
            z = self.down_sampling(x)
            z = self.recursive_layer(z)
            z = self.up_sampling(z)[:, :, :x.shape[2], :x.shape[3]]
            x = torch.cat((x, z), dim=1)
            x = self.post_smoothing(x)
        return x


class LevelLayer(Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_cycles, depth):
        super().__init__()

        self.cycles = ModuleList()
        self.cycles.append(VCycleLayer(in_channels, inter_channels, depth))
        for i in range(num_cycles - 2):
            self.cycles.append(VCycleLayer(inter_channels, inter_channels, depth))
        self.cycles.append(VCycleLayer(inter_channels, out_channels, depth))

        self.in_out_skip_conv = Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x_orig = x.clone()
        x = self.cycles[0](x)
        for v_cycle in self.cycles[1:-1]:
            x = v_cycle(x) + x
        x = self.cycles[-1](x) + self.in_out_skip_conv(x_orig)
        return x


class BigVCycleModel(Module):
    def __init__(self, num_levels, model_size, obstacle_channel):
        super().__init__()
        self.num_levels = num_levels

        self.layers = ModuleList()

        in_channels = 2 if obstacle_channel else 1
        tmp_model = VCycleLayer(64, 64, num_levels - 1)
        num_params_in_cycle = sum([p.numel() for p in tmp_model.parameters() if p.requires_grad])
        num_cycles = int(np.ceil(model_size / num_params_in_cycle))
        print(num_cycles)

        self.layers.append(VCycleLayer(in_channels, 64, num_levels - 1))
        for i in range(num_cycles - 1):
            self.layers.append(VCycleLayer(64, 64, num_levels - 1))
        self.layers.append(Conv2d(64, 1, 1))

    def forward(self, x, obstacle_levels=None):
        if obstacle_levels is not None:
            x = torch.cat((x, obstacle_levels), dim=1)
        for i, layer in enumerate(self.layers):
            if 0 < i and i < len(self.layers) - 1:
                x = layer(x) + x
            else:
                x = layer(x)
        return x


class BigFlatModel(Module):
    def __init__(self, num_levels, model_size, obstacle_channel):
        super().__init__()
        self.num_levels = num_levels

        self.layers = ModuleList()

        in_channels = 2 if obstacle_channel else 1
        tmp_model = Conv2d(128, 128, 3, padding=1)
        num_params_in_cycle = sum([p.numel() for p in tmp_model.parameters() if p.requires_grad])
        num_cycles = int(np.ceil(model_size / num_params_in_cycle))
        print(num_cycles)

        self.layers.append(Sequential(Conv2d(in_channels, 128, 3, padding=1), ReLU()))
        for i in range(num_cycles - 1):
            self.layers.append(Sequential(Conv2d(128, 128, 3, padding=1), ReLU()))
        self.layers.append(Conv2d(128, 1, 3, padding=1))

    def forward(self, x, obstacle_levels=None):
        if obstacle_levels is not None:
            x = torch.cat((x, obstacle_levels), dim=1)
        for i, layer in enumerate(self.layers):
            if 0 < i and i < len(self.layers) - 1:
                x = layer(x) + x
            else:
                x = layer(x)
        return x


class BigConvModel(Module):
    def __init__(self, num_levels, model_size_per_level, obstacle_channel,
                 upsampling_matrices, upsampling_img_to_vec, upsampling_vec_to_img, img_shapes):
        super().__init__()
        self.num_levels = num_levels

        self.upsampling_matrices = upsampling_matrices
        self.upsampling_vec_to_img = upsampling_vec_to_img
        self.upsampling_img_to_vec = upsampling_img_to_vec
        self.img_shapes = img_shapes

        self.level_layers = ModuleList()
        for i in range(num_levels):
            in_channels = 32 if i == 0 else 64

            level0_size = 32#64 if i == 0 else 32

            tmp_model = VCycleLayer(level0_size, level0_size, i)
            num_params_in_cycle = sum([p.numel() for p in tmp_model.parameters() if p.requires_grad])
            num_cycles = min(int(np.ceil(model_size_per_level[i] / num_params_in_cycle)), 16)
            print('{} Cycles on Level {}'.format(num_cycles, i))

            self.level_layers.append(LevelLayer(in_channels=in_channels, out_channels=32, inter_channels=level0_size,
                                                num_cycles=num_cycles, depth=i))

        self.downsampling_layers = ModuleList()
        in_channels = 2 if obstacle_channel else 1
        out_channels = 31 if obstacle_channel else 32
        self.downsampling_layers.append(ModuleList([Conv2d(in_channels, out_channels, 1, padding=0), ReLU()]))
        for i in range(1, num_levels):
            level_downsampling_layers = ModuleList()
            level_downsampling_layers.append(Conv2d(in_channels, 32, 1, padding=0))
            level_downsampling_layers.append(ReLU())
            for j in range(i - 1):
                level_downsampling_layers.append(Conv2d(32, 32, 3, stride=2, padding=1))
                level_downsampling_layers.append(ReLU())
            level_downsampling_layers.append(Conv2d(32, out_channels, 3, stride=2, padding=1))
            level_downsampling_layers.append(ReLU())
            self.downsampling_layers.append(level_downsampling_layers)


        self.upsampling_layers = ModuleList()
        for i in range(num_levels - 1):
            upsampling_channels = 32
            self.upsampling_layers.append(Sequential(ConvTranspose2d(32, upsampling_channels, 5, stride=2, padding=1),
                                                     ReLU()))

        self.output_layers = ModuleList()
        self.output_layers.append(Conv2d(32, 1, 3, padding=1))
        for i in range(1, num_levels):
            self.output_layers.append(Sequential(Conv2d(32, 4, 5, padding=2, stride=2),
                                                 PixelShuffle(2)))

    def upsample(self, img_batch, level):
        order0 = self.upsampling_img_to_vec[level]
        order1 = self.upsampling_vec_to_img[level]
        a_pt_sparse = self.upsampling_matrices[level]
        img_shape = self.img_shapes[level]

        img_batch = img_batch.view(img_batch.shape[0], -1)[:, order0].permute(1, 0)
        img_batch = a_pt_sparse.mm(img_batch).permute(1, 0)
        img_batch = img_batch[:, order1].view(img_batch.shape[0], 1, *img_shape)
        return img_batch

    def forward(self, x_levels, obstacle_levels=None, num_levels=None):
        if num_levels is None:
            num_levels = self.num_levels

        if obstacle_levels is not None:
            x_levels = [torch.cat((x, y), dim=1) for x, y in zip(x_levels, obstacle_levels)]
        else:
            x_levels = [x.clone() for x in x_levels]

        x_L = x_levels[-1]
        for level, level_downsampling_layers in enumerate(self.downsampling_layers):
            x_extra = x_L.clone()
            for layer in level_downsampling_layers:
                x_extra = layer(x_extra)
            if obstacle_levels is not None:
                x_levels[self.num_levels - level - 1] = torch.cat((obstacle_levels[num_levels - level - 1], x_extra), dim=1)
            else:
                x_levels[self.num_levels - level - 1] = x_extra#torch.cat((x_levels[num_levels - level - 1], x_extra), dim=1)

        out_levels = []
        #summed_solution = None
        level_input_data = x_levels[0]
        for i in range(num_levels):
            intermediate = self.level_layers[i](level_input_data)
            out = self.output_layers[i](intermediate)[:, :, :intermediate.shape[2], :intermediate.shape[3]]
            out = out.clone()

            out_levels.append(out)

            if i < num_levels - 1:

                w, h = x_levels[i + 1].shape[2:4]
                up_sampled_data = self.upsampling_layers[i](intermediate)[:, :, :w, :h]
                level_input_data = torch.cat((up_sampled_data, x_levels[i+1]), dim=1)

        return out_levels




