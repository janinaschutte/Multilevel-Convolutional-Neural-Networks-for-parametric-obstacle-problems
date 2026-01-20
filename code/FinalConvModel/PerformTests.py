import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from dolfin import *
import time

from .Model import BigConvModel
from .DataHandler import MultiLevelRefinementSampler
from .TrainModel import perform_run


if __name__ == "__main__":
    '''
    perform_run('log-normal', 100, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100level7', torch.device('cuda:2'), './code/FinalConvModel/Data/lognormal100')
    
    perform_run('uniform', 10, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform10level7_onlyLin', torch.device('cuda:0'), './code/FinalConvModel/Data/uniform10')
    '''

    '''
    for decay in [1.2, 0.9, 0.7, 0.5]:
        num_params = decay ** np.arange(7)
        num_params *= 7 * (8e+5) / np.sum(num_params)
        num_params = [int(x) for x in num_params]
        print(decay, num_params)

        if decay > 1:
            batch_size = 16
        else:
            batch_size = 20

        perform_run('log-normal', 100, 7, [10000 for i in range(7)], 200, batch_size,
               1024, 1024, num_params, False, True, './code/FinalConvModel/ModelSaves',
               'lognormal100level7_decay{}'.format(decay), torch.device('cuda:2'), './code/FinalConvModel/Data/lognormal100')

        perform_run('uniform', 10, 7, [10000 for i in range(7)], 200, batch_size,
               1024, 1024, num_params, False, True, './code/FinalConvModel/ModelSaves',
               'uniform10level7_decay{}'.format(decay), torch.device('cuda:2'), './code/FinalConvModel/Data/uniform10')

    '''


    '''
    perform_run('uniform', 50, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform50level7', torch.device('cuda:0'), './code/FinalConvModel/Data/uniform50')
    
    
    perform_run('uniform', 100, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform100level7', torch.device('cuda:1'), './code/FinalConvModel/Data/uniform100')
    
    perform_run('uniform', 200, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform200level7', torch.device('cuda:2'), './code/FinalConvModel/Data/uniform200')
    
    perform_run('cookie', 16, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'cookie16fixedlevel7_onlyLin', torch.device('cuda:1'), './code/FinalConvModel/Data/cookie16fixed')
    '''
    '''
    ### DEBUG

    perform_run('cookie', 16, 7, [10000 for i in range(7)], 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'cookie16fixedlevel7', torch.device('cuda:0'), './code/FinalConvModel/Data/cookie16fixed')
    
    perform_run('uniform', 10, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform10level7', torch.device('cuda:3'), './code/FinalConvModel/Data/uniform10')
    
    ### END DEBUG
    '''

    ### Now come the real tests!!

    '''
    # process 27698
    perform_run('cookie', 16, 7, [10000 for i in range(7)], 200, 18,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'cookie16fixedlevel7', torch.device('cuda:0'), './code/FinalConvModel/Data/cookie16fixed')

    perform_run('obstacle', 10, 7, [10000 for i in range(7)], 200, 18,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'obstacle10level7', torch.device('cuda:0'), './code/FinalConvModel/Data/obstacle10')

    
    # process 28072
    perform_run('cookie', 64, 7, [10000 for i in range(7)], 200, 18,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'cookie64fixedlevel7', torch.device('cuda:1'), './code/FinalConvModel/Data/cookie64fixed')

    perform_run('obstacle', 50, 7, [10000 for i in range(7)], 200, 18,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'obstacle50level7', torch.device('cuda:1'), './code/FinalConvModel/Data/obstacle50')

    
    # process 28214
    perform_run('cookie', 32, 7, [10000 for i in range(7)], 200, 18,
                1024, 1024, [8e+5 for i in range(7)], False, False, './code/FinalConvModel/ModelSaves',
                'cookie32varlevel7', torch.device('cuda:2'), './code/FinalConvModel/Data/cookie32variable')

    perform_run('obstacle', 11, 7, [10000 for i in range(7)], 200, 18,
                1024, 1024, [8e+5 for i in range(7)], True, True, './code/FinalConvModel/ModelSaves',
                'obstacle11varlevel7', torch.device('cuda:2'), './code/FinalConvModel/Data/obstacle11variable')

    
    # process 28595
    perform_run('cookie', 128, 7, [10000 for i in range(7)], 200, 18,
                1024, 1024, [8e+5 for i in range(7)], False, False, './code/FinalConvModel/ModelSaves',
                'cookie128varlevel7', torch.device('cuda:3'), './code/FinalConvModel/Data/cookie128variable')
    
    perform_run('obstacle', 51, 7, [10000 for i in range(7)], 200, 18,
                1024, 1024, [8e+5 for i in range(7)], True, True, './code/FinalConvModel/ModelSaves',
                'obstacle51varlevel7', torch.device('cuda:3'), './code/FinalConvModel/Data/obstacle51variable')
    
    '''
    ###TODO OBSTACLE 51 AND OBSTACLE 11 ARE NOT DONE YET
    '''
    # process 23228
    perform_run('uniform', 10, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform10level7', torch.device('cuda:2'), './code/FinalConvModel/Data/uniform10')
    
    # process 23337
    perform_run('log-normal', 100, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100level7', torch.device('cuda:3'), './code/FinalConvModel/Data/lognormal100')
    '''
    '''
    # process 2530
    perform_run('uniform', 100, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform100level7', torch.device('cuda:0'), './code/FinalConvModel/Data/uniform100')
    
    # process 2646
    perform_run('log-normal', 10, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal10level7', torch.device('cuda:1'), './code/FinalConvModel/Data/lognormal10')
    '''

    '''
    # process 104777
    perform_run('uniform', 200, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform200level7', torch.device('cuda:2'), './code/FinalConvModel/Data/uniform200')

    
    # process 10583
    perform_run('log-normal', 200, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal200level7', torch.device('cuda:3'), './code/FinalConvModel/Data/lognormal200')
    '''
    '''
    # process 48752
    perform_run('obstacle', 51, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], True, True, './code/FinalConvModel/ModelSaves',
                'obstacle51level7', torch.device('cuda:0'), './code/FinalConvModel/Data/obstacle51variable')
    
    # process 48895
    perform_run('obstacle', 11, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], True, True, './code/FinalConvModel/ModelSaves',
                'obstacle11level7', torch.device('cuda:1'), './code/FinalConvModel/Data/obstacle11variable')
    '''

    '''
    # process 34718
    perform_run('uniform', 50, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform50level7', torch.device('cuda:2'), './code/FinalConvModel/Data/uniform50')
    

    # process 47310
    perform_run('log-normal', 50, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal50level7', torch.device('cuda:3'), './code/FinalConvModel/Data/lognormal50')

    '''











    '''
    decay = 0.9
    data_set_size = [int(10000 * decay**x) for x in range(7)]
    
    perform_run('log-normal', 100, 7, data_set_size, 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100level7_datadecay{}'.format(decay), torch.device('cuda:2'),
                './code/FinalConvModel/Data/lognormal100')

    
    decay = 0.5
    data_set_size = [int(10000 * decay ** x) for x in range(7)]
    
    perform_run('log-normal', 100, 7, data_set_size, 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100level7_datadecay{}'.format(decay), torch.device('cuda:3'),
                './code/FinalConvModel/Data/lognormal100')
    
    '''

    '''
    
    decay = 0.5
    num_params = decay ** np.arange(7)
    num_params *= 7 * (8e+5) / np.sum(num_params)
    num_params = [int(x) for x in num_params]
    print(decay, num_params)

    # process 34868
    perform_run('log-normal', 100, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, num_params, False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100level7_decay{}'.format(decay), torch.device('cuda:0'),
                './code/FinalConvModel/Data/lognormal100')
                
    

    decay = 0.7
    num_params = decay ** np.arange(7)
    num_params *= 7 * (8e+5) / np.sum(num_params)
    num_params = [int(x) for x in num_params]
    print(decay, num_params)

    # process 35242
    perform_run('log-normal', 100, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, num_params, False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100level7_decay{}'.format(decay), torch.device('cuda:1'),
                './code/FinalConvModel/Data/lognormal100')
    

    decay = 0.9
    num_params = decay ** np.arange(7)
    num_params *= 7 * (8e+5) / np.sum(num_params)
    num_params = [int(x) for x in num_params]
    print(decay, num_params)

    # process 35358
    perform_run('log-normal', 100, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, num_params, False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100level7_decay{}'.format(decay), torch.device('cuda:3'),
                './code/FinalConvModel/Data/lognormal100')
    

    
    decay = 1.2
    num_params = decay ** np.arange(7)
    num_params *= 7 * (8e+5) / np.sum(num_params)
    num_params = [int(x) for x in num_params]
    print(decay, num_params)

    # process 26930
    perform_run('log-normal', 100, 7, [10000 for i in range(7)], 200, 16,
                1024, 1024, num_params, False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100level7_decay{}'.format(decay), torch.device('cuda:0'),
                './code/FinalConvModel/Data/lognormal100')
    
    
    '''
    '''
    # process 19020
    decay = 0.5
    data_set_size = [int(10000 * decay ** x) for x in range(7)]
    perform_run('log-normal', 100, 7, data_set_size, 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100level7_datadecay{}'.format(decay), torch.device('cuda:1'),
                './code/FinalConvModel/Data/lognormal100')
    
    

    # process 18647
    decay = 0.9
    data_set_size = [int(10000 * decay ** x) for x in range(7)]
    perform_run('log-normal', 100, 7, data_set_size, 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100level7_datadecay{}'.format(decay), torch.device('cuda:3'),
                './code/FinalConvModel/Data/lognormal100')
    
    

    # process 5288
    decay = 0.5
    data_set_size = [int(10000 * decay ** x) for x in range(7)]
    perform_run('uniform', 10, 7, data_set_size, 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform10level7_datadecay{}'.format(decay), torch.device('cuda:1'),
                './code/FinalConvModel/Data/uniform10')
    
    # process 18133
    decay = 0.9
    data_set_size = [int(10000 * decay ** x) for x in range(7)]
    perform_run('uniform', 10, 7, data_set_size, 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform10level7_datadecay{}'.format(decay), torch.device('cuda:3'),
                './code/FinalConvModel/Data/uniform10')
    '''

    '''
    # process 35531
    perform_run('log-normal', 100, 7, [232 for i in range(7)], 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100_232samples', torch.device('cuda:0'),
                './code/FinalConvModel/Data/lognormal100')
    
    # process 35651
    perform_run('uniform', 100, 7, [232 for i in range(7)], 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform100_232samples', torch.device('cuda:1'),
                './code/FinalConvModel/Data/uniform100')
    
    # process 35754
    perform_run('uniform', 100, 7, [10000, 5000, 2500, 1250, 625, 312, 156], 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform100_datadecay05', torch.device('cuda:2'),
                './code/FinalConvModel/Data/uniform100')
    
    
    
    # process 31237
    perform_run('log-normal', 10, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal10_extra1', torch.device('cuda:0'), './code/FinalConvModel/Data/lognormal10')

    perform_run('log-normal', 50, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal50_extra1', torch.device('cuda:0'), './code/FinalConvModel/Data/lognormal50')
    
    
    # process 31352
    perform_run('log-normal', 10, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal10_extra2', torch.device('cuda:1'), './code/FinalConvModel/Data/lognormal10')
    
    perform_run('log-normal', 50, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal50_extra2', torch.device('cuda:1'), './code/FinalConvModel/Data/lognormal50')
    
    
    # process 31460
    perform_run('log-normal', 100, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100_extra1', torch.device('cuda:2'), './code/FinalConvModel/Data/lognormal100')
                
    perform_run('log-normal', 100, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100_extra2', torch.device('cuda:2'), './code/FinalConvModel/Data/lognormal100')
    
    
    # process 31556
    perform_run('log-normal', 200, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal200_extra1', torch.device('cuda:3'), './code/FinalConvModel/Data/lognormal200')
                
    perform_run('log-normal', 200, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal200_extra2', torch.device('cuda:3'), './code/FinalConvModel/Data/lognormal200')
    '''
    '''
    # process 7796
    perform_run('cookie', 16, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'cookie16_extra1', torch.device('cuda:0'), './code/FinalConvModel/Data/cookie16fixed')
                
    perform_run('cookie', 16, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'cookie16_extra2', torch.device('cuda:0'), './code/FinalConvModel/Data/cookie16fixed')

    perform_run('uniform', 10, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform10_extra1', torch.device('cuda:0'), './code/FinalConvModel/Data/uniform10')

    perform_run('uniform', 10, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform10_extra2', torch.device('cuda:0'), './code/FinalConvModel/Data/uniform10')
    
    
    # process 7917
    perform_run('cookie', 64, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'cookie64_extra1', torch.device('cuda:1'), './code/FinalConvModel/Data/cookie64fixed')
                
    perform_run('cookie', 64, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'cookie64_extra2', torch.device('cuda:1'), './code/FinalConvModel/Data/cookie64fixed')
                
    perform_run('uniform', 50, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform50_extra1', torch.device('cuda:1'), './code/FinalConvModel/Data/uniform50')
                
    perform_run('uniform', 50, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform50_extra2', torch.device('cuda:1'), './code/FinalConvModel/Data/uniform50')
    
    
    # process 8025
    perform_run('cookie', 32, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, False, './code/FinalConvModel/ModelSaves',
                'cookie32_extra1', torch.device('cuda:2'), './code/FinalConvModel/Data/cookie32variable')
                
    perform_run('cookie', 32, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, False, './code/FinalConvModel/ModelSaves',
                'cookie32_extra2', torch.device('cuda:2'), './code/FinalConvModel/Data/cookie32variable')
                
    perform_run('uniform', 100, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform100_extra1', torch.device('cuda:2'), './code/FinalConvModel/Data/uniform100')
                
    perform_run('uniform', 100, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform100_extra2', torch.device('cuda:2'), './code/FinalConvModel/Data/uniform100')
    
                
    # process 8128
    perform_run('cookie', 128, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, False, './code/FinalConvModel/ModelSaves',
                'cookie128_extra1', torch.device('cuda:3'), './code/FinalConvModel/Data/cookie128variable')
                
    perform_run('cookie', 128, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, False, './code/FinalConvModel/ModelSaves',
                'cookie128_extra2', torch.device('cuda:3'), './code/FinalConvModel/Data/cookie128variable')
                
    perform_run('uniform', 200, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform200_extra1', torch.device('cuda:3'), './code/FinalConvModel/Data/uniform200')
                
    perform_run('uniform', 200, 7, [10000 for i in range(7)], 200, 20,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform200_extra2', torch.device('cuda:3'), './code/FinalConvModel/Data/uniform200')
    '''
    '''
    
    # process 14977
    perform_run('uniform', 100, 7, [10000, 5000, 2500, 1250, 625, 312, 156], 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform100_datadecay05_extra1', torch.device('cuda:0'),
                './code/FinalConvModel/Data/uniform100')

    perform_run('uniform', 100, 7, [10000, 5000, 2500, 1250, 625, 312, 156], 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform100_datadecay05_extra2', torch.device('cuda:0'),
                './code/FinalConvModel/Data/uniform100')

    
    # 32734
    perform_run('log-normal', 100, 7, [10000, 5000, 2500, 1250, 625, 312, 156], 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100_datadecay05_extra1', torch.device('cuda:1'),
                './code/FinalConvModel/Data/lognormal100')

    perform_run('log-normal', 100, 7, [10000, 5000, 2500, 1250, 625, 312, 156], 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'lognormal100_datadecay05_extra2', torch.device('cuda:1'),
                './code/FinalConvModel/Data/lognormal100')
    '''
    '''
    
    # process 11082
    perform_run('uniform', 100, 7, [10000, 5000, 2500, 1250, 625, 312, 156], 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, True, './code/FinalConvModel/ModelSaves',
                'uniform100_datadecay05_extra3', torch.device('cuda:3'),
                './code/FinalConvModel/Data/uniform100')
    '''

    # process 6847
    perform_run('cookie', 32, 7, [10000, 5000, 2500, 1250, 625, 312, 156], 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, False, './code/FinalConvModel/ModelSaves',
                'cookie32_datadecay05_extra1', torch.device('cuda:3'),
                './code/FinalConvModel/Data/cookie32variable')

    perform_run('cookie', 32, 7, [10000, 5000, 2500, 1250, 625, 312, 156], 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, False, './code/FinalConvModel/ModelSaves',
                'cookie32_datadecay05_extra2', torch.device('cuda:3'),
                './code/FinalConvModel/Data/cookie32variable')

    perform_run('cookie', 32, 7, [10000, 5000, 2500, 1250, 625, 312, 156], 200, 16,
                1024, 1024, [8e+5 for i in range(7)], False, False, './code/FinalConvModel/ModelSaves',
                'cookie32_datadecay05_extra3', torch.device('cuda:3'),
                './code/FinalConvModel/Data/cookie32variable')
