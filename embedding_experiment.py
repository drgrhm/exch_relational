import os
import numpy as np
from main import main
import tensorflow as tf
from layers import ExchangeableLayer, FeatureDropoutLayer, PoolingLayer

if __name__ == "__main__":

    data_set = 'toy'
    units_in = 1
    embedding_size_data = 2
    embedding_size_network = 2
    units = 128
    units_out = 1

    # activation = tf.nn.relu
    activation = lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x) # Leaky Relu
    dropout_rate = 0.2
    skip_connections = True

    auto_restore = False
    save_model = True

    opts = {'epochs':2000,
            'data_folder':'data',
            'data_set':data_set,
            'split_sizes':[.8, .1, .1],  # train, validation, test split
            'noise_rate':dropout_rate,
            'regularization_rate':.00001,
            'learning_rate':.0001,
            'evaluate_only':False,  # If True, don't train the model, just evaluate it
            'toy_data':{'size':[1500, 1000, 500],
                        'sparsity':.01,
                        'embedding_size':embedding_size_data,
                        'min_observed':2,  # generate at least 2 entries per row and column (sparsity rate will be affected)
                        },
            'encoder_opts':{'pool_mode':'mean',
                            'dropout_rate':dropout_rate,
                            'units_in':units_in,
                            'units_out':units_out,
                            'layers':[
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':embedding_size_network, 'activation':None},
                                {'type':PoolingLayer, 'units_out':embedding_size_network},
                            ],
                            },
            'decoder_opts':{'pool_mode':'mean',
                            'dropout_rate':dropout_rate,
                            'units_in':embedding_size_network,
                            'units_out':units_out,
                            'layers':[
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units_out, 'activation':None},
                            ],
                            },
            'verbosity':2,
            'checkpoints_folder':'checkpoints',
            'image_save_folder':'img',
            'restore_point_epoch':-1,
            'save_model':save_model,
            'save_frequency':100,  # Save model every save_frequency epochs
            'loss_save_tolerance':.0,  # If loss changes by more than loss_save_tolerance (as % of old value), save the model
            # 'debug':True,  # Set random seeds or not
            # 'seed':9858776,
            # 'seed': 9870112,
            }

    np.random.seed(9870112)
    seeds = np.random.randint(low=0, high=1000000, size=3)

    for seed in seeds:

        path_img = 'img/' + str(seed)
        path_cpt = 'checkpoints/' + str(seed)
        # os.mkdir(path_img)
        os.mkdir(path_cpt)

        opts['debug'] = True
        opts['seed'] = seed
        # opts['image_save_folder'] = path_img
        opts['checkpoints_folder'] = path_cpt

        main(opts)

        # print(seed)