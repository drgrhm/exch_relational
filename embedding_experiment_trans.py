import os
import numpy as np
from main import main
import tensorflow as tf
from util import plot_embeddings, gaussian_embeddings
from data_util import ToyDataLoader
from layers import ExchangeableLayer, FeatureDropoutLayer, PoolingLayer, BatchNormLayer

if __name__ == "__main__":

    data_set = 'toy'
    units_in = 1
    units = 64
    units_out = 1

    embedding_size_data = 2
    embedding_size_network = 2

    # activation = tf.nn.relu
    activation = lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x) # Leaky Relu
    dropout_rate = 0.2
    skip_connections = False

    auto_restore = False
    save_model = True

    opts = {'epochs':5000,
            'data_folder':'data',
            'data_set':data_set,
            'split_sizes':[.8, .2, 0],  # train, validation, test split
            'noise_rate':dropout_rate,
            'regularization_rate':.00001,
            'learning_rate':.0001,
            'evaluate_only':False,  # If True, don't train the model, just evaluate it
            'calculate_loss':[True, False, False],  # Which tables do we calculate loss on.
            'toy_data':{'size':[200, 200, 200],
                        'sparsity':.1,
                        'embedding_size':embedding_size_data,
                        'min_observed':5,  # generate at least min_observed entries per row and column (sparsity rate may be affected)
                        },
            'encoder_opts':{'pool_mode':'mean',
                            'dropout_rate':dropout_rate,
                            'units_in':units_in,
                            'units_out':units_out,
                            'layers':[
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'save_embeddings':True},
                                {'type':BatchNormLayer, 'units_out':units},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':BatchNormLayer, 'units_out':units},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':BatchNormLayer, 'units_out':units},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':BatchNormLayer, 'units_out':units},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':BatchNormLayer, 'units_out':units},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':BatchNormLayer, 'units_out':units},
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
                                {'type':BatchNormLayer, 'units_out':units},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':BatchNormLayer, 'units_out':units},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':BatchNormLayer, 'units_out':units},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':BatchNormLayer, 'units_out':units},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':BatchNormLayer, 'units_out':units},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                {'type':BatchNormLayer, 'units_out':units},
                                {'type':FeatureDropoutLayer, 'units_out':units},
                                {'type':ExchangeableLayer, 'units_out':units_out, 'activation':None},
                            ],
                            },
            'verbosity':2,
            'checkpoints_folder':'checkpoints',
            'image_save_folder':'img',
            'restore_point_epoch':-1,
            'save_model':save_model,
            'save_frequency':1000,  # Save model every save_frequency epochs
            'loss_save_improvement':.005,  # If loss changes by more than loss_save_tolerance (as % of old value), save the model
            # 'debug':True,  # Set random seeds or not
            # 'seed':9858776,
            # 'seed': 9870112,
            }

    np.random.seed(9858776)
    seeds = np.random.randint(low=0, high=1000000, size=1)

    for seed in seeds:

        print('===== Seed ', seed, '=====')

        path_cpt = 'checkpoints/embedding_experiment_trans/' + str(seed)

        image_path = 'img/embedding_experiment_trans/'

        opts['debug'] = True
        opts['seed'] = 1223334444
        opts['checkpoints_folder'] = path_cpt
        opts['auto_restore'] = True
        restore_point = opts['checkpoints_folder'] + '/best.ckpt'
        opts['evaluate_only'] = True
        opts['save_model'] = False

        np.random.seed(opts['seed'])

        embeddings = {}
        embeddings['student'] = gaussian_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][0])
        embeddings['course'] = gaussian_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][1])
        embeddings['prof'] = gaussian_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][2])


        opts['data'] = ToyDataLoader(opts['toy_data']['size'],
                                                 opts['toy_data']['sparsity'],
                                                 opts['split_sizes'],
                                                 opts['encoder_opts']['units_in'],
                                                 opts['toy_data']['embedding_size'],
                                                 opts['toy_data']['min_observed'],
                                                 embeddings=embeddings)

        _, _, student_embeds, course_embeds, prof_embeds = main(opts, restore_point)


        # embeds_file = open(path_cpt + '/embeddings_vl_best.npz', 'rb')
        # embeds_data = np.load(embeds_file)
        #
        # embeds_s = embeds_data['student_embeds_in']
        # embeds_c = embeds_data['course_embeds_in']
        # embeds_p = embeds_data['prof_embeds_in']

        os.mkdir(image_path + str(seed))

        plot_embeddings(embeddings['student'], np.squeeze(student_embeds), image_path + str(seed) + '/student_embeddings_inductive.pdf')
        plot_embeddings(embeddings['course'], np.squeeze(course_embeds), image_path + str(seed) + '/course_embeddings_inductive.pdf')
        plot_embeddings(embeddings['prof'], np.squeeze(prof_embeds), image_path + str(seed) + '/prof_embeddings_inductive.pdf')

