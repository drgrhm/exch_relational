import os
import numpy as np
from main import main
from layers import ExchangeableLayer, FeatureDropoutLayer, PoolingLayer
from data_util import ToyDataLoader
from util import gaussian_embeddings, np_rmse_loss, update_observed
import tensorflow as tf


if __name__ == "__main__":

    data_set = 'toy'
    units_in = 1
    embedding_size_data = 2
    embedding_size_network = 16
    units = 128
    units_out = 1

    # activation = tf.nn.relu
    activation = lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x) # Leaky Relu
    dropout_rate = 0.2
    skip_connections = True

    auto_restore = False
    # save_model = False

    opts = {'epochs':10000,
                'data_folder':'data',
                'data_set':data_set,
                'split_sizes':[.8, .2, .0], # train, validation, test split
                'noise_rate':dropout_rate,
                'regularization_rate':.00001,
                'learning_rate':.0001,
                'evaluate_only':False,  # If True, don't train the model, just evaluate it
                'toy_data':{'size':[200, 100, 50],
                            # 'sparsity':1.,
                            'embedding_size':embedding_size_data,
                            'min_observed':2, # generate at least 2 entries per row and column (sparsity rate will be affected)
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
                                        # {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                        # {'type':FeatureDropoutLayer, 'units_out':units},
                                        # {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                        # {'type':FeatureDropoutLayer, 'units_out':units},
                                        {'type':ExchangeableLayer, 'units_out':embedding_size_network,  'activation':None},
                                        {'type':PoolingLayer, 'units_out':embedding_size_network},
                                       ],
                                },
                'decoder_opts': {'pool_mode':'mean',
                                 'dropout_rate':dropout_rate,
                                 'units_in':embedding_size_network,
                                 'units_out':units_out,
                                  'layers': [
                                      {'type':ExchangeableLayer, 'units_out':units, 'activation':activation},
                                      {'type':FeatureDropoutLayer, 'units_out':units},
                                      {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                      {'type':FeatureDropoutLayer, 'units_out':units},
                                      {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                      {'type':FeatureDropoutLayer, 'units_out':units},
                                      {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                      {'type':FeatureDropoutLayer, 'units_out':units},
                                      {'type':ExchangeableLayer, 'units_out':units, 'activation': activation, 'skip_connections':skip_connections},
                                      {'type':FeatureDropoutLayer, 'units_out': units},
                                      {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                      {'type':FeatureDropoutLayer, 'units_out':units},
                                      # {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                      # {'type':FeatureDropoutLayer, 'units_out':units},
                                      # {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                      # {'type':FeatureDropoutLayer, 'units_out':units},
                                      {'type':ExchangeableLayer, 'units_out':units_out, 'activation':None},
                              ],
                             },
                'verbosity':2,
                'checkpoints_folder':'checkpoints',
                'image_save_folder':'img',
                'restore_point_epoch':-1,
                # 'save_model':save_model,
                'save_frequency':100, # Save model every save_frequency epochs
                'loss_save_tolerance':.0, # If loss changes by more than loss_save_tolerance (as % of old value), save the model
                'debug':True, # Set random seeds or not
                # 'seed':9858776,
                'seed': 9870112,
                }

    np.random.seed(9858776)

    percent_observed = [1., .5, .4, .3, .2, .1] # Must be decreasing
    percent_training = [.9, .8, .7, .6, .5]

    # percent_observed = [1., .9, .85, .8, .75, .7, .65, .6, .55, .5]  # Must be decreasing
    # percent_training = [.9, .8, .7]

    embeddings = {}
    embeddings['student'] = gaussian_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][0])
    embeddings['course'] = gaussian_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][1])
    embeddings['prof'] = gaussian_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][2])

    num_alpha = max(4, opts['toy_data']['embedding_size'])
    alpha = {'sc':2 * np.random.randn(num_alpha), 'sp':2 * np.random.randn(num_alpha), 'cp':2 * np.random.randn(num_alpha)}

    observed = [{}]
    observed[0]['sc'] = np.ones((opts['toy_data']['size'][0], opts['toy_data']['size'][1]))
    observed[0]['sp'] = np.ones((opts['toy_data']['size'][0], opts['toy_data']['size'][2]))
    observed[0]['cp'] = np.ones((opts['toy_data']['size'][1], opts['toy_data']['size'][2]))
    for i in range(1, len(percent_observed)): # data matrices are percent_observed[i]% observed

        p_prev = percent_observed[i-1]
        p = percent_observed[i]

        p_keep = p / p_prev

        observed.append({})
        observed[i]['sc'] = update_observed(observed[i-1]['sc'], p_keep, opts['toy_data']['min_observed'])
        observed[i]['sp'] = update_observed(observed[i-1]['sp'], p_keep, opts['toy_data']['min_observed'])
        observed[i]['cp'] = update_observed(observed[i-1]['cp'], p_keep, opts['toy_data']['min_observed'])

    percent_observed = percent_observed[1:] # remove 1.0 from list, since some data must be unobserved to make predictions
    observed = observed[1:] # remove corresponding matrix

    checkpoints_folder = opts['checkpoints_folder']
    os.mkdir(checkpoints_folder + '/sparsity_experiment')


    loss_ts = np.zeros((len(percent_observed), len(percent_training)))
    loss_mean = np.zeros((len(percent_observed), len(percent_training)))

    for i, p in enumerate(percent_observed):
        for j, q in enumerate(percent_training):

            print('===== Model building loop ', i, j, '=====')

            opts['toy_data']['sparsity'] = p
            opts['data'] = ToyDataLoader(opts['toy_data']['size'],
                                         opts['toy_data']['sparsity'],
                                         opts['split_sizes'],
                                         opts['encoder_opts']['units_in'],
                                         opts['toy_data']['embedding_size'],
                                         opts['toy_data']['min_observed'],
                                         embeddings=embeddings,
                                         alpha=alpha,
                                         observed=observed[i])

            mean_tr = np.mean(opts['data'].tables['student_course'].values_tr)
            split_tr = opts['data'].tables['student_course'].split[opts['data'].tables['student_course'].split <= 1]
            loss_mean[i, j] = np_rmse_loss(opts['data'].tables['student_course'].values_all, mean_tr * np.ones_like(opts['data'].tables['student_course'].values_all), split_tr)  # Loss when predicting training mean

            opts['checkpoints_folder'] = checkpoints_folder + '/sparsity_experiment/' + str(i) + '-' + str(j)
            os.mkdir(opts['checkpoints_folder'])
            opts['save_model'] = True

            main(opts)

    for i, p in enumerate(percent_observed):
        for j, q in enumerate(percent_training):

            print('===== Prediction loop ', i, j, '=====')

            unobserved = {'sc': 1 - observed[i]['sc'], 'sp': 1 - observed[i]['sp'], 'cp': 1 - observed[i]['cp']}
            opts['split_sizes'] = [q, 1. - q, 0]

            opts['checkpoints_folder'] = checkpoints_folder + '/sparsity_experiment/' + str(i) + '-' + str(j)
            opts['data'] = ToyDataLoader(opts['toy_data']['size'],
                                         opts['toy_data']['sparsity'],
                                         opts['split_sizes'],
                                         opts['encoder_opts']['units_in'],
                                         opts['toy_data']['embedding_size'],
                                         opts['toy_data']['min_observed'],
                                         embeddings=embeddings,
                                         alpha=alpha,
                                         observed=unobserved)

            opts['auto_restore'] = True
            restore_point = opts['checkpoints_folder'] + '/best.ckpt'
            opts['evaluate_only'] = True

            loss_ts[i,j] = main(opts, restore_point)

            path = os.path.join(opts['checkpoints_folder'], 'loss.npz')
            file = open(path, 'wb')
            np.savez(file, loss_ts=loss_ts, loss_mean=loss_mean)
            file.close()

    path = os.path.join(checkpoints_folder, 'sparsity_experiment', 'loss.npz')
    file = open(path, 'wb')
    np.savez(file, loss_ts=loss_ts, loss_mean=loss_mean)
    file.close()

    print("Test loss:\n", loss_ts)
    print("Predict mean loss:\n", loss_mean)









