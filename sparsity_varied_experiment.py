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
    embedding_size_network = 10
    units = 64
    units_out = 1

    # activation = tf.nn.relu
    activation = lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x) # Leaky Relu
    dropout_rate = 0.02
    skip_connections = True

    auto_restore = False
    # save_model = False

    opts = {'epochs':50000,
                'data_folder':'data',
                'data_set':data_set,
                # 'split_sizes':[.8, .2, .0], # train, validation, test split
                'noise_rate':dropout_rate,
                'regularization_rate':.00001,
                'learning_rate':.0001,
                'evaluate_only':False,  # If True, don't train the model, just evaluate it
                'calculate_loss':[True, False, False],  # Which tables do we calculate loss on.
                'toy_data':{'size':[200, 200, 200],
                            # 'sparsity':1.,
                            'embedding_size':embedding_size_data,
                            'min_observed':1, # generate at least 2 entries per row and column (sparsity rate will be affected)
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
                'save_frequency':1000000, # Save model every save_frequency epochs
                'loss_save_improvement':.005, # If loss changes by more than loss_save_tolerance (as % of old value), save the model
                'debug':True, # Set random seeds or not
                # 'seed':9858776,
                'seed': 9870112,
                }

    np.random.seed(1122333)

    percent_observed = [1., .9, .8, .7, .6, .5, .4, .3, .2, .1]  # Must be decreasing

    observed_new = [{}]
    observed_new[0]['sc'] = np.ones((opts['toy_data']['size'][0], opts['toy_data']['size'][1]))
    observed_new[0]['sp'] = np.ones((opts['toy_data']['size'][0], opts['toy_data']['size'][2]))
    observed_new[0]['cp'] = np.ones((opts['toy_data']['size'][1], opts['toy_data']['size'][2]))
    for i in range(1, len(percent_observed)):  # data matrices are percent_observed[i]% observed

        p_prev = percent_observed[i - 1]
        p = percent_observed[i]

        p_keep = p / p_prev

        observed_new.append({})
        observed_new[i]['sc'] = update_observed(observed_new[i - 1]['sc'], p_keep, opts['toy_data']['min_observed'])
        observed_new[i]['sp'] = update_observed(observed_new[i - 1]['sp'], p_keep, opts['toy_data']['min_observed'])
        observed_new[i]['cp'] = update_observed(observed_new[i - 1]['cp'], p_keep, opts['toy_data']['min_observed'])

    percent_observed = percent_observed[1:]  # remove 1.0 from list, since some data must be unobserved to make predictions
    observed_new = observed_new[1:]  # remove corresponding matrix


    np.random.seed(9873866)

    checkpoints_folder = opts['checkpoints_folder']

    n_runs = 5
    loss_ts = np.zeros((n_runs, 9, 9))
    loss_mean = np.zeros((n_runs, 9, 9))

    for k in range(n_runs):

        print('######################## Run ', k, '########################')

        percent_observed = [1., .9, .8, .7, .6, .5, .4, .3, .2, .1]  # Must be decreasing

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

        # observed_new = []

        for i, p in enumerate(percent_observed):

                print('========== Observed loop ', i, '==========')

                observed_new.append({})
                count_sc = observed[i]['sc'].shape[0] * observed[i]['sc'].shape[1]
                count_sp = observed[i]['sp'].shape[0] * observed[i]['sp'].shape[1]
                count_cp = observed[i]['cp'].shape[0] * observed[i]['cp'].shape[1]

                opts['split_sizes'] = [.8, .2, 0]
                opts['checkpoints_folder'] = checkpoints_folder + '/sparsity_varied_experiment/' + str(k) + '/' + str(i)

                for j, q in enumerate(percent_observed):

                    # if j >= i:

                    print('===== Predict loop ', j, '=====')

                    # observed_new[i]['sc'] = update_observed(observed[i]['sc'], q / p, opts['toy_data']['min_observed'])
                    # observed_new[i]['sp'] = update_observed(observed[i]['sp'], q / p, opts['toy_data']['min_observed'])
                    # observed_new[i]['cp'] = update_observed(observed[i]['cp'], q / p, opts['toy_data']['min_observed'])

                    opts['toy_data']['sparsity'] = q

                    opts['data'] = ToyDataLoader(opts['toy_data']['size'],
                                                 opts['toy_data']['sparsity'],
                                                 opts['split_sizes'],
                                                 opts['encoder_opts']['units_in'],
                                                 opts['toy_data']['embedding_size'],
                                                 opts['toy_data']['min_observed'],
                                                 embeddings=embeddings,
                                                 # alpha=alpha,
                                                 alpha=None,
                                                 observed=observed_new[j],
                                                 predict_unobserved=True)
                    opts['auto_restore'] = True
                    restore_point = opts['checkpoints_folder'] + '/best.ckpt'
                    opts['evaluate_only'] = True
                    opts['save_model'] = False

                    loss_ts[k, i, j], loss_mean[k, i, j] = main(opts, restore_point)
        print(loss_ts[k,:,:])
        print(loss_mean[k, :, :])

        path = os.path.join(checkpoints_folder, 'sparsity_varied_experiment', 'run_{:d}_loss_varied.npz'.format(k))
        file = open(path, 'wb')
        np.savez(file, loss_ts=loss_ts[k,:,:], loss_mean=loss_mean[k,:,:])
        file.close()

        file_ts = open(checkpoints_folder + '/sparsity_varied_experiment/run_{:d}_loss_varied_ts.csv'.format(k), 'w')
        np.savetxt(file_ts, loss_ts[k,:,:])
        file_ts.close()

        file_mean = open(checkpoints_folder + '/sparsity_varied_experiment/run_{:d}_loss_varied_mean.csv'.format(k), 'w')
        np.savetxt(file_mean, loss_mean[k, :, :])
        file_mean.close()














