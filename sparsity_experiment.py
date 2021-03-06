import os
import numpy as np
from main import main
from layers import ExchangeableLayer, FeatureDropoutLayer, PoolingLayer, BatchNormLayer
from data_util import ToyDataLoader
from util import _uniform_embeddings, np_rmse_loss, update_observed
import tensorflow as tf
import pickle

if __name__ == "__main__":

    data_set = 'toy'
    units_in = 1
    embedding_size_data = 2
    embedding_size_network = 10
    units = 64
    units_out = 1

    # activation = tf.nn.relu
    activation = lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x) # Leaky Relu
    dropout_rate = 0.2
    skip_connections = False

    auto_restore = False
    # save_model = False

    opts = {'epochs':4000,
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
                            'min_observed':5, # generate at least min_observed entries per row and column (sparsity rate may be affected)
                },
                'encoder_opts':{'pool_mode':'mean',
                              'dropout_rate':dropout_rate,
                              'units_in':units_in,
                              'units_out':units_out,
                              'side_info':True,
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
                'decoder_opts': {'pool_mode':'mean',
                                 'dropout_rate':dropout_rate,
                                 'units_in':embedding_size_network,
                                 'units_out':units_out,
                                 'side_info':True,
                                  'layers': [
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
                # 'save_model':save_model,
                'save_frequency':1000000, # Save model every save_frequency epochs
                'loss_save_improvement':.02, # If loss changes by more than loss_save_tolerance (as % of old value), save the model
                'debug':True, # Set random seeds or not
                # 'seed':9858776,
                'seed': 9870112,
                }

    # np.random.seed(9873866)
    # np.random.seed(9999999)
    # np.random.seed(8888888)
    # np.random.seed(7777777)
    # np.random.seed(6666666)
    np.random.seed(33)

    checkpoints_folder = opts['checkpoints_folder']
    os.mkdir(checkpoints_folder + '/sparsity_experiment')

    n_runs = 5
    for k in range(n_runs):

        print('######################## Run ', k, '########################')
        # percent_observed = [1, .02]
        # percent_observed = np.logspace(0, -1.6, num=11, endpoint=True) # log_10(-1.6) corresponds to 2.5% sparsity level, which ensures at least 3 entries per row and column. Include 1. just for constructing observed masks
        percent_observed = [1., .9, .8, .7, .6, .5, .4, .3, .2, .1] # Must be decreasing, include 1. just for constructing observed masks

        embeddings = {}
        embeddings['student'] = _uniform_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][0])
        embeddings['course'] = _uniform_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][1])
        embeddings['prof'] = _uniform_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][2])

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

        loss_ts = np.zeros(len(percent_observed))
        loss_mean = np.zeros(len(percent_observed))

        os.mkdir(checkpoints_folder + '/sparsity_experiment/' + str(k))

        pickle.dump({'embeddings':embeddings, 'observed':observed}, open(checkpoints_folder + '/sparsity_experiment/' + str(k) + '/data.p', 'wb'))

        for i, p in enumerate(percent_observed):
            print("===== Model building loop {:d} ({:4f} fraction) =====".format(i,p))

            opts['auto_restore'] = False
            opts['evaluate_only'] = False
            opts['split_sizes'] = [.8, .2, .0]
            opts['data'] = ToyDataLoader(opts['toy_data']['size'],
                                         p,
                                         opts['split_sizes'],
                                         opts['encoder_opts']['units_in'],
                                         opts['toy_data']['embedding_size'],
                                         opts['toy_data']['min_observed'],
                                         embeddings=embeddings,
                                         observed=observed[i],
                                         predict_unobserved=False)

            mean_tr = np.mean(opts['data'].tables['student_course'].values_tr)
            split_tr = opts['data'].tables['student_course'].split[opts['data'].tables['student_course'].split <= 1]
            loss_mean[i] = np_rmse_loss(opts['data'].tables['student_course'].values_all, mean_tr * np.ones_like(opts['data'].tables['student_course'].values_all), split_tr)  # Loss when predicting training mean

            opts['checkpoints_folder'] = checkpoints_folder + '/sparsity_experiment/' + str(k) + '/' + str(i)
            os.mkdir(opts['checkpoints_folder'])
            opts['save_model'] = True

            main(opts)

        # for i, p in enumerate(percent_observed):
        #     print('===== Prediction loop ', i, '=====')
        #
        #     opts['toy_data']['sparsity'] = p
        #     opts['split_sizes'] = None
        #     opts['checkpoints_folder'] = checkpoints_folder + '/sparsity_experiment/' + str(k) + '/' + str(i)
        #     opts['data'] = ToyDataLoader(opts['toy_data']['size'],
        #                                  opts['toy_data']['sparsity'],
        #                                  opts['split_sizes'],
        #                                  opts['encoder_opts']['units_in'],
        #                                  opts['toy_data']['embedding_size'],
        #                                  opts['toy_data']['min_observed'],
        #                                  embeddings=embeddings,
        #                                  observed=observed[i],
        #                                  predict_unobserved=True)
        #
        #     opts['auto_restore'] = True
        #     restore_point = opts['checkpoints_folder'] + '/best.ckpt'
        #     opts['evaluate_only'] = True
        #
        #     loss_ts[i], _, _, _, _ = main(opts, restore_point)
        #
        # path = os.path.join(checkpoints_folder, 'sparsity_experiment', str(k), 'loss.npz')
        # file = open(path, 'wb')
        # np.savez(file, loss_ts=loss_ts, loss_mean=loss_mean)
        # file.close()









