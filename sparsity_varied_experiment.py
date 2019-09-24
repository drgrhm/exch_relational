import os
import numpy as np
from main import main
from layers import ExchangeableLayer, FeatureDropoutLayer, PoolingLayer, BatchNormLayer
from data_util import ToyDataLoader
from util import _uniform_embeddings, np_rmse_loss, update_observed
import tensorflow as tf
import pickle


if __name__ == "__main__":

    np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)

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
                            'min_observed':5, # generate at least min_observed entries per row and column (sparsity rate will be affected)
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
                'decoder_opts': {'pool_mode':'mean',
                                 'dropout_rate':dropout_rate,
                                 'units_in':embedding_size_network,
                                 'units_out':units_out,
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
                'loss_save_improvement':.005, # If loss changes by more than loss_save_tolerance (as % of old value), save the model
                'debug':True, # Set random seeds or not
                # 'seed':9858776,
                'seed': 9870112,
                }

    checkpoints_folder = opts['checkpoints_folder']

    n_runs = 5
    n_models = 9
    loss_ts = np.zeros((n_runs, n_models, n_models))
    loss_mean = np.zeros((n_runs, n_models, n_models))

    #########
    inductive = True
    if inductive:
        np.random.seed(1111)

    k_base = 0

    for k in range(n_runs): # repeat experiment n_runs times and average
        print('######################## Run ', k, '########################')
        percent_observed = [.9, .8, .7, .6, .5, .4, .3, .2, .1]  # Must be decreasing

        data = pickle.load(open('checkpoints/sparsity_experiment/{}/data.p'.format(k), 'rb'))
        observed = data['observed']

        if inductive:
            # generage new embeddings

            embeddings = {}
            embeddings['student'] = _uniform_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][0])
            embeddings['course'] = _uniform_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][1])
            embeddings['prof'] = _uniform_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][2])

            #####
            observed = [{}]
            observed[0]['sc'] = np.ones((opts['toy_data']['size'][0], opts['toy_data']['size'][1]))
            observed[0]['sp'] = np.ones((opts['toy_data']['size'][0], opts['toy_data']['size'][2]))
            observed[0]['cp'] = np.ones((opts['toy_data']['size'][1], opts['toy_data']['size'][2]))
            percent_observed = [1., .9, .8, .7, .6, .5, .4, .3, .2, .1]

            for i in range(1, len(percent_observed)):  # data matrices are percent_observed[i]% observed

                p_prev = percent_observed[i - 1]
                p = percent_observed[i]

                p_keep = p / p_prev

                observed.append({})
                observed[i]['sc'] = update_observed(observed[i - 1]['sc'], p_keep, opts['toy_data']['min_observed'])
                observed[i]['sp'] = update_observed(observed[i - 1]['sp'], p_keep, opts['toy_data']['min_observed'])
                observed[i]['cp'] = update_observed(observed[i - 1]['cp'], p_keep, opts['toy_data']['min_observed'])

            percent_observed = percent_observed[1:]  # remove 1.0 from list, since some data must be unobserved to make predictions
            observed = observed[1:]  # remove corresponding matrix
            #######

        else:
            # data = pickle.load(open('checkpoints/sparsity_experiment/{}/data.p'.format(k), 'rb'))
            embeddings = data['embeddings']
            # observed = data['observed']

        # Which of the non-test data (i.e. entries not being predicted) to use for making predictions
        observed_new = [{}]
        observed_new[0]['sc'] = np.copy(observed[0]['sc'])  # start from the 90% observed data
        observed_new[0]['sp'] = np.copy(observed[0]['sp'])
        observed_new[0]['cp'] = np.copy(observed[0]['cp'])

        # Data used for making predictions
        for i in range(1, len(percent_observed)):  # data matrices are percent_observed[i]% observed

            p_prev = percent_observed[i - 1]
            po = percent_observed[i]

            p_keep = po / p_prev

            observed_new.append({})
            observed_new[i]['sc'] = update_observed(observed_new[i - 1]['sc'], p_keep, opts['toy_data']['min_observed'])
            observed_new[i]['sp'] = update_observed(observed_new[i - 1]['sp'], p_keep, opts['toy_data']['min_observed'])
            observed_new[i]['cp'] = update_observed(observed_new[i - 1]['cp'], p_keep, opts['toy_data']['min_observed'])

        predict = {} # Predict entries that were never used for training in any model
        predict['sc'] = np.ones_like(observed[0]['sc']) - observed[0]['sc']
        predict['sp'] = np.ones_like(observed[0]['sp']) - observed[0]['sp']
        predict['cp'] = np.ones_like(observed[0]['cp']) - observed[0]['cp']

        for i, p in enumerate(percent_observed): # training sparsity level
            print("======= Model building loop {:d} ({:4f} fraction) =======".format(i, p))

            opts['split_sizes'] = [.8, .2, 0]
            opts['checkpoints_folder'] = checkpoints_folder + '/sparsity_varied_experiment/' + str(k + k_base) + '/' + str(i)

            for j, q in enumerate(percent_observed): # prediction sparsity level
                print("===== Prediction loop {:d} ({:4f} fraction) =====".format(j, q))

                opts['toy_data']['sparsity'] = q

                opts['data'] = ToyDataLoader(opts['toy_data']['size'],
                                             opts['toy_data']['sparsity'],
                                             opts['split_sizes'],
                                             opts['encoder_opts']['units_in'],
                                             opts['toy_data']['embedding_size'],
                                             opts['toy_data']['min_observed'],
                                             embeddings=embeddings,
                                             observed=observed_new[j],
                                             predict_unobserved=False,
                                             predict=predict
                                             )
                opts['auto_restore'] = True
                restore_point = opts['checkpoints_folder'] + '/best.ckpt'
                opts['evaluate_only'] = True
                opts['save_model'] = False

                loss_ts[k, i, j], loss_mean[k, i, j], _, _, _ = main(opts, restore_point)

        print(loss_ts[k,:,:])
        print(loss_mean[k, :, :])

        if inductive:
            file_name = 'run_{:d}_loss_varied_inductive.npz'.format(k + k_base)
        else:
            file_name = 'run_{:d}_loss_varied.npz'.format(k + k_base)

        path = os.path.join(checkpoints_folder, 'sparsity_varied_experiment', file_name)
        file = open(path, 'wb')
        np.savez(file, loss_ts=loss_ts[k,:,:], loss_mean=loss_mean[k,:,:])
        file.close()

        # file_ts = open(checkpoints_folder + '/sparsity_varied_experiment/run_{:d}_loss_varied_ts.csv'.format(k), 'w')
        # np.savetxt(file_ts, loss_ts[k,:,:])
        # file_ts.close()
        #
        # file_mean = open(checkpoints_folder + '/sparsity_varied_experiment/run_{:d}_loss_varied_mean.csv'.format(k), 'w')
        # np.savetxt(file_mean, loss_mean[k, :, :])
        # file_mean.close()














