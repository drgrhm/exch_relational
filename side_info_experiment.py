import os
import numpy as np
from main import main
from layers import ExchangeableLayer, FeatureDropoutLayer, PoolingLayer, BatchNormLayer
from data_util import ToyDataLoader
from util import gaussian_embeddings, np_rmse_loss, update_observed, choose_observed
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
    dropout_rate = 0.2
    skip_connections = False

    auto_restore = False
    # save_model = False

    opts = {'epochs':10000,
            'data_folder':'data',
            'data_set':data_set,
            'split_sizes':[.8, .2, 0], # train, validation, test split
            'noise_rate':dropout_rate,
            'regularization_rate':.00001,
            'learning_rate':.0001,
            'evaluate_only':False,  # If True, don't train the model, just evaluate it
            'calculate_loss':[True, False, False],  # Which tables do we calculate loss on.
            'toy_data':{'size':[200, 200, 200],
                        'sparsity':.1,
                        'embedding_size':embedding_size_data,
                        'min_observed':3, # generate at least min_observed entries per row and column (sparsity rate may be affected)
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
            'loss_save_improvement':.02, # If loss changes by more than loss_save_improvement since last save (as fraction of old value), save the model
            'debug':True, # Set random seeds or not
            'seed':9858776,
            # 'seed': 9870112,
            }

    np.random.seed(9858776)


    checkpoints_folder = opts['checkpoints_folder']
    os.mkdir(checkpoints_folder + '/side_info_experiment')

    n_runs = 10
    for k in range(n_runs):

        print('######################## Run ', k, '########################')

        os.mkdir(checkpoints_folder + '/side_info_experiment/' + str(k))

        embeddings = {}
        embeddings['student'] = gaussian_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][0])
        embeddings['course'] = gaussian_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][1])
        embeddings['prof'] = gaussian_embeddings(opts['toy_data']['embedding_size'], opts['toy_data']['size'][2])

        # percent_observed = np.logspace(0, -1.6, num=11, endpoint=True)  # Must be decreasing. log_10(-1.6) corresponds to 2.5% sparsity level, which ensures at least 3 entries per row and column. Include 1. just for constructing observed masks
        percent_observed = [1., .9, .8, .7, .6, .5, .4, .3, .2, .1, .0]  # Must be decreasing
        # percent_observed = [1., .5, .3, .1, .0]  # Must be decreasing

        observed_sc = choose_observed(0, opts['toy_data']['size'][0:2], opts['toy_data']['sparsity'], min_observed=opts['toy_data']['min_observed'])

        observed = [{}]
        observed[0]['sc'] = observed_sc
        observed[0]['sp'] = np.ones((opts['toy_data']['size'][0], opts['toy_data']['size'][2]))
        observed[0]['cp'] = np.ones((opts['toy_data']['size'][1], opts['toy_data']['size'][2]))

        for i in range(1, len(percent_observed)):  # data matrices are percent_observed[i]% observed

            p_prev = percent_observed[i - 1]
            p = percent_observed[i]

            p_keep = p / p_prev

            observed.append({})
            observed[i]['sc'] = observed_sc
            observed[i]['sp'] = update_observed(observed[i - 1]['sp'], p_keep, min_observed=0)
            observed[i]['cp'] = update_observed(observed[i - 1]['cp'], p_keep, min_observed=0)

        percent_observed = percent_observed[1:]  # remove 1.0 from list, since some data must be unobserved to make predictions
        observed = observed[1:]  # remove corresponding matrix

        losses_ts = []
        losses_mean = []

        for i in range(len(percent_observed)):

            print("===== Model building loop {:d} =====".format(i))

            opts['data'] = ToyDataLoader(opts['toy_data']['size'],
                                         opts['toy_data']['sparsity'],
                                         opts['split_sizes'],
                                         opts['encoder_opts']['units_in'],
                                         opts['toy_data']['embedding_size'],
                                         opts['toy_data']['min_observed'],
                                         embeddings=embeddings,
                                         observed=observed[i],
                                         predict_unobserved=False)

            opts['checkpoints_folder'] = checkpoints_folder + '/side_info_experiment/' + str(k) + '/'+ str(i)
            os.mkdir(opts['checkpoints_folder'])
            opts['save_model'] = True

            if percent_observed[i] == 0.:
                opts['encoder_opts']['side_info'] = False
                opts['decoder_opts']['side_info'] = False

            loss_ts, loss_mean, _, _, _ = main(opts)
            # loss_ts, loss_mean = 0, 0


        #     losses_ts.append(loss_ts)
        #     losses_mean.append(loss_mean)
        #
        # path = os.path.join(checkpoints_folder, 'side_info_experiment', str(k), 'loss.npz')
        # file = open(path, 'wb')
        # np.savez(file, losses_ts=losses_ts, losses_mean=losses_mean)
        # file.close()



            for j, q in enumerate(percent_observed):  # prediction sparsity level
                print("===== Prediction loop {:d} ({:4f} fraction) =====".format(j, q))

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
            #     opts['auto_restore'] = True
            #     restore_point = opts['checkpoints_folder'] + '/best.ckpt'
            #     opts['evaluate_only'] = True
            #     opts['save_model'] = False
            #
            #     if percent_observed[i] == 0.:
            #         opts['encoder_opts']['side_info'] = False
            #         opts['decoder_opts']['side_info'] = False
            #
            #     loss_ts[k, i, j], loss_mean[k, i, j], _, _, _ = main(opts, restore_point)







        # print("Test loss:\n", losses_ts)
        # print("Predict mean loss:\n", losses_mean)