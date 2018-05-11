import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from data_util import DataLoader, Table
from model import Model
from layers import ExchangeableLayer, FeatureDropoutLayer


## Noise mask has 0's corresponding to values to be predicted
def table_rmse_loss(values, values_rec, noise_mask):
    non_noise = tf.cast(1 - noise_mask, tf.float32)
    return tf.sqrt( tf.reduce_sum(((values - values_rec)**2)*non_noise) / (tf.reduce_sum(non_noise) + 1e-10) )


def table_prediction_accuracy(indices, values, values_rec, shape, split):
    
    in_t = sparse_transpose(indices, values, shape, split)
    out_t = sparse_transpose(indices, values_rec, shape, split)
    vals_in_t = in_t['values']
    vals_out_t = out_t['values']
    noise = 1 - in_t['split']            

    preds_in = np.sign(np.reshape(vals_in_t[noise == 0], [-1,2])[:,0])
    num_vals = preds_in.shape[0]

    vals_out = np.reshape(vals_out_t[noise == 0], [-1,2])                    
    preds_out = (np.abs(vals_out[:,0]) + np.abs(vals_out[:,1]) ) / 2
    preds_out = np.sign(np.round(np.sign(vals_out[:,0]) * preds_out, decimals=1)) ## TODO better way 


    # print('preds_in:  ', preds_in[0:20].astype(np.float32))
    # print('preds_out: ', preds_out[0:20])


    return np.sum(preds_in == preds_out) / num_vals



def make_uniform_noise_mask(noise_rate, num_vals):
    """A 0/1 noise mask. 0's correspond to dropped out values."""
    n0 = int(noise_rate * num_vals)
    n1 = num_vals - n0
    noise_mask = np.concatenate((np.zeros(n0), np.ones(n1)))
    np.random.shuffle(noise_mask)
    return noise_mask

def make_by_col_noise_mask(noise_rate, num_vals, shape, column_indices):
    """A 0/1 noise mask. 0's correspond to dropped out values. Drop out columns."""
    num_cols = shape[1]
    n0 = int(noise_rate * num_cols)
    n1 = num_cols - n0
    column_mask = np.concatenate((np.zeros(n0), np.ones(n1)))
    np.random.shuffle(column_mask)
    noise_mask = np.take(column_mask, column_indices)
    return noise_mask    

# def make_noisy_values(values, noise_rate, noise_value):
#     """replace noise_rate fraction of values with noise_value."""
#     num_vals = values.shape[0]
#     noise = make_noise_mask(noise_rate, num_vals) 
#     values_noisy = np.copy(values)
#     values_noisy[noise == 0] = noise_value
#     return noise, values_noisy



# For debugging
def sparse_array_to_dense(indices, values, shape, num_features):
    out = np.zeros(list(shape) + [num_features])
    inds = expand_array_indices(indices, num_features)
    inds = list(zip(*inds))
    out[inds] = values
    return out

def expand_array_indices(indices, num_features):    
    num_vals = indices.shape[0]
    inds_exp = np.reshape(np.tile(range(num_features), reps=[num_vals]), newshape=[-1, 1]) # expand dimension of mask indices
    inds = np.tile(indices, reps=[num_features,1]) # duplicate indices num_features times
    inds = np.reshape(inds, newshape=[num_features, num_vals, 2])
    inds = np.reshape(np.transpose(inds, axes=[1,0,2]), newshape=[-1,2])
    inds = np.concatenate((inds, inds_exp), axis=1)
    return inds

def sparse_transpose(indices, values, shape, split):
    trans = np.concatenate((indices, values[:,None], split[:,None]), axis=1)
    trans[:,[0,1,2,3]] = trans[:,[1,0,2,3]]
    trans = list(trans)
    trans.sort(key=lambda row: row[0])
    trans = np.array(trans)    
    inds = trans[:,0:2]
    vals = trans[:,2]
    split = trans[:,3]
    shape[[0,1]] = shape[[1,0]]
    return {'indices':inds, 'values':vals, 'shape':shape, 'split':split}




def main(opts):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    tr, vl, ts = opts['data_split']

    if opts['debug']:
        np.random.seed(12345)

    data = DataLoader('data', tr, vl, ts)

    # print(np.squeeze(sparse_array_to_dense(data.team_match.indices_all, data.team_match.values_all, data.team_match.shape, 1)))
    # print("")
    # print(np.squeeze(sparse_array_to_dense(data.team_match.indices_tr, data.team_match.values_tr, data.team_match.shape, 1)))
    # print("")
    # print(np.squeeze(sparse_array_to_dense(data.team_match.indices_vl, data.team_match.values_vl, data.team_match.shape, 1)))
    # print("")
    # print(np.squeeze(sparse_array_to_dense(data.team_match.indices_tr_vl, data.team_match.values_tr_vl, data.team_match.shape, 1)))


    with tf.Graph().as_default():

        if opts['debug']:
            tf.set_random_seed(12345)

        model = Model(**opts['layer_opts'])

        team_player = {}
        team_player['indices'] = tf.placeholder(tf.int32, shape=(None, 2), name='team_player_indices')
        team_player['values'] = tf.placeholder(tf.float32, shape=(None), name='team_player_values_noisy')
        team_player['shape'] = data.team_player.shape
        team_player_noise_mask = tf.placeholder(tf.float32, shape=(None), name='team_player_noise_mask')
        team_player_values = tf.placeholder(tf.float32, shape=(None), name='team_player_values')

        team_match = {}
        team_match['indices'] = tf.placeholder(tf.int32, shape=(None, 2), name='team_match_indices')        
        team_match['values'] = tf.placeholder(tf.float32, shape=(None), name='team_match_values_noisy')
        team_match['shape'] = data.team_match.shape
        team_match_noise_mask = tf.placeholder(tf.float32, shape=(None), name='team_match_noise_mask')
        team_match_values = tf.placeholder(tf.float32, shape=(None), name='team_match_values')


        team_player_out_tr, team_match_out_tr = model.get_output(team_player, team_match)
        team_player_out_vl, team_match_out_vl = model.get_output(team_player, team_match, reuse=True, is_training=False)

        rec_loss_tr = 0
        # rec_loss_tr += table_rmse_loss(team_player_values, team_player_out_tr['values'], team_player_noise_mask)
        rec_loss_tr += table_rmse_loss(team_match_values, team_match_out_tr['values'], team_match_noise_mask)
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss_tr = rec_loss_tr + opts['regularization_rate']*reg_loss

        rec_loss_vl = 0
        # rec_loss_vl += table_rmse_loss(team_player_values, team_player_out_vl['values'], team_player_noise_mask)
        rec_loss_vl += table_rmse_loss(team_match_values, team_match_out_vl['values'], team_match_noise_mask)

        train_step = tf.train.AdamOptimizer(opts['learning_rate']).minimize(total_loss_tr)
        # train_step = tf.train.GradientDescentOptimizer(opts['learning_rate']).minimize(total_loss_tr)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())


        if opts['layer_opts']['pool_mode'] == 'mean':
            noise_value = 0
        if opts['layer_opts']['pool_mode'] == 'max':
            noise_value = -1e10

        losses_tr = []
        losses_vl = []
        losses_vl_mean = []
        accuracies_tr = []
        accuracies_vl = []

        accuracy_vl_best = 0
        accuracy_vl_best_ep = 0


        for ep in range(opts['epochs']):
            if opts['verbosity'] > 0:
                print('------- epoch:', ep, '-------')

            # team_player_noise = make_noise_mask(opts['noise_rate'], data.team_player.num_values_tr) 
            # team_player_values_noisy = np.copy(data.team_player.values_tr)
            # team_player_values_noisy[team_player_noise == 0] = noise_value

            team_player_noise = np.ones_like(data.team_player.values_tr)
            team_player_values_noisy = np.copy(data.team_player.values_tr)


            # team_match_noise = make_uniform_noise_mask(opts['noise_rate'], data.team_match.num_values_tr) 
            team_match_noise = make_by_col_noise_mask(opts['noise_rate'], data.team_match.num_values_tr, data.team_match.shape, data.team_match.indices_tr[:,1]) 
            team_match_values_noisy = np.copy(data.team_match.values_tr)
            team_match_values_noisy[team_match_noise == 0] = noise_value

            tr_dict = {team_player['indices']:data.team_player.indices_tr, 
                       team_player['values']:team_player_values_noisy, # noisy values
                       team_player_noise_mask:team_player_noise,
                       team_player_values:data.team_player.values_tr, # clean values

                       team_match['indices']:data.team_match.indices_tr, 
                       team_match['values']:team_match_values_noisy, # noisy values
                       team_match_noise_mask:team_match_noise,
                       team_match_values:data.team_match.values_tr # clean values
                      }


            _, loss_tr, team_match_vals_out_tr, = sess.run([train_step, total_loss_tr, team_match_out_tr['values']], feed_dict=tr_dict)
            losses_tr.append(loss_tr)
              
            pred_accuracy_tr = 0
            pred_accuracy_tr = table_prediction_accuracy(data.team_match.indices_tr, data.team_match.values_tr, team_match_vals_out_tr, data.team_match.shape, team_match_noise)
            accuracies_tr.append(pred_accuracy_tr)



            # team_player_noise = 1 - data.team_player.split[data.team_player.split <= 1]
            # team_player_values_noisy = data.team_player.values_tr_vl
            # team_player_values_noisy[team_player_noise == 0] = noise_value

            team_player_noise = np.ones_like(data.team_player.values_tr_vl)
            team_player_values_noisy = np.copy(data.team_player.values_tr_vl)

            team_match_noise = 1 - data.team_match.split[data.team_match.split <= 1]
            team_match_values_noisy = np.copy(data.team_match.values_tr_vl)
            team_match_values_noisy[team_match_noise == 0] = noise_value

            vl_dict = {team_player['indices']:data.team_player.indices_tr_vl, 
                       team_player['values']:team_player_values_noisy, # noisy values
                       team_player_noise_mask:team_player_noise,
                       team_player_values:data.team_player.values_tr_vl, # clean values

                       team_match['indices']:data.team_match.indices_tr_vl, 
                       team_match['values']:team_match_values_noisy, # noisy values
                       team_match_noise_mask:team_match_noise,
                       team_match_values:data.team_match.values_tr_vl # clean values
                      }
            
            loss_vl, team_match_vals_out_vl = sess.run([rec_loss_vl, team_match_out_vl['values']], feed_dict=vl_dict)            
            losses_vl.append(loss_vl)



            mean_tr = data.team_match.mean_tr
            non_noise = 1 - team_match_noise
            loss_vl_mean = np.sqrt( np.sum(((data.team_match.values_tr_vl - mean_tr)**2)*non_noise) / np.sum(non_noise) )
            losses_vl_mean.append(loss_vl_mean)







            # split = data.team_match.split


            # in_t = sparse_transpose(data.team_match.indices_tr_vl, data.team_match.values_tr_vl, data.team_match.shape, split[split < 2])
            # out_t = sparse_transpose(data.team_match.indices_tr_vl, team_match_vals_out_vl, data.team_match.shape, split[split < 2])
            # vals_in_t = in_t['values']
            # vals_out_t = out_t['values']
            
            # noise = 1 - in_t['split']            

            # preds_in = np.sign(np.reshape(vals_in_t[noise == 0], [-1,2])[:,0])
            # num_vals = preds_in.shape[0]

            # vals_out = np.reshape(vals_out_t[noise == 0], [-1,2])                    
            # preds_out = (np.abs(vals_out[:,0]) + np.abs(vals_out[:,1]) ) / 2
            # preds_out = np.sign(np.round(np.sign(vals_out[:,0]) * preds_out, decimals=1))

            # # print(vals_out[0:20,:])
            # print('preds_in:  ', preds_in[0:20].astype(np.float32))
            # print('preds_out: ', preds_out[0:20])


            # pred_accuracy_vl = np.sum(preds_in == preds_out) / num_vals




            split = data.team_match.split
            pred_accuracy_vl = table_prediction_accuracy(data.team_match.indices_tr_vl, data.team_match.values_tr_vl, team_match_vals_out_vl, data.team_match.shape, split[split < 2])





            accuracies_vl.append(pred_accuracy_vl)


            if pred_accuracy_vl > accuracy_vl_best:
                accuracy_vl_best = pred_accuracy_vl
                accuracy_vl_best_ep = ep
                ## TODO save model

            
            if opts['verbosity'] > 0:
                # print("epoch {:5d}. training loss: {:5.15f} \t validation loss: {:5.5f} \t predicting mean: {:5.5f} \t train accuracy rate: {:5.5f} \t val accuracy rate: {:5.5f}".format(ep+1, loss_tr, loss_vl, loss_vl_mean, pred_accuracy_tr, pred_accuracy_vl))
                print("epoch {:5d}. train accuracy rate: {:5.5f} \t val accuracy rate: {:5.5f} \t best val accuracy rate: {:5.5f} at epoch {:5d}".format(ep+1, pred_accuracy_tr, pred_accuracy_vl, accuracy_vl_best, accuracy_vl_best_ep+1))

        

        show_last = 1000
        plt.title('RMSE Loss')
        plt.plot(range(opts['epochs'])[-show_last:], losses_tr[-show_last:], '.-', color='blue')
        plt.plot(range(opts['epochs'])[-show_last:], losses_vl[-show_last:], '.-', color='green')
        plt.plot(range(opts['epochs'])[-show_last:], losses_vl_mean[-show_last:], '.-', color='red')
        plt.xlabel('epoch')
        plt.ylabel('RMSE')
        plt.legend(('training', 'validation', 'mean'))
        # plt.show()
        plt.savefig("rmse.pdf", bbox_inches='tight')
        plt.clf()

        plt.title('Prediction Accuracy')
        plt.plot(range(opts['epochs'])[-show_last:], accuracies_tr[-show_last:], '.-', color='blue')
        plt.plot(range(opts['epochs'])[-show_last:], accuracies_vl[-show_last:], '.-', color='green')
        plt.plot(range(opts['epochs'])[-show_last:], [.46 for _ in range(opts['epochs'])[-show_last:]], '.-', color='red')
        plt.plot(range(opts['epochs'])[-show_last:], [.53 for _ in range(opts['epochs'])[-show_last:]], '.-', color='yellow')
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.legend(('training prediction', 'validation prediction', 'baseline', 'experts'))        
        # plt.show()
        plt.savefig("pred.pdf", bbox_inches='tight')



if __name__ == "__main__":
    np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)


    units = 128
    activation = tf.nn.relu
    dropout_rate = 0.2    


    opts = {'epochs':2000,
            'data_split':[.8, .2, .0], # train, validation, test split
            'noise_rate':dropout_rate, # match vl/tr or ts/(tr+vl) ?
            'regularization_rate':.00001,
            'learning_rate':.0001,
            'layer_opts':{'pool_mode':'mean',
                          'dropout_rate':dropout_rate,
                          'layers':[{'type':ExchangeableLayer, 'units':units, 'activation':activation},
                                    {'type':FeatureDropoutLayer, 'units':units},
                                    {'type':ExchangeableLayer, 'units':units, 'activation':activation},                                    
                                    # {'type':ExchangeableLayer, 'units':units, 'activation':activation},
                                    # {'type':ExchangeableLayer, 'units':units, 'activation':activation},
                                    # {'type':ExchangeableLayer, 'units':units, 'activation':activation},
                                    {'type':ExchangeableLayer, 'units':1,  'activation':None},
                                   ],
                         },
            'debug':True,
            'verbosity':2
            }
    main(opts)






