import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
from copy import deepcopy
from data_util import DataLoader, Table
from model import Model
from layers import ExchangeableLayer, FeatureDropoutLayer


## Noise mask has 0's corresponding to values to be predicted
def table_rmse_loss(values, values_out, noise_mask, num_features):

    values = tf.reshape(values, [-1, num_features])[:,0]
    values_out = tf.reshape(values_out, [-1, num_features])[:,0]

    non_noise = tf.reshape(tf.cast(1 - noise_mask, tf.float32), [-1, num_features])[:,0]

    AA = values - values_out
    
    return tf.sqrt( tf.reduce_sum((AA**2)*non_noise) / (tf.reduce_sum(non_noise) + 1e-10) )


def table_prediction_accuracy(indices, values, values_out, shape, split, num_features, threshold):
    
    values = np.reshape(values, [-1, num_features])[:,0]
    values_out = np.reshape(values_out, [-1, num_features])[:,0]


    in_t = sparse_transpose(indices, values, shape, split)
    out_t = sparse_transpose(indices, values_out, shape, split)
    vals_in_t = in_t['values']
    vals_out_t = out_t['values']
    noise = 1 - in_t['split']

    vals_in_t = np.reshape(vals_in_t[noise == 0], [-1,2])

    preds_in = np.sign(np.round(vals_in_t[:,0] - vals_in_t[:,1], threshold))
    num_vals = preds_in.shape[0]

    vals_out_t = np.reshape(vals_out_t[noise == 0], [-1,2])
    preds_out = np.sign(np.round(vals_out_t[:,0] - vals_out_t[:,1], threshold))

    # # preds_out = (np.abs(vals_out[:,0]) + np.abs(vals_out[:,1]) ) / 2
    # # preds_out = np.sign(np.round(np.sign(vals_out[:,0]) * preds_out, decimals=1)) ## TODO better way 

    print("")
    print('vals_in_t:  ', preds_in[0:20].astype(np.float32))
    print('vals_out_t: ', preds_out[0:20])

    return np.sum(preds_in == preds_out) / num_vals

    # return 0



# def table_prediction_accuracy(values, values_out, noise_mask, num_features):

#     print()

#     vals = np.reshape(values, [-1, num_features])[noise_mask == 0]
#     vals_out = np.reshape(values_out, [-1, num_features])[noise_mask == 0]

#     num_vals = vals.shape[0]
    
#     probs_out = np.exp(vals_out - np.max(vals_out, axis=1)[:,None])
#     probs_out = probs_out / np.sum(probs_out, axis=1)[:,None]
    
#     preds = np.zeros_like(vals)
#     max_inds = np.argmax(vals_out, axis=1)
    
#     preds[np.arange(num_vals), max_inds] = 1

#     # print(vals[20:40])
#     # print('')
#     # print(preds[20:40])

#     print('input:  ', np.mean(vals, axis=0))
#     print('output: ', np.mean(preds, axis=0))

#     return np.sum(preds*vals) / num_vals




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




def main(opts, restore_point=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    tr, vl, ts = opts['data_split']

    if opts['debug']:
        np.random.seed(12345)

    data = DataLoader('data', opts['model_opts']['units_in'], opts['data_split'])

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

        model = Model(**opts['model_opts'])

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
        rec_loss_tr += table_rmse_loss(team_match_values, team_match_out_tr['values'], team_match_noise_mask, opts['model_opts']['units_in'])
        reg_loss = np.sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss_tr = rec_loss_tr + reg_loss

        rec_loss_vl = 0
        # rec_loss_vl += table_rmse_loss(team_player_values, team_player_out_vl['values'], team_player_noise_mask)
        rec_loss_vl += table_rmse_loss(team_match_values, team_match_out_vl['values'], team_match_noise_mask, opts['model_opts']['units_in'])

        train_step = tf.train.AdamOptimizer(opts['learning_rate']).minimize(total_loss_tr)
        # train_step = tf.train.GradientDescentOptimizer(opts['learning_rate']).minimize(total_loss_tr)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        if restore_point is not None:
            saver.restore(sess, restore_point)

        if opts['model_opts']['pool_mode'] == 'mean':
            noise_value = 0
        if opts['model_opts']['pool_mode'] == 'max':
            noise_value = -1e10

        losses_tr = []
        losses_vl = []
        losses_vl_mean = []
        accuracies_tr = []
        accuracies_vl = []

        accuracy_tr_best = 0
        accuracy_tr_best_ep = 0

        accuracy_vl_best = 0
        accuracy_vl_best_ep = 0


        for ep in range(opts['restore_point_epoch'] + 1, opts['restore_point_epoch'] + opts['epochs'] + 1):
            if opts['verbosity'] > 0:
                print('------- epoch:', ep, '-------')

            team_player_noise = np.ones_like(data.team_player.values_tr)
            team_player_values_noisy = np.copy(data.team_player.values_tr)

            team_match_noise = make_by_col_noise_mask(opts['noise_rate'], data.team_match.num_values_tr, data.team_match.shape, data.team_match.indices_tr[:,1]) 
            noise_in = np.array([i for i in team_match_noise for _ in range(opts['model_opts']['units_in'])])
            noise_out = np.array([i for i in team_match_noise for _ in range(opts['model_opts']['units_out'])])
            team_match_values_noisy = np.copy(data.team_match.values_tr)
            team_match_values_noisy[noise_in == 0] = noise_value

            tr_dict = {team_player['indices']:data.team_player.indices_tr, 
                       team_player['values']:team_player_values_noisy, # noisy values
                       team_player_noise_mask:team_player_noise,
                       team_player_values:data.team_player.values_tr, # clean values

                       team_match['indices']:data.team_match.indices_tr, 
                       team_match['values']:team_match_values_noisy, # noisy values
                       team_match_noise_mask:noise_out,
                       team_match_values:data.team_match.values_tr # clean values
                      }


            _, loss_tr, team_match_vals_out_tr, = sess.run([train_step, total_loss_tr, team_match_out_tr['values']], feed_dict=tr_dict)
            losses_tr.append(loss_tr)
              
            pred_accuracy_tr = table_prediction_accuracy(data.team_match.indices_tr, 
                                                         data.team_match.values_tr, 
                                                         team_match_vals_out_tr, 
                                                         data.team_match.shape, 
                                                         team_match_noise, 
                                                         opts['model_opts']['units_out'],
                                                         opts['draw_threshold'])
            accuracies_tr.append(pred_accuracy_tr)

            if pred_accuracy_tr > accuracy_tr_best:
                accuracy_tr_best = pred_accuracy_tr
                accuracy_tr_best_ep = ep

            team_player_noise = np.ones_like(data.team_player.values_tr_vl)
            team_player_values_noisy = np.copy(data.team_player.values_tr_vl)

            team_match_noise = 1 - data.team_match.split[data.team_match.split <= 1]
            noise_in = np.array([i for i in team_match_noise for _ in range(opts['model_opts']['units_in'])])
            noise_out = np.array([i for i in team_match_noise for _ in range(opts['model_opts']['units_out'])])
            
            team_match_values_noisy = np.copy(data.team_match.values_tr_vl)
            team_match_values_noisy[noise_in == 0] = noise_value

            vl_dict = {team_player['indices']:data.team_player.indices_tr_vl, 
                       team_player['values']:team_player_values_noisy, # noisy values
                       team_player_noise_mask:team_player_noise,
                       team_player_values:data.team_player.values_tr_vl, # clean values

                       team_match['indices']:data.team_match.indices_tr_vl, 
                       team_match['values']:team_match_values_noisy, # noisy values
                       team_match_noise_mask:noise_out,
                       team_match_values:data.team_match.values_tr_vl # clean values
                      }
            
            loss_vl, team_match_vals_out_vl = sess.run([rec_loss_vl, team_match_out_vl['values']], feed_dict=vl_dict)            
            losses_vl.append(loss_vl)


            mean_tr = data.team_match.mean_tr
            non_noise = 1 - team_match_noise
            values_in = np.reshape(data.team_match.values_tr_vl, [-1, opts['model_opts']['units_in']])[:,0]
            loss_vl_mean = np.sqrt( np.sum(((values_in - mean_tr)**2)*non_noise) / np.sum(non_noise) )
            losses_vl_mean.append(loss_vl_mean)

            split = data.team_match.split
            pred_accuracy_vl = table_prediction_accuracy(data.team_match.indices_tr_vl, 
                                                         data.team_match.values_tr_vl, 
                                                         team_match_vals_out_vl, 
                                                         data.team_match.shape, 
                                                         split[split < 2], 
                                                         opts['model_opts']['units_out'], 
                                                         opts['draw_threshold'])
            accuracies_vl.append(pred_accuracy_vl)


            if pred_accuracy_vl > accuracy_vl_best:
                accuracy_vl_best = pred_accuracy_vl
                accuracy_vl_best_ep = ep

                if opts['save_model']:
                    path = os.path.join(opts['checkpoints_folder'], 'epoch_{:05d}'.format(ep) + '.ckpt')
                    saver.save(sess, path)

            
            if opts['verbosity'] > 0:
                print("epoch {:5d}. train accuracy rate: {:5.5f} \t val accuracy rate: {:5.5f} \t best train accuracy rate: {:5.5f} at epoch {:5d} \t best val accuracy rate: {:5.5f} at epoch {:5d}".format(ep, pred_accuracy_tr, pred_accuracy_vl, accuracy_tr_best, accuracy_tr_best_ep, accuracy_vl_best, accuracy_vl_best_ep))

        

        # show_last = opts['epochs']
        show_last = 1900
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
    units_in = 2
    units_out = units_in
    # activation = tf.nn.tanh
    activation = tf.nn.relu
    # activation = lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x) # Leaky Relu
    dropout_rate = 0.2
    regularization_rate = 0.0001
    auto_restore = False
    save_model = False
    skip_connections = True



    opts = {'epochs':500,
            'data_split':[.8, .2, .0], # train, validation, test split
            'noise_rate':dropout_rate, # match vl/tr or ts/(tr+vl) ?            
            'learning_rate':.005,
            'draw_threshold':1, 
            'model_opts':{'pool_mode':'mean',
                          'dropout_rate':dropout_rate,
                          'regularization_rate':regularization_rate,
                          'units_in':units_in,
                          'units_out':units_out,
                          'layers':[{'type':ExchangeableLayer, 'units':units, 'activation':activation},
                                    # {'type':FeatureDropoutLayer, 'units':units},
                                    {'type':ExchangeableLayer, 'units':units, 'activation':activation, 'skip_connections':skip_connections},
                                    # {'type':FeatureDropoutLayer, 'units':units},
                                    {'type':ExchangeableLayer, 'units':units, 'activation':activation, 'skip_connections':skip_connections},               
                                    {'type':ExchangeableLayer, 'units':units_out,  'activation':None},
                                   ],
                         },
            'verbosity':2,
            'checkpoints_folder':'checkpoints',
            'restore_point_epoch':-1,
            'save_model':save_model,
            'debug':True,
            }

    restore_point = None

    if auto_restore:         
        restore_point_epoch = sorted(glob.glob(opts['checkpoints_folder'] + "/epoch_*.ckpt*"))[-1].split(".")[0].split("_")[-1]
        restore_point = opts['checkpoints_folder'] + "/epoch_" + restore_point_epoch + ".ckpt"
        opts['restore_point_epoch'] = int(restore_point_epoch)

    main(opts, restore_point)






