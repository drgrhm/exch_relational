import numpy as np
import tensorflow as tf
import matplotlib
# import scipy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
from copy import deepcopy
from util import *
from data_util import DataLoader, Table
from model import Model
from layers import ExchangeableLayer, FeatureDropoutLayer


## Noise mask has 0's corresponding to values to be predicted
def table_rmse_loss(values, values_out, noise_mask):
    prediction_mask = tf.cast(1 - noise_mask, tf.float32)
    return tf.sqrt( tf.reduce_sum(((values - values_out)**2)*prediction_mask) / (tf.reduce_sum(prediction_mask) + 1e-10) )


def table_prediction_rmse_loss(values, values_out, noise_mask, num_features):
    # prediction_mask = tf.cast(1 - noise_mask, tf.float32)

    prediction_mask = tf.reshape(tf.cast(1 - noise_mask, tf.float32), [-1,num_features])
    vals = tf.reshape(values, [-1,num_features]) * prediction_mask
    vals_out = tf.reshape(values_out, [-1,num_features]) * prediction_mask

    num_vals = tf.shape(vals)[0]
    categories = tf.cast(tf.reshape(tf.tile(tf.range(num_features), [num_vals]), [-1,num_features]), tf.float32)

    preds = tf.reduce_sum(vals * categories, axis=1)
    preds_out = tf.reduce_sum(vals_out * categories, axis=1)

    num_vals_prediction = tf.reduce_sum(prediction_mask) / num_features

    return tf.sqrt(tf.reduce_sum( ((preds - preds_out)**2) ) / num_vals_prediction)




def table_cross_entropy_loss(values, values_out, noise_mask, num_features):
    prediction_mask = tf.cast(1 - noise_mask, tf.float32)
    vals = tf.reshape(prediction_mask * values, shape=[-1,num_features])
    out = tf.reshape(prediction_mask * values_out, shape=[-1,num_features])
    return - tf.reduce_mean(tf.reduce_sum(vals * (out - tf.reduce_logsumexp(out, axis=1, keep_dims=True)), axis=1))
    


def table_ordinal_hinge_loss(values, values_out, noise_mask, num_features):
    
    prediction_mask = tf.reshape(tf.cast(1 - noise_mask, tf.float32), [-1,num_features])
    
    vals = tf.reshape(values, [-1,num_features]) * prediction_mask
    vals_out = tf.reshape(values_out, [-1,num_features]) * prediction_mask

    num_vals = tf.shape(vals)[0]
    categories = tf.cast(tf.reshape(tf.tile(tf.range(num_features), [num_vals]), [-1,num_features]), tf.float32)

    preds = tf.reduce_sum(vals * categories, axis=1)[:,None]
    preds_out = tf.reduce_sum(vals_out * categories, axis=1)[:,None]

    greater = tf.cast(tf.greater_equal(categories, preds), tf.float32)
    less = tf.cast(tf.less_equal(categories, preds), tf.float32)
    not_equal = tf.cast(tf.not_equal(categories, preds), tf.float32)

    preds_out = preds_out * (greater - less)
    preds_out = (preds_out + 1) * not_equal
    out = categories * (less - greater) + preds_out
    out = tf.maximum(out, tf.zeros_like(out))
    out = tf.reduce_sum(out, axis=1)

    return tf.reduce_sum(out)


def misprediction_loss(values, values_out, noise_mask, num_features):

    pred_mask = 1 - noise_mask
    vals = values * pred_mask
    vals = tf.reshape(vals, [-1, num_features])

    vals_out = values_out * pred_mask
    vals_out = tf.reshape(vals_out, [-1, num_features])

    num_vals = tf.reduce_sum(pred_mask) / num_features
    
    probs_out = tf.exp(vals_out - tf.reduce_max(vals_out, axis=1)[:,None])
    probs_out = probs_out / tf.reduce_sum(probs_out, axis=1)[:,None]

    # preds = tf.zeros_like(vals)
    # max_inds = tf.argmax(vals_out, axis=1)
    # preds[np.arange(num_vals), max_inds] = 1.0
    # return 1 - tf.reduce_sum(preds*vals) / num_vals

    return tf.reduce_sum(1 - vals*probs_out)






def one_hot_prediction_accuracy(values, values_out, noise_mask, num_features):

    vals = np.reshape(values[noise_mask == 0], [-1, num_features])
    vals_out = np.reshape(values_out[noise_mask == 0], [-1, num_features])

    num_vals = vals.shape[0]
    
    probs_out = np.exp(vals_out - np.max(vals_out, axis=1)[:,None])
    probs_out = probs_out / np.sum(probs_out, axis=1)[:,None]
    
    preds = np.zeros_like(vals)
    max_inds = np.argmax(vals_out, axis=1)
    
    preds[np.arange(num_vals), max_inds] = 1

    # print(vals[20:40])
    # print('')
    # print(preds[20:40])

    print('input:  ', np.mean(vals, axis=0))
    print('output: ', np.mean(preds, axis=0))

    return np.sum(preds*vals) / num_vals


def make_uniform_noise_mask(noise_rate, num_vals):
    """A 0/1 noise mask. 0's correspond to dropped out values."""
    n0 = int(noise_rate * num_vals)
    n1 = num_vals - n0
    noise_mask = np.concatenate((np.zeros(n0), np.ones(n1)))
    np.random.shuffle(noise_mask)
    return noise_mask

def make_by_col_noise(noise_rate, num_vals, shape, column_indices):
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




def main(opts, restore_point=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    if opts['debug']:
        np.random.seed(12345)

    data = DataLoader(opts['data_folder'], opts['data_set'], opts['split_sizes'], opts['model_opts']['units_in'], opts['team_match_one_hot'])


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

        rec_loss_tr = table_rmse_loss(team_match_values, team_match_out_tr['values'], team_match_noise_mask)
        # rec_loss_tr = table_cross_entropy_loss(team_match_values, team_match_out_tr['values'], team_match_noise_mask, opts['model_opts']['units_out'])
        # rec_loss_tr = table_ordinal_hinge_loss(team_match_values, team_match_out_tr['values'], team_match_noise_mask, opts['model_opts']['units_out'])
        # rec_loss_tr = table_prediction_rmse_loss(team_match_values, team_match_out_tr['values'], team_match_noise_mask, opts['model_opts']['units_out'])
        # rec_loss_tr = misprediction_loss(team_match_values, team_match_out_tr['values'], team_match_noise_mask, opts['model_opts']['units_out'])

        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss_tr = rec_loss_tr + reg_loss
        
        rec_loss_vl = table_rmse_loss(team_match_values, team_match_out_vl['values'], team_match_noise_mask)
        # rec_loss_vl = table_cross_entropy_loss(team_match_values, team_match_out_vl['values'], team_match_noise_mask, opts['model_opts']['units_out'])
        # rec_loss_vl = table_ordinal_hinge_loss(team_match_values, team_match_out_vl['values'], team_match_noise_mask, opts['model_opts']['units_out'])
        # rec_loss_vl = table_prediction_rmse_loss(team_match_values, team_match_out_vl['values'], team_match_noise_mask, opts['model_opts']['units_out'])
        # rec_loss_vl = misprediction_loss(team_match_values, team_match_out_vl['values'], team_match_noise_mask, opts['model_opts']['units_out'])


        train_step = tf.train.AdamOptimizer(opts['learning_rate']).minimize(total_loss_tr)
        # train_step = tf.train.GradientDescentOptimizer(opts['learning_rate']).minimize(total_loss_tr)
        # train_step = tf.train.RMSPropOptimizer(opts['learning_rate']).minimize(total_loss_tr)
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
        losses_vl_baseline = []

        accuracies_tr = []
        accuracies_vl = []

        accuracy_vl_best = 0
        accuracy_vl_best_ep = 0


        for ep in range(opts['restore_point_epoch'] + 1, opts['restore_point_epoch'] + opts['epochs'] + 1):
            if opts['verbosity'] > 0:
                print('------- epoch:', ep, '-------')

            ## Training 
            team_player_noise = np.ones_like(data.team_player.values_tr)
            team_player_values_noisy = np.copy(data.team_player.values_tr)

            team_match_noise = make_by_col_noise(opts['noise_rate'], data.team_match.num_values_tr, data.team_match.shape, data.team_match.indices_tr[:,1]) 
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
              
            num_features = opts['model_opts']['units_out']
            pred_accuracy_tr = one_hot_prediction_accuracy(data.team_match.values_tr, team_match_vals_out_tr, noise_out, num_features)
            accuracies_tr.append(pred_accuracy_tr)



            ## Validation
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


            means = np.mean(np.reshape(data.team_match.values_tr_vl, [-1, num_features]), axis=0)
            vals = np.reshape(noise_out * data.team_match.values_tr_vl, [-1,num_features])
            out = np.reshape(means, [-1,num_features])
            loss_vl_baseline = - np.mean(np.sum(vals * (out - np.max(out, axis=1)), axis=1))
            losses_vl_baseline.append(loss_vl_baseline)    
            num_vals = vals.shape[0]

            pred_accuracy_vl = one_hot_prediction_accuracy(data.team_match.values_tr_vl, team_match_vals_out_vl, noise_out, num_features)
            accuracies_vl.append(pred_accuracy_vl)

            if pred_accuracy_vl > accuracy_vl_best:
                accuracy_vl_best = pred_accuracy_vl
                accuracy_vl_best_ep = ep

                if opts['save_model']:
                    path = os.path.join(opts['checkpoints_folder'], 'epoch_{:05d}'.format(ep) + '.ckpt')
                    saver.save(sess, path)

            
            if opts['verbosity'] > 0:
                # print("epoch {:5d}. training loss: {:5.15f} \t validation loss: {:5.5f} \t predicting mean: {:5.5f}".format(ep+1, loss_tr, loss_vl, loss_vl_baseline))
                print("epoch {:5d}. train accuracy rate: {:5.5f} \t val accuracy rate: {:5.5f} \t best val accuracy rate: {:5.5f} at epoch {:5d}".format(ep, pred_accuracy_tr, pred_accuracy_vl, accuracy_vl_best, accuracy_vl_best_ep))

        

        show_last = opts['epochs']
        plt.title('CE Loss')
        plt.plot(range(opts['epochs'])[-show_last:], losses_vl_baseline[-show_last:], '.-', color='red')
        plt.plot(range(opts['epochs'])[-show_last:], losses_tr[-show_last:], '.-', color='blue')
        plt.plot(range(opts['epochs'])[-show_last:], losses_vl[-show_last:], '.-', color='green')        
        plt.xlabel('epoch')
        plt.ylabel('CE')
        plt.legend(('mean', 'training', 'validation'))
        # plt.show()
        plt.savefig("rmse.pdf", bbox_inches='tight')
        plt.clf()

        plt.title('Prediction Accuracy')
        plt.plot(range(opts['epochs'])[-show_last:], [.46 for _ in range(opts['epochs'])[-show_last:]], '.-', color='pink')
        plt.plot(range(opts['epochs'])[-show_last:], [.53 for _ in range(opts['epochs'])[-show_last:]], '.-', color='yellow')
        plt.plot(range(opts['epochs'])[-show_last:], accuracies_tr[-show_last:], '.-', color='blue')
        plt.plot(range(opts['epochs'])[-show_last:], accuracies_vl[-show_last:], '.-', color='green')
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.legend(( 'baseline', 'experts', 'training prediction', 'validation prediction'))        
        # plt.show()
        plt.savefig("pred.pdf", bbox_inches='tight')



if __name__ == "__main__":
    np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)


    # data_set = 'debug'
    data_set = 'soccer'
    
    one_hot = True
    units_in = 3
    units = 256
    units_out = units_in

    activation = tf.nn.relu
    # activation = lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x) # Leaky Relu
    regularization_rate = 0.00001
    dropout_rate = 0.2
    skip_connections = True

    auto_restore = False
    save_model = False
    


<<<<<<< HEAD
    opts = {'epochs':500,
            'data_folder':'data',
            'data_set':data_set,
            'split_sizes':[.8, .2, .0], # train, validation, test split
            'noise_rate':dropout_rate, # match vl/tr or ts/(tr+vl) ?            
            'learning_rate':.0001,
            'team_match_one_hot':one_hot,
            'model_opts':{'pool_mode':'mean',
                          'dropout_rate':dropout_rate,
                          'regularization_rate':regularization_rate,
                          'units_in':units_in,
                          'units_out':units_out,
                          'layers':[{'type':ExchangeableLayer, 'units_out':units, 'activation':activation},
                                    # {'type':FeatureDropoutLayer, 'units_out':units},
                                    {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                    # {'type':FeatureDropoutLayer, 'units_out':units},
                                    # {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},                                      
                                    {'type':ExchangeableLayer, 'units_out':units_out,  'activation':None},
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






