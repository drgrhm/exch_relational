import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
from copy import deepcopy
from util import *
from data_util import DataLoader, Table
from model import Model
from layers import ExchangeableLayer, FeatureDropoutLayer
from pprint import pprint


## Noise mask has 1's corresponding to values to be predicted
def table_rmse_loss(values, values_out, noise_mask):
    return tf.sqrt( tf.reduce_sum(((values - values_out)**2)*noise_mask) / (tf.reduce_sum(noise_mask) + 1e-10) )
    # return tf.sqrt( tf.reduce_sum(((values - values_out)**2)) / tf.cast(tf.shape(values)[0], tf.float32) ) 


    



def table_ce_loss(values, values_out, mask_split, dim):
    out = tf.reshape(values_out, shape=[-1,5])
    return - tf.reduce_mean(mask_split * tf.reduce_sum(tf.reshape(values, shape=[-1, dim]) * (out - tf.reduce_logsumexp(out, axis=1, keep_dims=True)), axis=1))


# def one_hot(x, d):
#     x = x.astype(np.int32)
#     n = x.shape[0]
#     out = np.zeros((n, d))
#     out[np.arange(n), x-1] = 1
#     return out.flatten()

# def one_hot_inv(x, d):
#     return np.argmax(np.reshape(x, [-1,d]), axis=1) + 1

# def one_hot_inv(x, d):

#     return tf.cast(tf.argmax(tf.reshape(x, [-1,d]), axis=1) + 1, tf.float32)


def main(opts, restore_point=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    if opts['debug']:
        np.random.seed(opts['seed'])
        # random.seed(opts['seed'])

    data = DataLoader(opts['data_folder'], opts['data_set'], opts['split_rates'], verbosity=opts['verbosity'])

    with tf.Graph().as_default():

        if opts['debug']:
            tf.set_random_seed(opts['seed'])

        model = Model(**opts['model_opts'])

        placeholders = {}
        tables_pl = {}
        for t, table in data.tables.items():
            inds = tf.placeholder(tf.int32, shape=(None, 2), name=(table.name + '_indices'))
            vals_clean = tf.placeholder(tf.float32, shape=(None), name=(table.name + '_values_clean'))
            vals_noisy = tf.placeholder(tf.float32, shape=(None), name=(table.name + '_values_noisy'))
            noise_mask = tf.placeholder(tf.float32, shape=(None), name=(table.name + '_noise_mask'))
            
            tables_pl[table.name] = {'indices':inds, 'values':vals_noisy, 'shape':table.shape}
            placeholders[table.name] = {'values_clean':vals_clean, 'noise_mask':noise_mask}

        tables_out_tr = model.get_output(tables_pl)
        tables_out_vl = model.get_output(tables_pl, reuse=True, is_training=False)

        ## Predictions made on table 0 only
        rmse_loss_tr = table_rmse_loss(placeholders['table_0']['values_clean'], tables_out_tr['table_0']['values'], placeholders['table_0']['noise_mask'])                
        rmse_loss_vl = table_rmse_loss(placeholders['table_0']['values_clean'], tables_out_vl['table_0']['values'], placeholders['table_0']['noise_mask'])

        # ce_loss_tr = table_ce_loss(placeholders['table_0']['values_clean'], tables_out_tr['table_0']['values'], placeholders['table_0']['noise_mask'], 5)

        total_loss_tr = rmse_loss_tr

        mean = np.mean(data.tables['table_0'].values[data.tables['table_0'].split == 0])    # mean training value
        mean = mean * tf.ones_like(tables_out_vl['table_0']['values'])
        rmse_loss_mean = table_rmse_loss(placeholders['table_0']['values_clean'], mean, placeholders['table_0']['noise_mask'])        


        train_step = tf.train.AdamOptimizer(opts['learning_rate']).minimize(total_loss_tr)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        

        if opts['model_opts']['pool_mode'] == 'mean':
            noise_value = 0
        if opts['model_opts']['pool_mode'] == 'max':
            noise_value = -1e10

        losses_tr = []
        losses_vl = []
        losses_mean = []
        losses_buss = []

        loss_tr_best = 5.0
        loss_tr_best_ep = 0

        loss_vl_best = 5.0
        loss_vl_best_ep = 0


        for ep in range(opts['restore_point_epoch'] + 1, opts['restore_point_epoch'] + opts['epochs'] + 1):
            print('------- epoch:', ep, '-------')


            tr_dict = {}
            for _, table in data.tables.items():
                if table.predict:
                    inds = table.indices[table.split == 0] # training entries only
                    vals_clean = table.values[table.split == 0]
                    n = inds.shape[0]
                    n_noisy = int(opts['split_rates'][1] * n)
                    n_clean = n - n_noisy
                    mask = np.concatenate((np.zeros(n_clean), np.ones(n_noisy)))
                    np.random.shuffle(mask)
                    vals_noisy = np.copy(vals_clean)
                    vals_noisy[mask == 1] = noise_value
                else:
                    inds = table.indices    # not predicting, so all entries 
                    vals_clean = table.values
                    vals_noisy = np.copy(vals_clean)
                    mask = np.zeros_like(vals_clean)
            

                tr_dict[tables_pl[table.name]['indices']] = inds
                tr_dict[tables_pl[table.name]['values']] = vals_noisy   # noisy values 
                tr_dict[placeholders[table.name]['noise_mask']] = mask
                tr_dict[placeholders[table.name]['values_clean']] = vals_clean  # clean values 


            _, loss_tr = sess.run([train_step, total_loss_tr], feed_dict=tr_dict)
            losses_tr.append(loss_tr)

            if loss_tr < loss_tr_best:
                loss_tr_best = loss_tr
                loss_tr_best_ep = ep
                   

            vl_dict = {}
            for _, table in data.tables.items():
                if table.predict:                    
                    inds = table.indices[table.split < 2] # training and validation entries
                    vals_clean = table.values[table.split < 2]
                    mask = table.split[table.split < 2]
                    vals_noisy = np.copy(vals_clean)
                    vals_noisy[mask == 1] = noise_value
                else: 
                    inds = table.indices
                    vals_clean = table.values
                    mask = np.zeros_like(table.values)
                    vals_noisy = np.copy(vals_clean)


                vl_dict[tables_pl[table.name]['indices']] = inds
                vl_dict[tables_pl[table.name]['values']] = vals_noisy   # noisy values 
                vl_dict[placeholders[table.name]['noise_mask']] = mask
                vl_dict[placeholders[table.name]['values_clean']] = vals_clean  # clean values 

            loss_vl, loss_mean, out_vl = sess.run([rmse_loss_vl, rmse_loss_mean, tables_out_vl['table_0']['values']], feed_dict=vl_dict)
            losses_vl.append(loss_vl)
            losses_mean.append(loss_mean)            

            if loss_vl < loss_vl_best:
                loss_vl_best = loss_vl
                loss_vl_best_ep = ep


            print(data.tables['table_0'].values[data.tables['table_0'].split == 1][0:15])
            print(out_vl[0:15])


            print("epoch {:5d}. train loss: {:5.5f}, val loss: {:5.5f} \t best train loss: {:5.5f} at epoch {:5d}, best val loss: {:5.5f} at epoch {:5d}".format(ep, loss_tr, loss_vl, loss_tr_best, loss_tr_best_ep, loss_vl_best, loss_vl_best_ep))
            

        show_last = opts['epochs']
        plt.title('RMSE Loss')
        plt.plot(range(opts['epochs'])[-show_last:], losses_mean[-show_last:], '.-', color='red')
        plt.plot(range(opts['epochs'])[-show_last:], losses_tr[-show_last:], '.-', color='blue')
        plt.plot(range(opts['epochs'])[-show_last:], losses_vl[-show_last:], '.-', color='green')
        plt.xlabel('epoch')
        plt.ylabel('RMSE')
        plt.legend(('mean', 'training', 'validation'))
        # plt.show()
        plt.savefig("rmse.pdf", bbox_inches='tight')
        plt.clf()



if __name__ == "__main__":
    
    ##....
    np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)
    ##....

    data_set = 'yelp'    
    # activation = tf.nn.relu    
    activation = lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x) # Leaky Relu    
    regularization_rate = 0.00001
    dropout_rate = .5
    skip_connections = False
    units_in = 1

    opts = {'epochs':1,
            'learning_rate':.0001,
            'data_folder':'data',
            'data_set':data_set,
            'split_rates':[.8, .2, .0], # train, validation, test split
            'noise_rate':dropout_rate,            
            'model_opts':{'pool_mode':'mean',
                          'dropout_rate':dropout_rate,
                          'regularization_rate':regularization_rate,
                          'units_in':units_in,
                          # 'units_out':units_out,
                          'layers':[
                                    {'type':ExchangeableLayer, 'units_out':4, 'activation':activation, 'skip_connections':skip_connections},
                                    {'type':ExchangeableLayer, 'units_out':4, 'activation':activation, 'skip_connections':skip_connections},
                                    {'type':ExchangeableLayer, 'units_out':units_in, 'activation':None, 'skip_connections':skip_connections}
                                    ],
                         },
            'verbosity':2,    
            'restore_point_epoch':-1, # To continue counting epochs after loading saved model
            'debug':False,
            'seed':12345,
            }

    restore_point = None

    main(opts, restore_point)
















