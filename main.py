import numpy as np
import tensorflow as tf
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import os
import glob
import time
# from tqdm import tqdm 
from copy import deepcopy
from util import *
from data_util import DataLoader, Table
from model import Model
from layers import ExchangeableLayer, FeatureDropoutLayer
from pprint import pprint


## Noise mask has 1's corresponding to values to be predicted
def table_rmse_loss(values, values_out, noise_mask):
    return tf.sqrt( tf.reduce_sum(((values - values_out)**2)*noise_mask) / (tf.reduce_sum(noise_mask) + 1e-10) )
    # return tf.reduce_sum(((values - values_out)**2)*noise_mask)


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


def sample_uniform(tables, sample_rate, max_split_val):
    # num_vals = mask_indices.shape[0]
    # minibatch_size = np.minimum(minibatch_size, num_vals)
    # for n in range(iters_per_epoch):
    #     sample = np.random.choice(num_vals, size=minibatch_size, replace=False)
    #     yield sample

    num_samples = np.ceil(1 / sample_rate).astype(np.int64)
    for i in range(num_samples):
        out = {}
        for t, table in tables.items():
            num_obs = np.sum(table.split <= max_split_val)
            sample_size = int(sample_rate * num_obs)
            out[t] = np.random.choice(num_obs, sample_size, replace=False)
        yield out




def main(opts, restore_point=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    if opts['debug']:
        np.random.seed(opts['seed'])

    data = DataLoader(opts['data_folder'], opts['data_set'], opts['split_rates'], verbosity=opts['verbosity'])

    if not os.path.isdir(opts['ckpt_folder']):
        os.makedirs(opts['ckpt_folder'])

    with tf.Graph().as_default():

        if opts['debug']:
            tf.set_random_seed(opts['seed'])

        model = Model(**opts['model_opts'])

        placeholders = {}
        tables_pl = {}
        for t, table in data.tables.items():
            inds = tf.placeholder(tf.int32, shape=(None, 2), name=(table.name + '_indices'))
            vals_clean = tf.placeholder(tf.float64, shape=(None), name=(table.name + '_values_clean'))
            vals_noisy = tf.placeholder(tf.float64, shape=(None), name=(table.name + '_values_noisy'))
            noise_mask = tf.placeholder(tf.float64, shape=(None), name=(table.name + '_noise_mask'))
            
            tables_pl[table.name] = {'indices':inds, 'values':vals_noisy, 'shape':table.shape, 'entities':table.entities}
            placeholders[table.name] = {'values_clean':vals_clean, 'noise_mask':noise_mask}

        tables_out_tr = model.get_output(tables_pl)
        tables_out_vl = model.get_output(tables_pl, reuse=True, is_training=False)

        ## Predictions made on table 0 only
        rmse_loss_tr = table_rmse_loss(placeholders['table_0']['values_clean'], tables_out_tr['table_0']['values'], placeholders['table_0']['noise_mask'])                
        rmse_loss_vl = table_rmse_loss(placeholders['table_0']['values_clean'], tables_out_vl['table_0']['values'], placeholders['table_0']['noise_mask'])

        # ce_loss_tr = table_ce_loss(placeholders['table_0']['values_clean'], tables_out_tr['table_0']['values'], placeholders['table_0']['noise_mask'], 5)

        total_loss_tr = rmse_loss_tr

        table = data.tables['table_0']
        mean_tr = np.mean(table.values[table.split == 0])    # mean training value
        vals = table.values[table.split < 2]
        mean_vals = mean_tr * np.ones_like(vals)
        mask = table.split[table.split < 2]
        rmse_loss_mean = np.sqrt( np.sum(((vals - mean_vals)**2)*mask) / (np.sum(mask) + 1e-10) )

        train_step = tf.train.AdamOptimizer(opts['learning_rate']).minimize(total_loss_tr)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count = {'GPU': 0}))
        sess.run(tf.global_variables_initializer())
        

        if opts['model_opts']['pool_mode'] == 'mean':
            noise_value = 0
        if opts['model_opts']['pool_mode'] == 'max':
            noise_value = -1e10

        losses_tr = []
        losses_vl = []
        losses_ts = []
        losses_mean = []
        # losses_buss = []        

        loss_tr_best = np.inf
        loss_tr_best_ep = 0

        loss_vl_best = np.inf
        loss_vl_best_ep = 0

        loss_vl_last_save = np.inf

        sample_rate = opts['sample_rate']


        saver = tf.train.Saver()
        if restore_point is not None:
            saver.restore(sess, restore_point)
            losses_file = opts['ckpt_folder'] + "/losses.npz"
            if os.path.isfile(losses_file):
                losses = np.load(losses_file)
                losses_tr = list(losses['losses_tr'])
                losses_vl = list(losses['losses_vl'])
                losses_mean = list(losses['losses_mean'])


        for ep in range(opts['restore_point_epoch'] + 1, opts['restore_point_epoch'] + opts['epochs'] + 1):
            begin = time.time()            
            print('------- epoch:', ep, '-------')

            loss_tr, loss_vl, loss_ts, loss_mean = 0., 0., 0., 0.          

            # Training
            tr_dict = {}
            for t, table in data.tables.items():

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



            _, bloss_tr = sess.run([train_step, total_loss_tr], feed_dict=tr_dict)
            loss_tr += bloss_tr

            losses_tr.append(loss_tr)
            if loss_tr < loss_tr_best:
                loss_tr_best = loss_tr
                loss_tr_best_ep = ep
            

            # Validation 
            vl_dict = {}
            for t, table in data.tables.items():
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


            bloss_vl, bout_vl = sess.run([rmse_loss_vl, tables_out_vl['table_0']['values']], feed_dict=vl_dict)
            loss_vl += bloss_vl
                

            print(data.tables['table_0'].values[data.tables['table_0'].split == 1][0:15])
            print(bout_vl[0:15])

            losses_vl.append(loss_vl)
            losses_mean.append(rmse_loss_mean)


            if loss_vl < loss_vl_best:
                loss_vl_best = loss_vl
                loss_vl_best_ep = ep
                if loss_vl_last_save - loss_vl_best > 1e-3: # save if we made a .01 improvement since last save
                    losses = {'losses_tr':losses_tr, 'losses_vl':losses_vl, 'losses_mean':losses_mean}
                    np.savez_compressed(opts['ckpt_folder'] + "/losses.npz", **losses)
                    save_path = saver.save(sess, opts['ckpt_folder'] + "/ep_{:05d}.ckpt".format(ep))                    
                    print(".... model saved in file: %s" % save_path)
                    loss_vl_last_save = loss_vl_best
            

            # Testing 
            ts_dict = {}
            for t, table in data.tables.items():
                if table.predict:                    
                    inds = table.indices # all entries
                    vals_clean = table.values
                    mask = (table.split == 2) * 1.
                    vals_noisy = np.copy(vals_clean)
                    vals_noisy[mask == 1] = noise_value
                else: 
                    inds = table.indices
                    vals_clean = table.values
                    mask = np.zeros_like(table.values)
                    vals_noisy = np.copy(vals_clean)

                ts_dict[tables_pl[table.name]['indices']] = inds
                ts_dict[tables_pl[table.name]['values']] = vals_noisy   # noisy values 
                ts_dict[placeholders[table.name]['noise_mask']] = mask
                ts_dict[placeholders[table.name]['values_clean']] = vals_clean  # clean values 


            bloss_ts, = sess.run([rmse_loss_vl], feed_dict=ts_dict)
            loss_ts += bloss_ts

            losses_ts.append(loss_ts)


            print("epoch {:5d} took {:.1f}s. train loss: {:5.5f}, val loss: {:5.5f}, test loss {:5.5f} \t best train loss: {:5.5f} at epoch {:5d}, best val loss: {:5.5f} at epoch {:5d}".format(ep, time.time() - begin, loss_tr, loss_vl, loss_ts, loss_tr_best, loss_tr_best_ep, loss_vl_best, loss_vl_best_ep))
            



        # epochs = opts['epochs'] + opts['restore_point_epoch'] + 1
        # plt.title('RMSE Loss')
        # plt.plot(range(epochs), losses_mean, '.-', color='red')
        # plt.plot(range(epochs), losses_tr, '.-', color='blue')
        # plt.plot(range(epochs), losses_vl, '.-', color='green')
        # plt.xlabel('epoch')
        # plt.ylabel('RMSE')
        # plt.legend(('mean', 'training', 'validation'))
        # # plt.show()
        # plt.savefig("rmse.pdf", bbox_inches='tight')
        # plt.clf()



if __name__ == "__main__":
    
    ##....
    np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)
    ##....

    data_set = 'yelp'    
    # activation = tf.nn.relu    
    activation = lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x) # Leaky Relu    
    regularization_rate = 0.00001
    dropout_rate = .5
    skip_connections = True
    units_in = 1
    units = 128
    auto_restore = False


    opts = {'epochs':2000,
            'learning_rate':.0001,
            'sample_rate':.2,
            'sampling_threshold':.90,
            'data_folder':'data',
            'data_set':data_set,
            'split_rates':[.8, .1, .1], # train, validation, test split
            'noise_rate':dropout_rate,
            'model_opts':{'pool_mode':'mean',
                          'dropout_rate':dropout_rate,
                          'regularization_rate':regularization_rate,
                          'units_in':units_in,
                          # 'units_out':units_out,
                          'layers':[
                                    {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                    {'type':FeatureDropoutLayer, 'dropout_rate':dropout_rate},
                                    {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                    {'type':FeatureDropoutLayer, 'dropout_rate':dropout_rate},
                                    {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                    {'type':FeatureDropoutLayer, 'dropout_rate':dropout_rate},
                                    {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                    {'type':ExchangeableLayer, 'units_out':units_in, 'activation':None, 'skip_connections':skip_connections}
                                    ],
                         },
            'verbosity':2,    
            'restore_point_epoch':-1, # To continue counting epochs after loading saved model
            'ckpt_folder':'checkpoints',
            'debug':False,
            'seed':12345,
            }



    restore_point = None
    if auto_restore:
        checkpoints = sorted(glob.glob(opts['ckpt_folder'] + "/*.ckpt*"))
        if len(checkpoints) > 0:
            restore_point_epoch = checkpoints[-1].split(".")[0].split("_")[-1]
            restore_point = opts['ckpt_folder'] + "/ep_" + restore_point_epoch + ".ckpt"
            print(".... restoring from %s\n" % restore_point)
            opts["restore_point_epoch"] = int(restore_point_epoch) # Pass num_epochs so far to start counting from there. In case of another crash 


    main(opts, restore_point)
















