import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import glob
import math
from util import *
from data_util import DataLoader, ToyDataLoader
from model import Model
from layers import ExchangeableLayer, FeatureDropoutLayer, PoolingLayer


def rmse_loss(values_in, values_out, noise_mask):
    diffs = ((values_in - values_out)**2) * noise_mask
    return tf.sqrt(tf.reduce_sum(diffs) / tf.reduce_sum(noise_mask))


def main(opts, restore_point=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    if opts['debug']:
        np.random.seed(opts['seed'])


    data = opts.get('data', None)
    if data is None:
        data = ToyDataLoader(opts['toy_data']['size'],
                             opts['toy_data']['sparsity'],
                             opts['split_sizes'],
                             opts['encoder_opts']['units_in'],
                             opts['toy_data']['embedding_size'],
                             opts['toy_data']['min_observed'])

    if opts['split_sizes'][2] == 0:
        mean = np.mean(data.tables['student_course'].values_tr)
        split = data.tables['student_course'].split
        loss_mean = np_rmse_loss(data.tables['student_course'].values_tr_vl, mean * np.ones_like(data.tables['student_course'].values_tr_vl), split[split == 1])  # Loss on validation set when predicting training mean
    else:
        mean = np.mean(data.tables['student_course'].values_tr_vl)
        split = data.tables['student_course'].split
        loss_mean = np_rmse_loss(data.tables['student_course'].values_all, mean * np.ones_like(data.tables['student_course'].values_all), 1. * (split == 2))  # Loss on test set when predicting training/validation mean


    with tf.Graph().as_default():

        if opts['debug']:
            tf.set_random_seed(opts['seed'])


        ## Container for student_course data (loss calculated on this table only)
        student_course = {}
        student_course['indices'] = tf.placeholder(tf.int32, shape=(None, 2), name='student_course_indices')
        student_course['values'] = tf.placeholder(tf.float32, shape=(None), name='student_course_values')
        student_course['noise_mask'] = tf.placeholder(tf.float32, shape=(None), name='student_course_noise_mask')
        student_course['values_noisy'] = tf.placeholder(tf.float32, shape=(None), name='student_course_values_noisy')
        student_course['shape'] = data.tables['student_course'].shape

        ## Container for student_prof data (no loss calculated)
        student_prof = {}
        student_prof['indices'] = tf.placeholder(tf.int32, shape=(None, 2), name='student_prof_indices')
        student_prof['values'] = tf.placeholder(tf.float32, shape=(None), name='student_prof_values')
        # student_prof['noise_mask'] = tf.placeholder(tf.float32, shape=(None), name='student_course_noise_mask') # Not needed if no loss calculated in this table
        # student_prof['values_noisy'] = tf.placeholder(tf.float32, shape=(None), name='student_course_values_noisy')
        student_prof['shape'] = data.tables['student_prof'].shape

        ## Container for course_prof data (no loss calculated)
        course_prof = {}
        course_prof['indices'] = tf.placeholder(tf.int32, shape=(None, 2), name='course_prof_indices')
        course_prof['values'] = tf.placeholder(tf.float32, shape=(None), name='course_prof_values')
        # course_prof['noise_mask'] = tf.placeholder(tf.float32, shape=(None), name='course_course_noise_mask') # Not needed if no loss calculated in this table
        # course_prof['values_noisy'] = tf.placeholder(tf.float32, shape=(None), name='course_course_values_noisy')
        course_prof['shape'] = data.tables['course_prof'].shape



        ## Encoder
        encoder_tables= {}
        encoder_tables['student_course'] = {}
        encoder_tables['student_course']['indices'] = student_course['indices']
        encoder_tables['student_course']['values'] = student_course['values_noisy']
        encoder_tables['student_course']['noise_mask'] = student_course['noise_mask'] #TODO dont need this?
        encoder_tables['student_course']['shape'] = student_course['shape']

        encoder_tables['student_prof'] = {}
        encoder_tables['student_prof']['indices'] = student_prof['indices']
        encoder_tables['student_prof']['values'] = student_prof['values']
        encoder_tables['student_prof']['shape'] = student_prof['shape']

        encoder_tables['course_prof'] = {}
        encoder_tables['course_prof']['indices'] = course_prof['indices']
        encoder_tables['course_prof']['values'] = course_prof['values']
        encoder_tables['course_prof']['shape'] = course_prof['shape']


        with tf.variable_scope('encoder'):
            encoder = Model(**opts['encoder_opts'])
            encoder_out_tr = encoder.get_output(encoder_tables)
            encoder_out_vl = encoder.get_output(encoder_tables, reuse=True, is_training=False)

        student_embeds_tr = encoder_out_tr['student_course']['row_embeds']
        course_embeds_tr = encoder_out_tr['student_course']['col_embeds']
        prof_embeds_tr = encoder_out_tr['student_prof']['col_embeds']

        student_embeds_vl = encoder_out_vl['student_course']['row_embeds']
        course_embeds_vl = encoder_out_vl['student_course']['col_embeds']
        prof_embeds_vl = encoder_out_vl['student_prof']['col_embeds']

        ## Decoder
        decoder_tables_tr = {}
        decoder_tables_tr['student_course'] = {}
        decoder_tables_tr['student_course']['indices'] = student_course['indices']
        decoder_tables_tr['student_course']['row_embeds'] = student_embeds_tr  # not passing encoder output values to decoder, just embeddings
        decoder_tables_tr['student_course']['col_embeds'] = course_embeds_tr
        decoder_tables_tr['student_course']['shape'] = student_course['shape']

        decoder_tables_tr['student_prof'] = {}
        decoder_tables_tr['student_prof']['indices'] = student_prof['indices']
        decoder_tables_tr['student_prof']['row_embeds'] = student_embeds_tr  # not passing encoder output values to decoder, just embeddings
        decoder_tables_tr['student_prof']['col_embeds'] = prof_embeds_tr
        decoder_tables_tr['student_prof']['shape'] = student_prof['shape']

        decoder_tables_tr['course_prof'] = {}
        decoder_tables_tr['course_prof']['indices'] = course_prof['indices']
        decoder_tables_tr['course_prof']['row_embeds'] = course_embeds_tr  # not passing encoder output values to decoder, just embeddings
        decoder_tables_tr['course_prof']['col_embeds'] = prof_embeds_tr
        decoder_tables_tr['course_prof']['shape'] = course_prof['shape']

        decoder_tables_vl = {}
        decoder_tables_vl['student_course'] = {}
        decoder_tables_vl['student_course']['indices'] = student_course['indices']
        decoder_tables_vl['student_course']['row_embeds'] = student_embeds_vl  # not passing encoder output values to decoder, just embeddings
        decoder_tables_vl['student_course']['col_embeds'] = course_embeds_vl
        decoder_tables_vl['student_course']['shape'] = student_course['shape']

        decoder_tables_vl['student_prof'] = {}
        decoder_tables_vl['student_prof']['indices'] = student_prof['indices']
        decoder_tables_vl['student_prof']['row_embeds'] = student_embeds_vl  # not passing encoder output values to decoder, just embeddings
        decoder_tables_vl['student_prof']['col_embeds'] = prof_embeds_vl
        decoder_tables_vl['student_prof']['shape'] = student_prof['shape']

        decoder_tables_vl['course_prof'] = {}
        decoder_tables_vl['course_prof']['indices'] = course_prof['indices']
        decoder_tables_vl['course_prof']['row_embeds'] = course_embeds_vl  # not passing encoder output values to decoder, just embeddings
        decoder_tables_vl['course_prof']['col_embeds'] = prof_embeds_vl
        decoder_tables_vl['course_prof']['shape'] = course_prof['shape']


        with tf.variable_scope('decoder'):
            decoder = Model(**opts['decoder_opts'])
            decoder_out_tr = decoder.get_output(decoder_tables_tr)
            decoder_out_vl = decoder.get_output(decoder_tables_vl, reuse=True, is_training=False)


        rec_loss_tr = rmse_loss(student_course['values'], decoder_out_tr['student_course']['values'], student_course['noise_mask'])
        rec_loss_vl = rmse_loss(student_course['values'], decoder_out_vl['student_course']['values'], student_course['noise_mask'])


        train_step = tf.train.AdamOptimizer(opts['learning_rate']).minimize(rec_loss_tr)
        # train_step = tf.train.MomentumOptimizer(opts['learning_rate'], 0.1).minimize(rec_loss_tr)
        # train_step = tf.train.RMSPropOptimizer(opts['learning_rate'], 0.99, momentum=0.9).minimize(rec_loss_tr)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())

        losses_tr = []
        losses_vl = []
        losses_ts = []
        loss_tr_best = math.inf
        loss_vl_best = math.inf
        loss_ts_vl_best = math.inf
        loss_tr_best_ep = 0
        loss_vl_best_ep = 0

        student_embeds_out_tr = []
        course_embeds_out_tr = []
        prof_embeds_out_tr = []
        student_embeds_out_vl = []
        course_embeds_out_vl = []
        prof_embeds_out_vl = []

        student_embeds_out_tr_best = []
        course_embeds_out_tr_best = []
        prof_embeds_out_tr_best = []
        student_embeds_out_vl_best = []
        course_embeds_out_vl_best = []
        prof_embeds_out_vl_best = []

        saver = tf.train.Saver()
        if restore_point is not None:
            saver.restore(sess, restore_point)
            loss_path = os.path.join(opts['checkpoints_folder'], 'loss.npz')
            loss_file = open(loss_path, 'rb')
            loss_data = np.load(loss_file)
            losses_tr = list(loss_data['losses_tr'])
            losses_vl = list(loss_data['losses_vl'])
            losses_ts = list(loss_data['losses_ts'])
            loss_tr_best = loss_data['losses_tr_best']
            loss_vl_best = loss_data['losses_vl_best']
            loss_tr_best_ep = loss_data['losses_tr_best_ep']
            loss_vl_best_ep = loss_data['losses_vl_best_ep']
            loss_ts_vl_best = loss_data['loss_ts_vl_best']
            loss_file.close()


        for ep in range(opts['restore_point_epoch'] + 1, opts['restore_point_epoch'] + opts['epochs'] + 1):
            if opts['verbosity'] > 0:
                print('------- epoch:', ep, '-------')

            ## Training
            n_tr = data.tables['student_course'].values_tr.shape[0]
            n_compute = int(n_tr * (opts['split_sizes'][0] + opts['split_sizes'][1]))
            n_predict = n_tr - n_compute

            split_tr = np.concatenate((np.zeros(n_compute, np.int32), np.ones(n_predict, np.int32)))
            np.random.shuffle(split_tr)
            vals_tr = data.tables['student_course'].values_tr * (split_tr == 0)

            tr_dict = {student_course['indices']:data.tables['student_course'].indices_tr,
                       student_course['values']:data.tables['student_course'].values_tr, # values used when calculating loss
                       student_course['noise_mask']:split_tr,
                       student_course['values_noisy']:vals_tr, # values used for making predictions
                       student_prof['indices']:data.tables['student_prof'].indices_tr,
                       student_prof['values']:data.tables['student_prof'].values_tr,
                       course_prof['indices']:data.tables['course_prof'].indices_tr,
                       course_prof['values']:data.tables['course_prof'].values_tr,
                       }

            _, loss_tr, student_embeds_out_tr, course_embeds_out_tr, prof_embeds_out_tr = sess.run([train_step,
                                                                                                    rec_loss_tr,
                                                                                                    encoder_out_tr['student_course']['row_embeds'],
                                                                                                    encoder_out_tr['student_course']['col_embeds'],
                                                                                                    encoder_out_tr['student_prof']['col_embeds']], feed_dict=tr_dict)
            losses_tr.append(loss_tr)

            if loss_tr < loss_tr_best:
                loss_tr_best = loss_tr
                loss_tr_best_ep = ep

                student_embeds_out_tr_best = student_embeds_out_tr
                course_embeds_out_tr_best = course_embeds_out_tr
                prof_embeds_out_tr_best = prof_embeds_out_tr


            ## Validation
            split_vl = data.tables['student_course'].split[data.tables['student_course'].split <= 1]
            vals_vl = data.tables['student_course'].values_tr_vl * (split_vl == 0)

            vl_dict = {student_course['indices']:data.tables['student_course'].indices_tr_vl,
                       student_course['values']:data.tables['student_course'].values_tr_vl, # values used when calculating loss
                       student_course['noise_mask']:split_vl,
                       student_course['values_noisy']:vals_vl, # values used for making predictions
                       student_prof['indices']:data.tables['student_prof'].indices_tr_vl,
                       student_prof['values']:data.tables['student_prof'].values_tr_vl,
                       course_prof['indices']:data.tables['course_prof'].indices_tr_vl,
                       course_prof['values']:data.tables['course_prof'].values_tr_vl,
                       }

            loss_vl, student_embeds_out_vl, course_embeds_out_vl, prof_embeds_out_vl = sess.run([rec_loss_vl,
                                                                             encoder_out_vl['student_course']['row_embeds'],
                                                                             encoder_out_vl['student_course']['col_embeds'],
                                                                             encoder_out_vl['student_prof']['col_embeds']], feed_dict=vl_dict)
            losses_vl.append(loss_vl)

            ## Testing
            if opts['split_sizes'][2] > 0:
                split_ts = 1. * (data.tables['student_course'].split == 2)
                vals_ts = data.tables['student_course'].values_all * (split_ts == 0)

                ts_dict = {student_course['indices']:data.tables['student_course'].indices_all,
                           student_course['values']:data.tables['student_course'].values_all,  # values used when calculating loss
                           student_course['noise_mask']:split_ts,
                           student_course['values_noisy']:vals_ts,  # values used for making predictions
                           student_prof['indices']:data.tables['student_prof'].indices_all,
                           student_prof['values']:data.tables['student_prof'].values_all,
                           course_prof['indices']:data.tables['course_prof'].indices_all,
                           course_prof['values']:data.tables['course_prof'].values_all,
                           }

                loss_ts, student_embeds_out_ts, course_embeds_out_ts, prof_embeds_out_ts = sess.run([rec_loss_vl,
                                                                                                              encoder_out_vl['student_course']['row_embeds'],
                                                                                                              encoder_out_vl['student_course']['col_embeds'],
                                                                                                              encoder_out_vl['student_prof']['col_embeds']], feed_dict=ts_dict)
            else:
                loss_ts = 0

            losses_ts.append(loss_ts)

            if loss_vl < loss_vl_best:
                # loss_improvement = (loss_vl_best - loss_vl) / loss_vl_best
                loss_vl_best = loss_vl
                loss_ts_vl_best = loss_ts
                loss_vl_best_ep = ep

                student_embeds_out_vl_best = student_embeds_out_vl
                course_embeds_out_vl_best = course_embeds_out_vl
                prof_embeds_out_vl_best = prof_embeds_out_vl



                # if loss_improvement > opts['loss_save_tolerance'] and opts['save_model']:
                if opts['save_model']:
                    model_path = os.path.join(opts['checkpoints_folder'], 'best.ckpt')
                    saver.save(sess, model_path)

                    loss_path = os.path.join(opts['checkpoints_folder'], 'loss.npz')
                    loss_file = open(loss_path, 'wb')
                    np.savez(loss_file,
                             losses_tr=losses_tr,
                             losses_vl=losses_vl,
                             losses_ts=losses_ts,
                             loss_tr_best=loss_tr_best,
                             loss_vl_best=loss_vl_best,
                             loss_ts_vl_best=loss_ts_vl_best,
                             loss_mean=loss_mean,
                             loss_tr_best_ep=loss_tr_best_ep,
                             loss_vl_best_ep=loss_vl_best_ep)
                    loss_file.close()

                    embeds_path = os.path.join(opts['checkpoints_folder'], 'embeddings_BEST.npz')
                    embeds_file = open(embeds_path, 'wb')
                    np.savez(embeds_file,
                             student_embeds_in=data.tables['student_course'].embeddings['student'],
                             course_embeds_in=data.tables['student_course'].embeddings['course'],
                             prof_embeds_in=data.tables['student_prof'].embeddings['prof'],
                             student_embeds_out_tr=student_embeds_out_tr,
                             course_embeds_out_tr=course_embeds_out_tr,
                             prof_embeds_out_tr=prof_embeds_out_tr,
                             student_embeds_out_vl=student_embeds_out_vl,
                             course_embeds_out_vl=course_embeds_out_vl,
                             prof_embeds_out_vl=prof_embeds_out_vl,
                             student_embeds_out_tr_best=student_embeds_out_tr_best,
                             course_embeds_out_tr_best=course_embeds_out_tr_best,
                             prof_embeds_out_tr_best=prof_embeds_out_tr_best,
                             student_embeds_out_vl_best=student_embeds_out_vl_best,
                             course_embeds_out_vl_best=course_embeds_out_vl_best,
                             prof_embeds_out_vl_best=prof_embeds_out_vl_best)
                    embeds_file.close()




            # if ep % opts['save_frequency'] == 0 and ep > 0 and opts['save_model']:
            #     model_path = os.path.join(opts['checkpoints_folder'], 'epoch_{:05d}'.format(ep) + '.ckpt')
            #     saver.save(sess, model_path)
            #
            #     loss_path = os.path.join(opts['checkpoints_folder'], 'loss.npz')
            #     loss_file = open(loss_path, 'wb')
            #     np.savez(loss_file,
            #              losses_tr=losses_tr,
            #              losses_vl=losses_vl,
            #              loss_tr_best=loss_tr_best,
            #              loss_vl_best=loss_vl_best,
            #              loss_tr_best_ep=loss_tr_best_ep,
            #              loss_vl_best_ep=loss_vl_best_ep)
            #     loss_file.close()
            #
            #     embeds_path = os.path.join(opts['checkpoints_folder'], 'embeddings.npz')
            #     embeds_file = open(embeds_path, 'wb')
            #     np.savez(embeds_file,
            #              student_embeds_in=data.tables['student_course'].embeddings['student'],
            #              course_embeds_in=data.tables['student_course'].embeddings['course'],
            #              prof_embeds_in=data.tables['student_prof'].embeddings['prof'],
            #              student_embeds_out_tr=student_embeds_out_tr,
            #              course_embeds_out_tr=course_embeds_out_tr,
            #              prof_embeds_out_tr=prof_embeds_out_tr,
            #              student_embeds_out_vl=student_embeds_out_vl,
            #              course_embeds_out_vl=course_embeds_out_vl,
            #              prof_embeds_out_vl=prof_embeds_out_vl,
            #              student_embeds_out_tr_best=student_embeds_out_tr_best,
            #              course_embeds_out_tr_best=course_embeds_out_tr_best,
            #              prof_embeds_out_tr_best=prof_embeds_out_tr_best,
            #              student_embeds_out_vl_best=student_embeds_out_vl_best,
            #              course_embeds_out_vl_best=course_embeds_out_vl_best,
            #              prof_embeds_out_vl_best=prof_embeds_out_vl_best)
            #     embeds_file.close()


            if opts['verbosity'] > 0:
                print("\t training loss:   {:5.5f}\t (best: {:5.5f} at epoch {:5d})".format(loss_tr, loss_tr_best, loss_tr_best_ep))
                print("\t validation loss: {:5.5f}\t (best: {:5.5f} at epoch {:5d})".format(loss_vl, loss_vl_best, loss_vl_best_ep))
                print("\t test loss:       {:5.5f}\t (at best validation: {:5.5f})".format(loss_ts, loss_ts_vl_best))
                print("\t predict mean:    {:5.5f}".format(loss_mean))



        return loss_vl_best


if __name__ == "__main__":
    # np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)

    # data_set = 'debug'
    # data_set = 'soccer'
    data_set = 'toy'

    units_in = 1
    embedding_size_data = 2
    embedding_size_network = 16
    units = 128
    units_out = 1

    # activation = tf.nn.relu
    activation = lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x) # Leaky Relu
    dropout_rate = 0.02
    skip_connections = True

    auto_restore = False
    save_model = True


    opts = {'epochs':10000,
            'data_folder':'data',
            'data_set':data_set,
            'split_sizes':[.8, .1, .1], # train, validation, test split
            'noise_rate':dropout_rate,
            'regularization_rate':.00001,
            'learning_rate':.0001,
            'toy_data':{'size':[200, 100, 50],
                        'sparsity':.1,
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
            'restore_point_epoch':-1,
            'save_model':save_model,
            'save_frequency':50, # Save model every save_frequency epochs
            'loss_save_tolerance':.01, # If loss changes by more than loss_save_tolerance (as % of old value), save the model
            'debug':True, # Set random seeds or not
            'seed':9858776,
            # 'seed': 9870112,
            }


    restore_point = None
    if auto_restore:         
        restore_point_epoch = sorted(glob.glob(opts['checkpoints_folder'] + "/epoch_*.ckpt*"))[-1].split(".")[0].split("_")[-1]
        restore_point = opts['checkpoints_folder'] + "/epoch_" + restore_point_epoch + ".ckpt"
        opts['restore_point_epoch'] = int(restore_point_epoch)

    main(opts, restore_point)






