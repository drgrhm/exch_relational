import numpy as np
import tensorflow as tf
# import matplotlib
# matplotlib.use('Agg')
import os
# import matplotlib.pyplot as plt
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

    loss_mean = 0
    if opts['split_sizes'] is None or opts['split_sizes'][2] == 0:
        if opts['calculate_loss'][0]:
            mean_sc = np.mean(data.tables['student_course'].values_tr)
            split_sc = data.tables['student_course'].split
            loss_mean += np_rmse_loss(data.tables['student_course'].values_tr_vl, mean_sc * np.ones_like(data.tables['student_course'].values_tr_vl), 1. * (split_sc == 1))  # Loss on validation set when predicting training mean
        if opts['calculate_loss'][1]:
            mean_sp = np.mean(data.tables['student_prof'].values_tr)
            split_sp = data.tables['student_prof'].split
            loss_mean += np_rmse_loss(data.tables['student_prof'].values_tr_vl, mean_sp * np.ones_like(data.tables['student_prof'].values_tr_vl), 1. * (split_sp == 1))  # Loss on validation set when predicting training mean
        if opts['calculate_loss'][2]:
            mean_cp = np.mean(data.tables['course_prof'].values_tr)
            split_cp = data.tables['course_prof'].split
            loss_mean += np_rmse_loss(data.tables['course_prof'].values_tr_vl, mean_cp * np.ones_like(data.tables['course_prof'].values_tr_vl), 1. * (split_cp == 1))  # Loss on validation set when predicting training mean
    else:
        if opts['calculate_loss'][0]:
            mean_sc = np.mean(data.tables['student_course'].values_tr_vl)
            split_sc = data.tables['student_course'].split
            loss_mean += np_rmse_loss(data.tables['student_course'].values_all, mean_sc * np.ones_like(data.tables['student_course'].values_all), 1. * (split_sc == 2))  # Loss on test set when predicting training/validation mean
        if opts['calculate_loss'][1]:
            mean_sp = np.mean(data.tables['student_prof'].values_tr_vl)
            split_sp = data.tables['student_prof'].split
            loss_mean += np_rmse_loss(data.tables['student_prof'].values_all, mean_sp * np.ones_like(data.tables['student_prof'].values_all), 1. * (split_sp == 2))  # Loss on test set when predicting training/validation mean
        if opts['calculate_loss'][2]:
            mean_cp = np.mean(data.tables['course_prof'].values_tr_vl)
            split_cp = data.tables['course_prof'].split
            loss_mean += np_rmse_loss(data.tables['course_prof'].values_all, mean_cp * np.ones_like(data.tables['course_prof'].values_all), 1. * (split_cp == 2))  # Loss on test set when predicting training/validation mean


    with tf.Graph().as_default():

        if opts['debug']:
            tf.set_random_seed(opts['seed'])


        ## Container for student_course data
        student_course = {}
        student_course['indices'] = tf.placeholder(tf.int32, shape=(None, 2), name='student_course_indices')
        student_course['values'] = tf.placeholder(tf.float32, shape=(None), name='student_course_values')
        student_course['noise_mask'] = tf.placeholder(tf.float32, shape=(None), name='student_course_noise_mask')
        student_course['values_noisy'] = tf.placeholder(tf.float32, shape=(None), name='student_course_values_noisy')
        student_course['shape'] = data.tables['student_course'].shape
        student_course['calculate_loss'] = opts['calculate_loss'][0]

        ## Container for student_prof data
        student_prof = {}
        student_prof['indices'] = tf.placeholder(tf.int32, shape=(None, 2), name='student_prof_indices')
        student_prof['values'] = tf.placeholder(tf.float32, shape=(None), name='student_prof_values')
        student_prof['noise_mask'] = tf.placeholder(tf.float32, shape=(None), name='student_prof_noise_mask') # Not needed if no loss calculated in this table
        student_prof['values_noisy'] = tf.placeholder(tf.float32, shape=(None), name='student_prof_values_noisy')
        student_prof['shape'] = data.tables['student_prof'].shape
        student_prof['calculate_loss'] = opts['calculate_loss'][1]

        ## Container for course_prof data
        course_prof = {}
        course_prof['indices'] = tf.placeholder(tf.int32, shape=(None, 2), name='course_prof_indices')
        course_prof['values'] = tf.placeholder(tf.float32, shape=(None), name='course_prof_values')
        course_prof['noise_mask'] = tf.placeholder(tf.float32, shape=(None), name='course_prof_noise_mask') # Not needed if no loss calculated in this table
        course_prof['values_noisy'] = tf.placeholder(tf.float32, shape=(None), name='course_prof_values_noisy')
        course_prof['shape'] = data.tables['course_prof'].shape
        course_prof['calculate_loss'] = opts['calculate_loss'][2]


        ## Encoder
        encoder_tables= {}
        encoder_tables['student_course'] = {}
        encoder_tables['student_course']['indices'] = student_course['indices']
        encoder_tables['student_course']['values'] = student_course['values_noisy']
        encoder_tables['student_course']['noise_mask'] = student_course['noise_mask']
        encoder_tables['student_course']['shape'] = student_course['shape']

        encoder_tables['student_prof'] = {}
        encoder_tables['student_prof']['indices'] = student_prof['indices']
        encoder_tables['student_prof']['values'] = student_prof['values']
        encoder_tables['student_prof']['noise_mask'] = student_prof['noise_mask']
        encoder_tables['student_prof']['shape'] = student_prof['shape']

        encoder_tables['course_prof'] = {}
        encoder_tables['course_prof']['indices'] = course_prof['indices']
        encoder_tables['course_prof']['values'] = course_prof['values']
        encoder_tables['course_prof']['noise_mask'] = course_prof['noise_mask']
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

        rec_loss_tr = 0
        rec_loss_vl = 0

        if student_course['calculate_loss']:
            rec_loss_tr += rmse_loss(student_course['values'], decoder_out_tr['student_course']['values'], student_course['noise_mask'])
            rec_loss_vl += rmse_loss(student_course['values'], decoder_out_vl['student_course']['values'], student_course['noise_mask'])
        if student_prof['calculate_loss']:
            rec_loss_tr += rmse_loss(student_prof['values'], decoder_out_tr['student_prof']['values'], student_prof['noise_mask'])
            rec_loss_vl += rmse_loss(student_prof['values'], decoder_out_vl['student_prof']['values'], student_prof['noise_mask'])
        if course_prof['calculate_loss']:
            rec_loss_tr += rmse_loss(course_prof['values'], decoder_out_tr['course_prof']['values'], course_prof['noise_mask'])
            rec_loss_vl += rmse_loss(course_prof['values'], decoder_out_vl['course_prof']['values'], course_prof['noise_mask'])

        train_step = tf.train.AdamOptimizer(opts['learning_rate']).minimize(rec_loss_tr)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())

        losses_tr = []
        losses_vl = []
        losses_ts = []
        loss_tr_best = math.inf
        loss_vl_best = math.inf
        loss_ts_vl_best = math.inf
        loss_vl_last_save = math.inf
        loss_improvement = math.inf
        loss_tr_best_ep = 0
        loss_vl_best_ep = 0

        student_embeds_out_vl_best = []
        course_embeds_out_vl_best = []
        prof_embeds_out_vl_best = []
        student_embeds_out_tr_best = []
        course_embeds_out_tr_best = []
        prof_embeds_out_tr_best = []

        saver = tf.train.Saver()
        if restore_point is not None:
            saver.restore(sess, restore_point)
            loss_path = os.path.join(opts['checkpoints_folder'], 'loss.npz')
            loss_file = open(loss_path, 'rb')
            loss_data = np.load(loss_file)
            losses_tr = list(loss_data['losses_tr'])
            losses_vl = list(loss_data['losses_vl'])
            losses_ts = list(loss_data['losses_ts'])
            loss_tr_best = loss_data['loss_tr_best']
            loss_vl_best = loss_data['loss_vl_best']
            loss_tr_best_ep = loss_data['loss_tr_best_ep']
            loss_vl_best_ep = loss_data['loss_vl_best_ep']
            loss_ts_vl_best = loss_data['loss_ts_vl_best']
            loss_file.close()

        ## Evaluation only
        if opts['evaluate_only']:

            if student_course['calculate_loss']:
                split_eval_sc = 1. * (data.tables['student_course'].split == 1)
            else:
                split_eval_sc = np.zeros_like(data.tables['student_course'].split == 1)
            vals_eval_sc = data.tables['student_course'].values_all * (split_eval_sc == 0)

            if student_prof['calculate_loss']:
                split_eval_sp = 1. * (data.tables['student_prof'].split == 1)
            else:
                split_eval_sp = np.zeros_like(data.tables['student_prof'].split == 1)
            vals_eval_sp = data.tables['student_prof'].values_all * (split_eval_sp == 0)

            if course_prof['calculate_loss']:
                split_eval_cp = 1. * (data.tables['course_prof'].split == 1)
            else:
                split_eval_cp = np.zeros_like(data.tables['course_prof'].split == 1)
            vals_eval_cp = data.tables['course_prof'].values_all * (split_eval_cp == 0)


            eval_dict = {student_course['indices']:data.tables['student_course'].indices_all,
                         student_course['values']:data.tables['student_course'].values_all,  # values used when calculating loss
                         student_course['noise_mask']:split_eval_sc,
                         student_course['values_noisy']:vals_eval_sc,  # values used for making predictions

                         student_prof['indices']:data.tables['student_prof'].indices_all,
                         student_prof['values']:data.tables['student_prof'].values_all,
                         student_prof['noise_mask']:split_eval_sp,
                         student_prof['values_noisy']:vals_eval_sp,

                         course_prof['indices']:data.tables['course_prof'].indices_all,
                         course_prof['values']:data.tables['course_prof'].values_all,
                         course_prof['noise_mask']:split_eval_cp,
                         course_prof['values_noisy']:vals_eval_cp,
                         }

            loss_eval, student_embeds_out_eval, course_embeds_out_eval, prof_embeds_out_eval = sess.run([rec_loss_vl,
                                                                                                         student_embeds_vl,
                                                                                                         course_embeds_vl,
                                                                                                         prof_embeds_vl], feed_dict=eval_dict)
            return loss_eval, loss_mean

        if student_course['calculate_loss']:
            split_vl_sc = data.tables['student_course'].split[data.tables['student_course'].split <= 1]
        else:
            split_vl_sc = np.zeros_like(data.tables['student_course'].split[data.tables['student_course'].split <= 1])
        vals_vl_sc = data.tables['student_course'].values_tr_vl * (split_vl_sc == 0)

        if student_prof['calculate_loss']:
            split_vl_sp = data.tables['student_prof'].split[data.tables['student_prof'].split <= 1]
        else:
            split_vl_sp = np.zeros_like(data.tables['student_prof'].split[data.tables['student_prof'].split <= 1])
        vals_vl_sp = data.tables['student_prof'].values_tr_vl * (split_vl_sp == 0)

        if course_prof['calculate_loss']:
            split_vl_cp = data.tables['course_prof'].split[data.tables['course_prof'].split <= 1]
        else:
            split_vl_cp = np.zeros_like(data.tables['course_prof'].split[data.tables['course_prof'].split <= 1])
        vals_vl_cp = data.tables['course_prof'].values_tr_vl * (split_vl_cp == 0)

        vl_dict = {student_course['indices']:data.tables['student_course'].indices_tr_vl,
                   student_course['values']:data.tables['student_course'].values_tr_vl,  # values used when calculating loss
                   student_course['noise_mask']:split_vl_sc,
                   student_course['values_noisy']:vals_vl_sc,  # values used for making predictions

                   student_prof['indices']:data.tables['student_prof'].indices_tr_vl,
                   student_prof['values']:data.tables['student_prof'].values_tr_vl,
                   student_prof['noise_mask']:split_vl_sp,
                   student_prof['values_noisy']:vals_vl_sp,

                   course_prof['indices']:data.tables['course_prof'].indices_tr_vl,
                   course_prof['values']:data.tables['course_prof'].values_tr_vl,
                   course_prof['noise_mask']:split_vl_cp,
                   course_prof['values_noisy']:vals_vl_cp,
                   }

        student_embeds_init, course_embeds_init, prof_embeds_init, = sess.run([encoder_out_vl['student_course']['row_embeds'],
                                                                               encoder_out_vl['student_course']['col_embeds'],
                                                                               encoder_out_vl['student_prof']['col_embeds']], feed_dict=vl_dict)

        # student_embeds_init, course_embeds_init, prof_embeds_init = 0,0,0

        for ep in range(opts['restore_point_epoch'] + 1, opts['restore_point_epoch'] + opts['epochs'] + 1):
            if opts['verbosity'] > 0:
                print('------- epoch:', ep, '-------')

            ## Training
            n_tr_sc = data.tables['student_course'].values_tr.shape[0]
            if student_course['calculate_loss']:
                n_compute_sc = int(n_tr_sc * (opts['split_sizes'][0]))
                n_predict_sc = n_tr_sc - n_compute_sc
            else:
                n_compute_sc = n_tr_sc
                n_predict_sc = 0
            split_tr_sc = np.concatenate((np.zeros(n_compute_sc, np.int32), np.ones(n_predict_sc, np.int32)))
            np.random.shuffle(split_tr_sc)
            vals_tr_sc = data.tables['student_course'].values_tr * (split_tr_sc == 0)

            n_tr_sp = data.tables['student_prof'].values_tr.shape[0]
            if student_prof['calculate_loss']:
                n_compute_sp = int(n_tr_sp * (opts['split_sizes'][0]))
                n_predict_sp = n_tr_sp - n_compute_sp
            else:
                n_compute_sp = n_tr_sp
                n_predict_sp = 0
            split_tr_sp = np.concatenate((np.zeros(n_compute_sp, np.int32), np.ones(n_predict_sp, np.int32)))
            np.random.shuffle(split_tr_sp)
            vals_tr_sp = data.tables['student_prof'].values_tr * (split_tr_sp == 0)

            n_tr_cp = data.tables['course_prof'].values_tr.shape[0]
            if course_prof['calculate_loss']:
                n_compute_cp = int(n_tr_cp * (opts['split_sizes'][0]))
                n_predict_cp = n_tr_cp - n_compute_cp
            else:
                n_compute_cp = n_tr_cp
                n_predict_cp = 0
            split_tr_cp = np.concatenate((np.zeros(n_compute_cp, np.int32), np.ones(n_predict_cp, np.int32)))
            np.random.shuffle(split_tr_cp)
            vals_tr_cp = data.tables['course_prof'].values_tr * (split_tr_cp == 0)

            tr_dict = {student_course['indices']:data.tables['student_course'].indices_tr,
                       student_course['values']:data.tables['student_course'].values_tr, # values used when calculating loss
                       student_course['noise_mask']:split_tr_sc,
                       student_course['values_noisy']:vals_tr_sc, # values used for making predictions
                       student_prof['indices']:data.tables['student_prof'].indices_tr,
                       student_prof['values']:data.tables['student_prof'].values_tr,
                       student_prof['noise_mask']:split_tr_sp,
                       student_prof['values_noisy']:vals_tr_sp,
                       course_prof['indices']:data.tables['course_prof'].indices_tr,
                       course_prof['values']:data.tables['course_prof'].values_tr,
                       course_prof['noise_mask']:split_tr_cp,
                       course_prof['values_noisy']:vals_tr_cp,  # values used for making predictions
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
            if student_course['calculate_loss']:
                split_vl_sc = data.tables['student_course'].split[data.tables['student_course'].split <= 1]
            else:
                split_vl_sc = np.zeros_like(data.tables['student_course'].split[data.tables['student_course'].split <= 1])
            vals_vl_sc = data.tables['student_course'].values_tr_vl * (split_vl_sc == 0)

            if student_prof['calculate_loss']:
                split_vl_sp = data.tables['student_prof'].split[data.tables['student_prof'].split <= 1]
            else:
                split_vl_sp = np.zeros_like(data.tables['student_prof'].split[data.tables['student_prof'].split <= 1])
            vals_vl_sp = data.tables['student_prof'].values_tr_vl * (split_vl_sp == 0)

            if course_prof['calculate_loss']:
                split_vl_cp = data.tables['course_prof'].split[data.tables['course_prof'].split <= 1]
            else:
                split_vl_cp = np.zeros_like(data.tables['course_prof'].split[data.tables['course_prof'].split <= 1])
            vals_vl_cp = data.tables['course_prof'].values_tr_vl * (split_vl_cp == 0)

            vl_dict = {student_course['indices']:data.tables['student_course'].indices_tr_vl,
                       student_course['values']:data.tables['student_course'].values_tr_vl, # values used when calculating loss
                       student_course['noise_mask']:split_vl_sc,
                       student_course['values_noisy']:vals_vl_sc, # values used for making predictions
                       student_prof['indices']:data.tables['student_prof'].indices_tr_vl,
                       student_prof['values']:data.tables['student_prof'].values_tr_vl,
                       student_prof['noise_mask']:split_vl_sp,
                       student_prof['values_noisy']:vals_vl_sp,  # values used for making predictions
                       course_prof['indices']:data.tables['course_prof'].indices_tr_vl,
                       course_prof['values']:data.tables['course_prof'].values_tr_vl,
                       course_prof['noise_mask']:split_vl_cp,
                       course_prof['values_noisy']:vals_vl_cp,
                       }

            loss_vl, student_embeds_out_vl, course_embeds_out_vl, prof_embeds_out_vl, = sess.run([rec_loss_vl,
                                                                                         encoder_out_vl['student_course']['row_embeds'],
                                                                                         encoder_out_vl['student_course']['col_embeds'],
                                                                                         encoder_out_vl['student_prof']['col_embeds']], feed_dict=vl_dict)
                                                                                         # encoder_out_vl['student_course']['row_embeds_init'],
                                                                                         # encoder_out_vl['student_course']['col_embeds_init'],
                                                                                         # encoder_out_vl['student_prof']['col_embeds_init']], feed_dict=vl_dict)
            losses_vl.append(loss_vl)

            ## Testing
            if opts['split_sizes'][2] > 0:
                if student_course['calculate_loss']:
                    split_ts_sc = 1. * (data.tables['student_course'].split == 2)
                else:
                    split_ts_sc = np.zeros_like(data.tables['student_course'].split == 2)
                vals_ts_sc = data.tables['student_course'].values_all * (split_ts_sc == 0)

                if student_prof['calculate_loss']:
                    split_ts_sp = 1. * (data.tables['student_prof'].split == 2)
                else:
                    split_ts_sp = np.zeros_like(data.tables['student_prof'].split == 2)
                vals_ts_sp = data.tables['student_prof'].values_all * (split_ts_sp == 0)

                if course_prof['calculate_loss']:
                    split_ts_cp = 1. * (data.tables['course_prof'].split == 2)
                else:
                    split_ts_cp = np.zeros_like(data.tables['course_prof'].split == 2)
                vals_ts_cp = data.tables['course_prof'].values_all * (split_ts_cp == 0)

                ts_dict = {student_course['indices']:data.tables['student_course'].indices_all,
                           student_course['values']:data.tables['student_course'].values_all,  # values used when calculating loss
                           student_course['noise_mask']:split_ts_sc,
                           student_course['values_noisy']:vals_ts_sc,  # values used for making predictions
                           student_prof['indices']:data.tables['student_prof'].indices_all,
                           student_prof['values']:data.tables['student_prof'].values_all,
                           student_prof['noise_mask']:split_ts_sp,
                           student_prof['values_noisy']:vals_ts_sp,
                           course_prof['indices']:data.tables['course_prof'].indices_all,
                           course_prof['values']:data.tables['course_prof'].values_all,
                           course_prof['noise_mask']:split_ts_cp,
                           course_prof['values_noisy']:vals_ts_cp,
                           }

                loss_ts, student_embeds_out_ts, course_embeds_out_ts, prof_embeds_out_ts = sess.run([rec_loss_vl,
                                                                                                              encoder_out_vl['student_course']['row_embeds'],
                                                                                                              encoder_out_vl['student_course']['col_embeds'],
                                                                                                              encoder_out_vl['student_prof']['col_embeds']], feed_dict=ts_dict)
            else:
                loss_ts = 0

            losses_ts.append(loss_ts)

            if loss_vl < loss_vl_best:
                if loss_vl_last_save < math.inf:
                    loss_improvement = (loss_vl_last_save - loss_vl) / loss_vl_last_save
                loss_vl_best = loss_vl
                loss_ts_vl_best = loss_ts
                loss_vl_best_ep = ep

                student_embeds_out_vl_best = student_embeds_out_vl
                course_embeds_out_vl_best = course_embeds_out_vl
                prof_embeds_out_vl_best = prof_embeds_out_vl

                ## save model when validation loss improves
                if opts['save_model'] and loss_improvement > opts['loss_save_improvement']:
                    loss_vl_last_save = loss_vl
                    print("-------  SAVING MODEL (loss improvement: {:f}) ------- ".format(loss_improvement))
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

                    embeds_path = os.path.join(opts['checkpoints_folder'], 'embeddings_vl_best.npz')
                    embeds_file = open(embeds_path, 'wb')
                    np.savez(embeds_file,
                             student_embeds_in=data.tables['student_course'].embeddings['student'],
                             course_embeds_in=data.tables['student_course'].embeddings['course'],
                             prof_embeds_in=data.tables['student_prof'].embeddings['prof'],
                             student_embeds_out_vl_best=student_embeds_out_vl_best,
                             course_embeds_out_vl_best=course_embeds_out_vl_best,
                             prof_embeds_out_vl_best=prof_embeds_out_vl_best,
                             student_embeds_init=student_embeds_init,
                             course_embeds_init=course_embeds_init,
                             prof_embeds_init=prof_embeds_init)
                    embeds_file.close()

            ## save EMBEDDINGS ONLY periodically.
            if ep % opts['save_frequency'] == 0 and opts['save_model']:
                # model_path = os.path.join(opts['checkpoints_folder'], 'ep_{:05d}.ckpt'.format(ep))
                # saver.save(sess, model_path)
                #
                # loss_path = os.path.join(opts['checkpoints_folder'], 'loss_{:05d}.npz'.format(ep))
                # loss_file = open(loss_path, 'wb')
                # np.savez(loss_file,
                #          losses_tr=losses_tr,
                #          losses_vl=losses_vl,
                #          losses_ts=losses_ts,
                #          loss_tr_best=loss_tr_best,
                #          loss_vl_best=loss_vl_best,
                #          loss_ts_vl_best=loss_ts_vl_best,
                #          loss_mean=loss_mean,
                #          loss_tr_best_ep=loss_tr_best_ep,
                #          loss_vl_best_ep=loss_vl_best_ep)

                embeds_path = os.path.join(opts['checkpoints_folder'], 'embeddings_{:05d}.npz'.format(ep))
                embeds_file = open(embeds_path, 'wb')
                np.savez(embeds_file,
                         student_embeds_in=data.tables['student_course'].embeddings['student'],
                         course_embeds_in=data.tables['student_course'].embeddings['course'],
                         prof_embeds_in=data.tables['student_prof'].embeddings['prof'],
                         student_embeds_out_tr_best=student_embeds_out_tr_best,
                         course_embeds_out_tr_best=course_embeds_out_tr_best,
                         prof_embeds_out_tr_best=prof_embeds_out_tr_best,
                         student_embeds_out_vl_best=student_embeds_out_vl_best,
                         course_embeds_out_vl_best=course_embeds_out_vl_best,
                         prof_embeds_out_vl_best=prof_embeds_out_vl_best,
                         student_embeds_init=student_embeds_init,
                         course_embeds_init=course_embeds_init,
                         prof_embeds_init=prof_embeds_init)
                embeds_file.close()

            if opts['verbosity'] > 0:
                print("\t training loss:   {:5.5f}\t (best: {:5.5f} at epoch {:5d})".format(loss_tr, loss_tr_best, loss_tr_best_ep))
                print("\t validation loss: {:5.5f}\t (best: {:5.5f} at epoch {:5d})".format(loss_vl, loss_vl_best, loss_vl_best_ep))
                print("\t test loss:       {:5.5f}\t (at best validation: {:5.5f})".format(loss_ts, loss_ts_vl_best))
                print("\t predict mean:    {:5.5f}".format(loss_mean))

        # return loss_vl_best, loss_mean
        return loss_ts_vl_best, loss_mean


if __name__ == "__main__":
    # np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)

    # data_set = 'debug'
    # data_set = 'soccer'
    data_set = 'toy'

    units_in = 1
    embedding_size_data = 2
    embedding_size_network = 2
    units = 64
    units_out = 1

    # activation = tf.nn.relu
    activation = lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x) # Leaky Relu
    dropout_rate = 0.02
    skip_connections = True

    auto_restore = False
    save_model = False

    opts = {'epochs':50000,
            'data_folder':'data',
            'data_set':data_set,
            'split_sizes':[.8, .1, .1], # train, validation, test split
            'noise_rate':dropout_rate,
            'regularization_rate':.00001,
            'learning_rate':.0001,
            'evaluate_only':False,  # If True, don't train the model, just evaluate it
            'calculate_loss':[True, True, True], # Which tables do we calculate loss on.
            'toy_data':{'size':[200, 200, 200],
                        'sparsity':.1,
                        'embedding_size':embedding_size_data,
                        'min_observed':5, # generate at least 2 entries per row and column (sparsity rate will be affected)
            },
            'encoder_opts':{'pool_mode':'mean',
                          'dropout_rate':dropout_rate,
                          'units_in':units_in,
                          'units_out':units_out,
                          'layers':[
                                    {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'save_embeddings':True},
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
                                    # # {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                    # # {'type':FeatureDropoutLayer, 'units_out':units},
                                    # # {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                    # # {'type':FeatureDropoutLayer, 'units_out':units},
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
            'save_frequency':5000000, # Save model every save_frequency epochs
            'loss_save_improvement':.005, # If loss changes by more than loss_save_tolerance (as % of old value), save the model
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






