import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import math
from copy import deepcopy
from util import *
from data_util import DataLoader, ToyDataLoader
from model import Model
from layers import ExchangeableLayer, FeatureDropoutLayer


# ## Noise mask has 0's corresponding to values to be predicted
# def table_rmse_loss(values, values_out, noise_mask):
#     prediction_mask = tf.cast(1 - noise_mask, tf.float32)
#     return tf.sqrt( tf.reduce_sum(((values - values_out)**2)*prediction_mask) / (tf.reduce_sum(prediction_mask) + 1e-10) )
#
#
# def table_cross_entropy_loss(values, values_out, noise_mask, num_features):
#     prediction_mask = tf.cast(1 - noise_mask, tf.float32)
#     vals = tf.reshape(prediction_mask * values, shape=[-1,num_features])
#     out = tf.reshape(prediction_mask * values_out, shape=[-1,num_features])
#     return - tf.reduce_mean(tf.reduce_sum(vals * (out - tf.reduce_logsumexp(out, axis=1, keep_dims=True)), axis=1))
    
#
#
# def table_ordinal_hinge_loss(values, values_out, noise_mask):
#     prediction_mask = tf.cast(1 - noise_mask, tf.float32)
#     vals = tf.reshape(values, [-1,d])
#     ones = tf.ones_like(vals)
#     preds = tf.cast(tf.where(tf.equal(vals, ones))[:,1][:,None], tf.float32)
#     preds = preds + tf.ones_like(preds)
#
#     vals_out = tf.reshape(values_out, [-1,d])
#     preds_out = tf.cast(tf.where(tf.equal(vals_out, ones))[:,1][:,None], tf.float32)
#     preds_out = preds_out + tf.ones_like(preds_out)
#
#     num_vals = tf.shape(vals)[0]
#     categories = tf.cast(tf.reshape(tf.tile(tf.range(1,d+1), [num_vals]), [-1,d]), tf.float32)
#
#     greater = tf.cast(tf.greater_equal(categories, preds), tf.float32)
#     less = tf.cast(tf.less_equal(categories, preds), tf.float32)
#     not_equal = tf.cast(tf.not_equal(categories, preds), tf.float32)
#
#     preds_out = preds_out * (greater - less)
#     preds_out = (preds_out + 1) * not_equal
#     out = categories * (less - greater) + preds_out
#     out = tf.maximum(out, tf.zeros_like(out))
#     out = tf.reduce_sum(out, axis=1) * prediction_mask
#
#     return tf.reduce_sum(out)


# def one_hot_prediction_accuracy(values, values_out, noise_mask, num_features):
#
#     vals = np.reshape(values[noise_mask == 0], [-1, num_features])
#     vals_out = np.reshape(values_out[noise_mask == 0], [-1, num_features])
#
#     num_vals = vals.shape[0]
#
#     probs_out = np.exp(vals_out - np.max(vals_out, axis=1)[:,None])
#     probs_out = probs_out / np.sum(probs_out, axis=1)[:,None]
#
#     preds = np.zeros_like(vals)
#     max_inds = np.argmax(vals_out, axis=1)
#
#     preds[np.arange(num_vals), max_inds] = 1
#
#     # print(vals[20:40])
#     # print('')
#     # print(preds[20:40])
#
#     print('input:  ', np.mean(vals, axis=0))
#     print('output: ', np.mean(preds, axis=0))
#
#     return np.sum(preds*vals) / num_vals


# def make_uniform_noise_mask(noise_rate, num_vals):
#     """A 0/1 noise mask. 0's correspond to dropped out values."""
#     n0 = int(noise_rate * num_vals)
#     n1 = num_vals - n0
#     noise_mask = np.concatenate((np.zeros(n0), np.ones(n1)))
#     np.random.shuffle(noise_mask)
#     return noise_mask
#
# def make_by_col_noise(noise_rate, num_vals, shape, column_indices):
#     """A 0/1 noise mask. 0's correspond to dropped out values. Drop out columns."""
#     num_cols = shape[1]
#     n0 = int(noise_rate * num_cols)
#     n1 = num_cols - n0
#     column_mask = np.concatenate((np.zeros(n0), np.ones(n1)))
#     np.random.shuffle(column_mask)
#     noise_mask = np.take(column_mask, column_indices)
#     return noise_mask

# def make_noisy_values(values, noise_rate, noise_value):
#     """replace noise_rate fraction of values with noise_value."""
#     num_vals = values.shape[0]
#     noise = make_noise_mask(noise_rate, num_vals) 
#     values_noisy = np.copy(values)
#     values_noisy[noise == 0] = noise_value
#     return noise, values_noisy




# def embedding_rmse_loss(embeddings_in, embeddings_out):
#     e_in = tf.cast(tf.reshape(embeddings_in, [-1]), tf.float32)
#     e_out = tf.cast(tf.reshape(embeddings_out, [-1]), tf.float32)
#     n_embeds = tf.cast(tf.shape(e_in), tf.float32)
#     return tf.sqrt( tf.reduce_sum((e_in - e_out)**2) / n_embeds )

def embedding_se_loss(embeddings_in, embeddings_out):
    e_in = tf.cast(tf.reshape(embeddings_in, [-1]), tf.float32)
    e_out = tf.cast(tf.reshape(embeddings_out, [-1]), tf.float32)
    n_embeds = tf.cast(tf.shape(e_in), tf.float32)
    return tf.reduce_sum((e_in - e_out)**2)

# def embedding_cossim_loss(embeddings_in, embeddings_out):
#     e_in = tf.cast(embeddings_in, tf.float32)
#     e_out = tf.cast(embeddings_out, tf.float32)
#     dot = tf.reduce_sum(tf.multiply(e_in, e_out), 1)
#     norm_in = tf.sqrt(tf.reduce_sum(e_in ** 2, 1))
#     norm_out = tf.sqrt(tf.reduce_sum(e_out ** 2, 1))
#     return -tf.reduce_sum(dot / (norm_in * norm_out))
#     # return tf.reduce_sum((norm_in * norm_out) / dot)
#     return tf.log(norm_in * norm_out) - tf.log(norm_in * norm_out + dot)


def main(opts, restore_point=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    if opts['debug']:
        np.random.seed(opts['seed'])

    # data = DataLoader(opts['data_folder'], opts['data_set'], opts['split_sizes'], opts['model_opts']['units_in'], opts['team_match_one_hot'])
    alphas = 2 * np.random.randn(4)
    # data_tr = ToyDataLoader(opts['toy_data']['size_tr'], opts['model_opts']['units_in'], opts['toy_data']['sparsity'], alphas=alphas)
    data_tr = ToyDataLoader(opts['toy_data']['size_tr'], 2, opts['toy_data']['sparsity'], alphas=alphas)
    data_vl = ToyDataLoader(opts['toy_data']['size_vl'], 2, opts['toy_data']['sparsity'], alphas=alphas)


    with tf.Graph().as_default():

        if opts['debug']:
            tf.set_random_seed(opts['seed'])

        model = Model(**opts['model_opts'])


        student_course_tr = {}
        student_course_tr['indices'] = tf.placeholder(tf.int32, shape=(None, 2), name='student_course_indices_tr')
        student_course_tr['values'] = tf.placeholder(tf.float32, shape=(None), name='student_course_values_noisy_tr')
        student_course_tr['shape'] = data_tr.tables['student_course'].shape

        student_course_vl = {}
        student_course_vl['indices'] = tf.placeholder(tf.int32, shape=(None, 2), name='student_course_indices_vl')
        student_course_vl['values'] = tf.placeholder(tf.float32, shape=(None), name='student_course_values_noisy_vl')
        student_course_vl['shape'] = data_vl.tables['student_course'].shape

        # student_course_noise_mask = tf.placeholder(tf.float32, shape=(None), name='student_course_noise_mask')
        # student_course_values = tf.placeholder(tf.float32, shape=(None), name='student_course_values')


        tables_in_tr = {}
        tables_in_tr['student_course'] = student_course_tr

        tables_in_vl = {}
        tables_in_vl['student_course'] = student_course_vl

        tables_out_tr = model.get_output(tables_in_tr)
        tables_out_vl = model.get_output(tables_in_vl, reuse=True, is_training=False)


        rec_loss_tr = 0
        rec_loss_tr += embedding_se_loss(data_tr.tables['student_course'].embeddings['student'], tables_out_tr['student_course']['row_embeds'])
        # rec_loss_tr += embedding_se_loss(data_tr.tables['student_course'].embeddings['course'], tables_out_tr['student_course']['col_embeds'])
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss_tr = rec_loss_tr + opts['regularization_rate']*reg_loss

        rec_loss_vl = 0
        rec_loss_vl += embedding_se_loss(data_vl.tables['student_course'].embeddings['student'], tables_out_vl['student_course']['row_embeds'])
        total_loss_vl = rec_loss_vl


        train_step = tf.train.AdamOptimizer(opts['learning_rate']).minimize(total_loss_tr)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())

    #     saver = tf.train.Saver()
    #     if restore_point is not None:
    #         saver.restore(sess, restore_point)

        # if opts['model_opts']['pool_mode'] == 'mean':
        #     noise_value = 0
        # if opts['model_opts']['pool_mode'] == 'max':
        #     noise_value = -1e10

        losses_tr = []
        losses_vl = []
    #     losses_vl_baseline = []
    #
    #     accuracies_tr = []
    #     accuracies_vl = []
    #     accuracies_vl_baseline = []
    #
    #     accuracy_vl_best = 0
    #     accuracy_vl_best_ep = 0
    #
    #
        for ep in range(opts['restore_point_epoch'] + 1, opts['restore_point_epoch'] + opts['epochs'] + 1):
            if opts['verbosity'] > 0:
                print('------- epoch:', ep, '-------')


                tr_dict = {student_course_tr['indices']:data_tr.tables['student_course'].indices,
                           student_course_tr['values']: data_tr.tables['student_course'].values}


                _, loss_tr, row_embeds_out_tr, col_embeds_out_tr = sess.run([train_step, total_loss_tr, tables_out_tr['student_course']['row_embeds'], tables_out_tr['student_course']['col_embeds']], feed_dict=tr_dict)
                losses_tr.append(loss_tr)


                vl_dict = {student_course_vl['indices']:data_vl.tables['student_course'].indices,
                           student_course_vl['values']: data_vl.tables['student_course'].values}

                loss_vl, row_embeds_out_vl, col_embeds_out_vl = sess.run([total_loss_vl, tables_out_vl['student_course']['row_embeds'], tables_out_vl['student_course']['col_embeds']], feed_dict=vl_dict)
                losses_vl.append(loss_vl)



        plt.title('Loss')
        plt.plot(range(opts['epochs']), losses_tr, '.-', color='blue')
        plt.plot(range(opts['epochs']), losses_vl, '.-', color='green')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(('training', 'validation'))
        plt.show()
        plt.savefig("img/loss.pdf", bbox_inches='tight')
        plt.clf()

        # plt.title('Features')
        #
        # s = [5 * math.log(1 + i) for i in range(opts['epochs'])]
        # c = [opts['epochs'] - i for i in range(opts['epochs'])]
        # plt.scatter(range(opts['epochs']), range(opts['epochs']), s=s, c=c)
        #
        # plt.savefig("img/features.pdf", bbox_inches='tight')

        plot_features(data_tr.tables['student_course'].embeddings['student'], row_embeds_out_tr, 'Student embeddings (train)', 'student_features_tr', sort=True)

        # plot_feature(data_tr.tables['student_course'].embeddings['student'][:,0], row_embeds_out_tr[:,0], 'Student embeddings 0', 'student_embeds_0', 'student', sort=True)
        # plot_feature(data_tr.tables['student_course'].embeddings['student'][:,1], row_embeds_out_tr[:,1], 'Student embeddings 1', 'student_embeds_1', 'student', sort=True)
        # plot_feature(data_tr.tables['student_course'].embeddings['course'][:,0], col_embeds_out_tr[:,0], 'Course embeddings 0', 'course_embeds_0', 'course', sort=True)
        # plot_feature(data_tr.tables['student_course'].embeddings['course'][:,1], col_embeds_out_tr[:,1], 'Course embeddings 1', 'course_embeds_1', 'course', sort=True)
        #
        # plot_feature(data_vl.tables['student_course'].embeddings['student'][:, 0], row_embeds_out_vl[:, 0], 'Student embeddings 0 - val', 'student_embeds_0_vl', 'student', sort=True)
        # plot_feature(data_vl.tables['student_course'].embeddings['student'][:, 1], row_embeds_out_vl[:, 1], 'Student embeddings 1 - val', 'student_embeds_1_vl', 'student', sort=True)
        # plot_feature(data_vl.tables['student_course'].embeddings['course'][:, 0], col_embeds_out_vl[:, 0], 'Course embeddings 0 - val', 'course_embeds_0_vl', 'course', sort=True)
        # plot_feature(data_vl.tables['student_course'].embeddings['course'][:, 1], col_embeds_out_vl[:, 1], 'Course embeddings 1 - val', 'course_embeds_1_vl', 'course', sort=True)
        #
        # plot_embeddings(data_tr.tables['student_course'].embeddings['student'], row_embeds_out_tr, 'Student embeddings', sort=True)

            ## Training
    #         team_player_noise = np.ones_like(data.team_player.values_tr)
    #         team_player_values_noisy = np.copy(data.team_player.values_tr)
    #
    #         team_match_noise = make_by_col_noise(opts['noise_rate'], data.team_match.num_values_tr, data.team_match.shape, data.team_match.indices_tr[:,1])
    #         noise_in = np.array([i for i in team_match_noise for _ in range(opts['model_opts']['units_in'])])
    #         noise_out = np.array([i for i in team_match_noise for _ in range(opts['model_opts']['units_out'])])
    #
    #         team_match_values_noisy = np.copy(data.team_match.values_tr)
    #         team_match_values_noisy[noise_in == 0] = noise_value
    #
    #         tr_dict = {team_player['indices']:data.team_player.indices_tr,
    #                    team_player['values']:team_player_values_noisy, # noisy values
    #                    team_player_noise_mask:team_player_noise,
    #                    team_player_values:data.team_player.values_tr, # clean values
    #
    #                    team_match['indices']:data.team_match.indices_tr,
    #                    team_match['values']:team_match_values_noisy, # noisy values
    #                    team_match_noise_mask:noise_out,
    #                    team_match_values:data.team_match.values_tr # clean values
    #                   }
    #
    #
    #         _, loss_tr, team_match_vals_out_tr, = sess.run([train_step, total_loss_tr, team_match_out_tr['values']], feed_dict=tr_dict)
    #         losses_tr.append(loss_tr)
    #
    #         num_features = opts['model_opts']['units_out']
    #         pred_accuracy_tr = one_hot_prediction_accuracy(data.team_match.values_tr, team_match_vals_out_tr, noise_out, num_features)
    #         accuracies_tr.append(pred_accuracy_tr)
    #
    #
    #
    #         ## Validation
    #         team_player_noise = np.ones_like(data.team_player.values_tr_vl)
    #         team_player_values_noisy = np.copy(data.team_player.values_tr_vl)
    #
    #         team_match_noise = 1 - data.team_match.split[data.team_match.split <= 1]
    #         noise_in = np.array([i for i in team_match_noise for _ in range(opts['model_opts']['units_in'])])
    #         noise_out = np.array([i for i in team_match_noise for _ in range(opts['model_opts']['units_out'])])
    #
    #         team_match_values_noisy = np.copy(data.team_match.values_tr_vl)
    #         team_match_values_noisy[noise_in == 0] = noise_value
    #
    #
    #         vl_dict = {team_player['indices']:data.team_player.indices_tr_vl,
    #                    team_player['values']:team_player_values_noisy, # noisy values
    #                    team_player_noise_mask:team_player_noise,
    #                    team_player_values:data.team_player.values_tr_vl, # clean values
    #
    #                    team_match['indices']:data.team_match.indices_tr_vl,
    #                    team_match['values']:team_match_values_noisy, # noisy values
    #                    team_match_noise_mask:noise_out,
    #                    team_match_values:data.team_match.values_tr_vl # clean values
    #                   }
    #
    #         loss_vl, team_match_vals_out_vl = sess.run([rec_loss_vl, team_match_out_vl['values']], feed_dict=vl_dict)
    #         losses_vl.append(loss_vl)
    #
    #
    #         means = np.mean(np.reshape(data.team_match.values_tr_vl, [-1, num_features]), axis=0)
    #         vals = np.reshape(noise_out * data.team_match.values_tr_vl, [-1,num_features])
    #         out = np.reshape(means, [-1,num_features])
    #         loss_vl_baseline = - np.mean(np.sum(vals * (out - np.max(out, axis=1)), axis=1))
    #         losses_vl_baseline.append(loss_vl_baseline)
    #         num_vals = vals.shape[0]
    #
    #         pred_accuracy_vl = one_hot_prediction_accuracy(data.team_match.values_tr_vl, team_match_vals_out_vl, noise_out, num_features)
    #         accuracies_vl.append(pred_accuracy_vl)
    #
    #
    #         print('MEANS ', means)
    #
    #         random_out = np.zeros([num_vals, num_features])
    #         random_out[np.arange(num_vals), np.random.choice(range(num_features), size=num_vals, p=means)] = 1
    #         random_out = np.reshape(random_out, [-1])
    #         pred_accuracy_vl_baseline = one_hot_prediction_accuracy(data.team_match.values_tr_vl, random_out, noise_out, num_features)
    #         accuracies_vl_baseline.append(pred_accuracy_vl_baseline)
    #
    #         if pred_accuracy_vl > accuracy_vl_best:
    #             accuracy_vl_best = pred_accuracy_vl
    #             accuracy_vl_best_ep = ep
    #
    #             if opts['save_model']:
    #                 path = os.path.join(opts['checkpoints_folder'], 'epoch_{:05d}'.format(ep) + '.ckpt')
    #                 saver.save(sess, path)
    #
    #
    #         if opts['verbosity'] > 0:
    #             # print("epoch {:5d}. training loss: {:5.15f} \t validation loss: {:5.5f} \t predicting mean: {:5.5f}".format(ep+1, loss_tr, loss_vl, loss_vl_baseline))
    #             print("epoch {:5d}. train accuracy rate: {:5.5f} \t val accuracy rate: {:5.5f} \t best val accuracy rate: {:5.5f} at epoch {:5d}".format(ep, pred_accuracy_tr, pred_accuracy_vl, accuracy_vl_best, accuracy_vl_best_ep))
    #
    #
    #
    #     show_last = opts['epochs']
    #     plt.title('CE Loss')
    #     plt.plot(range(opts['epochs'])[-show_last:], losses_vl_baseline[-show_last:], '.-', color='red')
    #     plt.plot(range(opts['epochs'])[-show_last:], losses_tr[-show_last:], '.-', color='blue')
    #     plt.plot(range(opts['epochs'])[-show_last:], losses_vl[-show_last:], '.-', color='green')
    #     plt.xlabel('epoch')
    #     plt.ylabel('CE')
    #     plt.legend(('mean', 'training', 'validation'))
    #     # plt.show()
    #     plt.savefig("rmse.pdf", bbox_inches='tight')
    #     plt.clf()
    #
    #     plt.title('Prediction Accuracy')
    #     plt.plot(range(opts['epochs'])[-show_last:], [.46 for _ in range(opts['epochs'])[-show_last:]], '.-', color='pink')
    #     plt.plot(range(opts['epochs'])[-show_last:], [.53 for _ in range(opts['epochs'])[-show_last:]], '.-', color='yellow')
    #     plt.plot(range(opts['epochs'])[-show_last:], accuracies_vl_baseline[-show_last:], '.-', color='red')
    #     plt.plot(range(opts['epochs'])[-show_last:], accuracies_tr[-show_last:], '.-', color='blue')
    #     plt.plot(range(opts['epochs'])[-show_last:], accuracies_vl[-show_last:], '.-', color='green')
    #     plt.xlabel('epoch')
    #     plt.ylabel('Accuracy')
    #     plt.legend(( 'baseline', 'experts', 'random', 'training prediction', 'validation prediction'))
    #     # plt.show()
    #     plt.savefig("pred.pdf", bbox_inches='tight')


if __name__ == "__main__":
    np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)


    # data_set = 'debug'
    # data_set = 'soccer'
    data_set = 'toy'

    one_hot = False
    units_in = 1
    embedding_size = 2
    units = 128
    # units = 2
    # units_out = units_in
    units_out = embedding_size

    # activation = tf.nn.relu
    activation = lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x) # Leaky Relu
    dropout_rate = 0.2
    skip_connections = True

    auto_restore = False
    save_model = False
    


    opts = {'epochs':200,
            'data_folder':'data',
            'data_set':data_set,
            'split_sizes':[.8, .2, .0], # train, validation, test split
            'noise_rate':dropout_rate, # match vl/tr or ts/(tr+vl) ?
            'regularization_rate':.00001,
            'learning_rate':.0001,
            'team_match_one_hot':one_hot,
            'toy_data':{'size_tr':[300, 200, 100],
                        'size_vl': [100, 70, 30],
                        'sparsity': .5,
            },
            'model_opts':{'pool_mode':'mean',
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
                                    {'type':ExchangeableLayer, 'units_out':units, 'activation':activation, 'skip_connections':skip_connections},
                                    {'type':ExchangeableLayer, 'units_out':units_out,  'activation':None, 'output_embeddings':True},
                                   ],
                         },
            'verbosity':2,
            'checkpoints_folder':'checkpoints',
            'restore_point_epoch':-1,
            'save_model':save_model,
            'debug':True,
            'seed':9858776,
            # 'seed': 9870112,
            }

    restore_point = None

    if auto_restore:         
        restore_point_epoch = sorted(glob.glob(opts['checkpoints_folder'] + "/epoch_*.ckpt*"))[-1].split(".")[0].split("_")[-1]
        restore_point = opts['checkpoints_folder'] + "/epoch_" + restore_point_epoch + ".ckpt"
        opts['restore_point_epoch'] = int(restore_point_epoch)

    main(opts, restore_point)






