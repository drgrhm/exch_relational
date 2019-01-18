# import numpy as np
# import matplotlib.pyplot as plt
import os
from util import *
import glob

if __name__ == "__main__":

    # experiment = 'sparsity'
    # experiment = 'embedding'
    # experiment = 'other_sparsity'
    experiment = 'side_info'

    ## Embeddings experiment
    if experiment == 'embedding':

        np.random.seed(9858776)
        seeds = np.random.randint(low=0, high=1000000, size=1)
        # seeds = [199527]
        for seed in seeds:

            image_path = 'img/embedding_experiment/' + str(seed) + '/'
            os.mkdir(image_path)
            checkpoint_path = 'checkpoints/embedding_experiment/' + str(seed) + '/'

            loss_file = open(checkpoint_path + 'loss.npz', 'rb')
            loss_data = np.load(loss_file)

            losses_tr = loss_data['losses_tr']
            losses_vl = loss_data['losses_vl']
            loss_mean = loss_data['loss_mean']
            plot_loss(losses_tr, losses_vl, loss_mean, 'Loss', image_path + 'loss.pdf')
            n_half = np.shape(losses_tr)[0] // 2
            plot_loss(losses_tr[n_half:], losses_vl[n_half:], loss_mean, 'Loss for last half', image_path + 'loss_last.pdf')

            loss_file.close()

            epochs = [int(s.split('_')[-1].split('.')[0]) for s in sorted(glob.glob(checkpoint_path + "embeddings_*.npz"))[:-1]] # Todo it would be better to save the 'save epochs' in a list as well than to do this:

            for i, epoch in enumerate(epochs):

                embeds_file = open(checkpoint_path + 'embeddings_{:05d}.npz'.format(epoch), 'rb')
                embeds_data = np.load(embeds_file)

                embeds_s = embeds_data['student_embeds_in']
                embeds_c = embeds_data['course_embeds_in']
                embeds_p = embeds_data['prof_embeds_in']

                predicts_vl_c = np.squeeze(embeds_data['course_embeds_out_vl_best'])
                predicts_vl_s = np.squeeze(embeds_data['student_embeds_out_vl_best'])
                predicts_vl_p = np.squeeze(embeds_data['prof_embeds_out_vl_best'])

                predicts_tr_c = np.squeeze(embeds_data['course_embeds_out_tr_best'])
                predicts_tr_s = np.squeeze(embeds_data['student_embeds_out_tr_best'])
                predicts_tr_p = np.squeeze(embeds_data['prof_embeds_out_tr_best'])

                plot_embeddings(embeds_s, predicts_tr_s, image_path + 'student_embeddings_tr_{:05d}.pdf'.format(epoch), title='Epoch {:d}'.format(epoch))
                plot_embeddings(embeds_c, predicts_tr_c, image_path + 'course_embeddings_tr_{:05d}.pdf'.format(epoch))
                plot_embeddings(embeds_p, predicts_tr_p, image_path + 'prof_embeddings_tr_{:05d}.pdf'.format(epoch))

                plot_embeddings(embeds_s, predicts_vl_s, image_path + 'student_embeddings_vl_{:05d}.pdf'.format(epoch), title='Epoch {:d}'.format(epoch))
                plot_embeddings(embeds_c, predicts_vl_c, image_path + 'course_embeddings_vl_{:05d}.pdf'.format(epoch))
                plot_embeddings(embeds_p, predicts_vl_p, image_path + 'prof_embeddings_vl_{:05d}.pdf'.format(epoch))

                if i == 0:
                    plot_embeddings(embeds_s, np.squeeze(embeds_data['student_embeds_init']), image_path + 'student_embeddings__init.pdf', title='Init')
                    plot_embeddings(embeds_c, np.squeeze(embeds_data['course_embeds_init']), image_path + 'course_embeddings__init.pdf', title='Init')
                    plot_embeddings(embeds_p, np.squeeze(embeds_data['prof_embeds_init']), image_path + 'prof_embeddings__init.pdf', title='Init')


    if experiment == 'sparsity':
        ## Sparsity experiment
        checkpoint_path = 'checkpoints/sparsity_experiment/'
        image_path = 'img/sparsity_experiment/'
        loss_file = open(checkpoint_path + 'loss.npz', 'rb')
        loss_data = np.load(loss_file)

        loss_ts = loss_data['loss_ts']
        loss_mean = loss_data['loss_mean']

        print(loss_ts)
        print(loss_mean)

        loss_ts = np.log(loss_ts)

        plt.plot(range(loss_ts.shape[0]), loss_ts, '.-', color='blue')
        # plt.plot(range(loss_ts.shape[0]), loss_mean, '.-', color='red')
        plt.legend(('test loss', 'predict mean'))

        plt.savefig(image_path + 'loss_ts.pdf', bbox_inches='tight')


    if experiment == 'other_sparsity':
        ## Sparsity experiment
        checkpoint_path = 'checkpoints/other_sparsity_experiment/'
        image_path = 'img/other_sparsity_experiment/'

        percent_observed = [.9, .8, .7, .6, .5, .4, .3, .2, .1]  # Must be decreasing

        losses_ts = []
        losses_mean = []
        improves = []

        for i, p in enumerate(percent_observed):

            loss_file = open(checkpoint_path + str(i) + '/loss.npz', 'rb')
            loss_data = np.load(loss_file)

            loss_ts = loss_data['loss_ts_vl_best']
            loss_mean = loss_data['loss_mean']

            losses_ts.append(loss_ts)
            losses_mean.append(loss_mean)

        plt.plot(range(len(losses_ts)), losses_ts, '.-', color='blue')
        # plt.plot(range(len(losses_ts)), losses_mean, '.-', color='red')
        plt.legend(('test loss', 'predict mean'))

        # improves = (normalize(losses_mean) - normalize(losses_ts)) / normalize(losses_mean)
        # plt.plot(range(len(losses_ts)), improves, '.-', color='blue')

        plt.savefig(image_path + 'loss_ts.pdf', bbox_inches='tight')


    if experiment == 'side_info':
        checkpoint_path = 'checkpoints/side_info_experiment/'
        image_path = 'img/side_info_experiment/'

        # loss_file = open(checkpoint_path + 'loss.npz', 'rb')
        # loss_data = np.load(loss_file)
        #
        # losses_ts = loss_data['losses_ts']
        # losses_mean = loss_data['losses_mean']
        #
        # ll = (losses_mean - losses_ts) / losses_mean

        # plt.plot(range(losses_ts.shape[0]), ll, '.-', color='blue')
        # plt.plot(range(losses_mean.shape[0]), losses_mean, '.-', color='red')
        # plt.legend(('test loss', 'predict mean'))
        #
        # plt.savefig(image_path + 'loss_ts.pdf', bbox_inches='tight')

        # percent_observed = [.9, .8, .7, .6, .5, .4, .3, .2, .1, .0]

        percent_observed = [.8, .0]  # Must be decreasing

        losses = {}
        n_runs = 3

        for i, p in enumerate(percent_observed):
            # losses[i] = {'test':0, 'mean':0}
            losses[i] = {'test':[], 'mean':[]}
            for k in range(n_runs):
                checkpoint_path = 'checkpoints/side_info_experiment/' + str(k) + '/' + str(i) + '/'
                loss_file = open(checkpoint_path + 'loss.npz', 'rb')

                loss_data = np.load(loss_file)

                losses[i]['test'].append(loss_data['loss_ts_vl_best'])
                losses[i]['mean'].append(loss_data['loss_mean'])

                # losses_tr = loss_data['losses_tr']
                # losses_vl = loss_data['losses_vl']
                # loss_mean = loss_data['loss_mean']
                # losses_mean = loss_mean * np.ones_like(losses_tr)

                # plt.plot(range(losses_tr.shape[0]), losses_tr, '.-', color='green')
                # plt.plot(range(losses_vl.shape[0]), losses_vl, '.-', color='blue')
                # plt.plot(range(losses_vl.shape[0]), losses_mean, '.-', color='red')
                # plt.legend(('training loss', 'validation loss', 'predict mean'))
                #
                # plt.savefig(image_path + 'loss_ts_' + str(i) + '.pdf', bbox_inches='tight')

        ii = 0
        # n_runs = 5
        # for k in range(n_runs):
        #
        #
        #     percent_observed = [1., .8, .6, .4, .2, .0]  # Must be decreasing
        #
        #
        #     for i, p in enumerate(percent_observed):
        #         checkpoint_path = 'checkpoints/side_info_experiment/' + str(k) + '/' + str(i) + '/'
        #
        #         loss_file = open(checkpoint_path + 'loss.npz', 'rb')
        #         loss_data = np.load(loss_file)
        #
        #         losses_tr = loss_data['losses_tr']
        #         losses_vl = loss_data['losses_vl']
        #         loss_mean = loss_data['loss_mean']
        #         losses_mean = loss_mean * np.ones_like(losses_tr)
        #
        #         plt.plot(range(losses_tr.shape[0]), losses_tr, '.-', color='green')
        #         plt.plot(range(losses_vl.shape[0]), losses_vl, '.-', color='blue')
        #         plt.plot(range(losses_vl.shape[0]), losses_mean, '.-', color='red')
        #         plt.legend(('training loss', 'validation loss', 'predict mean'))
        #
        #         plt.savefig(image_path + 'loss_ts_' + str(i) + '.pdf', bbox_inches='tight')



