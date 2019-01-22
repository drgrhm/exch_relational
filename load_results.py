# import numpy as np
# import matplotlib.pyplot as plt
import os
from util import *
import glob

if __name__ == "__main__":

    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

    # experiment = 'embedding'
    # experiment = 'sparsity'
    # experiment = 'side_info'
    experiment = 'sparsity_varied'

    ## Embeddings experiment
    if experiment == 'embedding':

        np.random.seed(9873866)
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

        percent_observed = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
        num_runs = 4

        loss_ts = np.zeros((num_runs, len(percent_observed)))
        loss_mean = np.zeros((num_runs, len(percent_observed)))

        for k in range(num_runs):
            loss_file = open(checkpoint_path + str(k) + '/loss.npz', 'rb')
            loss_data = np.load(loss_file)
            loss_ts[k, :] = np.array(loss_data['loss_ts'])
            loss_mean[k, :] = np.array(loss_data['loss_mean'])

        # for k in range(num_runs):
        #     for i, _ in enumerate(percent_observed):
        #         loss_file = open(checkpoint_path + str(k) + '/' + str(i) + '/loss.npz', 'rb')
        #         loss_data = np.load(loss_file)
        #         loss_ts[k, i] = np.array(loss_data['loss_ts_vl_best'])
        #         loss_mean[k, i] = np.array(loss_data['loss_mean'])

        colours_ts = ['#FF5733', '#33FFFF', '#7DFF33', '#3383FF', '#FC33FF']
        colours_mn = ['#FFBFB1', '#BDFFFF', '#C8FFA9', '#98BFFC', '#FBACFC']
        for k in range(num_runs):

            plt.plot(percent_observed, loss_ts[k, :], '.-', color=colours_ts[k])
            plt.plot(percent_observed, loss_mean[k, :], '.-', color=colours_mn[k])

        plt.xticks(percent_observed)
        plt.title("Sparsity")
        plt.xlabel("Percent observed")
        plt.ylabel("Loss")
        plt.savefig(image_path + 'sparsity_loss.pdf', bbox_inches='tight')
        plt.clf()


    # if experiment == 'side_info':
    #     ## Sparsity experiment
    #     checkpoint_path = 'checkpoints/side_info_experiment/'
    #     image_path = 'img/side_info_experiment/'
    #
    #     percent_observed = [.9, .8, .7, .6, .5, .4, .3, .2, .1, .0]
    #
    #     num_runs = 4
    #     loss_ts = np.zeros((num_runs, len(percent_observed)))
    #     loss_mean = np.zeros((num_runs, len(percent_observed)))
    #
    #     for k in range(num_runs):
    #         loss_file = open(checkpoint_path + str(k) + '/loss.npz', 'rb')
    #         loss_data = np.load(loss_file)
    #         loss_ts[k, :] = np.array(loss_data['losses_ts'])
    #         loss_mean[k, :] = np.array(loss_data['losses_mean'])
    #
    #     uppers = []
    #     lowers = []
    #     means_ts = []
    #     for j, p in enumerate(percent_observed):
    #         obs_ts = list(loss_ts[:, j])
    #         obs_ts.sort()
    #         mean_ts = np.mean(loss_ts, axis=0)
    #         means_ts.append(mean_ts)
    #     # loss_diff = loss_ts / loss_mean
    #     # for k in range(num_runs):
    #     #     plt.plot(percent_observed, loss_diff[k, :], '.-')
    #     #     plt.plot(percent_observed, loss_mean[k, :], '.-', color=colours_mn[k])
    #     plt.xticks(percent_observed)
    #     plt.title("Side-Info")
    #     plt.xlabel("Percent observed (side tables)")
    #     plt.ylabel("Loss")
    #     plt.savefig(image_path + 'loss_ts.pdf', bbox_inches='tight')


    if experiment == 'side_info':
        ## Sparsity experiment
        checkpoint_path = 'checkpoints/side_info_experiment/'
        image_path = 'img/side_info_experiment/'

        percent_observed = [.9, .8, .7, .6, .5, .4, .3, .2, .1, .0]

        num_runs = 5
        loss_ts = np.zeros((num_runs, len(percent_observed)))
        loss_mean = np.zeros((num_runs, len(percent_observed)))

        for k in range(num_runs):
            loss_file = open(checkpoint_path + str(k) + '/loss.npz', 'rb')
            loss_data = np.load(loss_file)
            loss_ts[k, :] = np.array(loss_data['losses_ts'])
            loss_mean[k, :] = np.array(loss_data['losses_mean'])


        # colours_ts = ['#FF5733', '#33FFFF', '#7DFF33', '#3383FF', '#FC33FF']
        # colours_mn = ['#FFBFB1', '#BDFFFF', '#C8FFA9', '#98BFFC', '#FBACFC']
        # for k in range(num_runs):
        #
        #     plt.plot(percent_observed, loss_ts[k, :], '.-', color=colours_ts[k])
        #     plt.plot(percent_observed, loss_mean[k, :], '.-', color=colours_mn[k])

        plt.plot(percent_observed, np.mean(loss_ts, axis=0), '.-', color='blue')
        plt.plot(percent_observed, np.mean(loss_mean, axis=0), '.-', color='red')

        # # loss_diff = loss_ts / loss_mean
        # loss_diff = loss_mean
        #
        # loss_diff = np.sort(loss_diff, axis=0)
        # loss_mean = np.mean(loss_diff, axis=0)
        #
        # loss_up = []
        # loss_lo = []
        #
        # for j, _ in enumerate(percent_observed):
        #     d_up = loss_diff[-1, j] - loss_mean[j]
        #     d_lo = loss_mean[j] - loss_diff[0, j]
        #
        #     if d_up > d_lo:
        #         loss_up.append(max(loss_diff[-2, j], loss_mean[j]))
        #         loss_lo.append(min(loss_diff[0, j], loss_mean[j]))
        #     else:
        #         loss_up.append(max(loss_diff[-1, j], loss_mean[j]))
        #         loss_lo.append(min(loss_diff[1, j], loss_mean[j]))
        #
        # loss_up = np.array(loss_up)
        # loss_lo = np.array(loss_lo)
        #
        # plt.plot(percent_observed, loss_mean, '.-', color='blue')
        # plt.plot(percent_observed, loss_up, '.-', color='green')
        # plt.plot(percent_observed, loss_lo, '.-', color='red')
        # plt.fill_between(percent_observed, loss_lo, loss_up)

        plt.xticks(percent_observed)
        plt.title("Side-Info")
        plt.xlabel("Percent observed (side tables)")
        plt.ylabel("Loss")
        plt.savefig(image_path + 'side_info_loss_mean.pdf', bbox_inches='tight')


    if experiment == 'sparsity_varied':
        ## Sparsity experiment
        checkpoint_path = 'checkpoints/sparsity_varied_experiment/'
        image_path = 'img/sparsity_varied_experiment/'

        percent_observed = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
        num_runs = 3

        # loss_ts = np.zeros((num_runs, len(percent_observed), len(percent_observed)))
        # loss_mean = np.zeros((num_runs, len(percent_observed), len(percent_observed)))

        loss_ts_ave = np.zeros((len(percent_observed), len(percent_observed)))
        loss_mean_ave = np.zeros((len(percent_observed), len(percent_observed)))

        for k in range(num_runs):
            loss_file = open(checkpoint_path + 'run_{:d}_loss_varied.npz'.format(k), 'rb')
            loss_data = np.load(loss_file)
            loss_ts = loss_data['loss_ts']
            loss_mean = loss_data['loss_mean']

            loss_ts_ave += loss_ts
            loss_mean_ave += loss_mean

        loss_ts_ave /= num_runs
        loss_mean_ave /= num_runs

        aa = 1

            # plt.imshow(loss_mean - loss_ts, cmap='hot', interpolation='nearest')
            # plt.savefig(image_path + 'mean_loss.pdf', bbox_inches='tight')
            # plt.clf()