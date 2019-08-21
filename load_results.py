# import numpy as np
# import matplotlib.pyplot as plt
import os
from util import *
import glob

if __name__ == "__main__":

    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

    # experiment = 'embedding'
    # experiment = 'sparsity'
    experiment = 'side_info'
    # experiment = 'sparsity_varied'
    # experiment = 'side_info_varied'

    ## Embeddings experiment
    if experiment == 'embedding':

        np.random.seed(9858776)
        seeds = np.random.randint(low=0, high=1000000, size=1)
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

            epochs = [int(s.split('_')[-1].split('.')[0]) for s in sorted(glob.glob(checkpoint_path + "embeddings_*.npz"))[:-1]]

            for i, epoch in enumerate(epochs):

                embeds_file = open(checkpoint_path + 'embeddings_{:05d}.npz'.format(epoch), 'rb')
                embeds_data = np.load(embeds_file)

                embeds_s = embeds_data['student_embeds_in']
                embeds_c = embeds_data['course_embeds_in']
                embeds_p = embeds_data['prof_embeds_in']

                predicts_vl_c = np.squeeze(embeds_data['course_embeds_out_vl_best'])
                predicts_vl_s = np.squeeze(embeds_data['student_embeds_out_vl_best'])
                predicts_vl_p = np.squeeze(embeds_data['prof_embeds_out_vl_best'])

                plot_embeddings(embeds_s, predicts_vl_s, image_path + 'student_embeddings_vl_{:05d}.pdf'.format(epoch))
                plot_embeddings(embeds_c, predicts_vl_c, image_path + 'course_embeddings_vl_{:05d}.pdf'.format(epoch))
                plot_embeddings(embeds_p, predicts_vl_p, image_path + 'prof_embeddings_vl_{:05d}.pdf'.format(epoch))

                if i == 0:
                    plot_embeddings(embeds_s, np.squeeze(embeds_data['student_embeds_init']), image_path + 'student_embeddings__init.pdf', xlabel='$(a)$')
                    plot_embeddings(embeds_c, np.squeeze(embeds_data['course_embeds_init']), image_path + 'course_embeddings__init.pdf', xlabel='$(a)$')
                    plot_embeddings(embeds_p, np.squeeze(embeds_data['prof_embeds_init']), image_path + 'prof_embeddings__init.pdf', xlabel='$(a)$')


    if experiment == 'sparsity':
        ## Sparsity experiment
        checkpoint_path = 'checkpoints/sparsity_experiment/'
        image_path = 'img/sparsity_experiment/'

        percent_observed = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
        num_runs = 5

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


    if experiment == 'side_info':
        ## Sparsity experiment
        checkpoint_path = 'checkpoints/side_info_experiment/'
        image_path = 'img/side_info_experiment/'

        # percent_observed = [.9, .8, .7, .6, .5, .4, .3, .2, .1, .0]
        # percent_observed = [.5, .1, .0]
        # percent_observed = [.5, .3, .1, .0]
        percent_observed = np.logspace(0, -1.6, num=11, endpoint=True)  # log_10(-1.6) corresponds to 2.5% sparsity level, which ensures at least 3 entries per row and column. Include 1. just for constructing observed masks
        percent_observed = percent_observed[1:]
        num_runs = 8

        loss_ts = np.zeros((num_runs, len(percent_observed)))
        loss_mean = np.zeros((num_runs, len(percent_observed)))

        for k in range(num_runs):
            loss_file = open(checkpoint_path + str(k) + '/loss.npz', 'rb')
            loss_data = np.load(loss_file)
            loss_ts[k, :] = np.array(loss_data['losses_ts'])
            loss_mean[k, :] = np.array(loss_data['losses_mean'])


        # colours_ts = ['#FF5733', '#33FFFF', '#7DFF33', '#3383FF', '#FC33FF', '#FD5733', '#3DFFFF', '#7FFF33', '#3D83FF', '#FD33FF']
        # colours_mn = ['#FFBFB1', '#BDFFFF', '#C8FFA9', '#98BFFC', '#FBACFC', '#FDBFB1', '#B3FFFF', '#CFFFA9', '#9DBFFC', '#FDACFC']
        # for k in range(num_runs):
        #
        #     plt.plot(percent_observed, loss_ts[k, :], '.-', color=colours_ts[k])
            # plt.plot(percent_observed, loss_mean[k, :], '.-', color=colours_mn[k], alpha=.3)

        mean_means = np.mean(loss_mean, axis=0)
        means = np.mean(loss_ts, axis=0)
        stds = np.std(loss_ts, axis=0)
        upper = means + 1.96 * stds
        lower = means - 1.96 * stds

        plt.plot(percent_observed, np.mean(loss_ts, axis=0), '.-', color='green')
        # plt.plot(percent_observed, np.mean(loss_mean, axis=0), '.-', color='red')
        plt.fill_between(percent_observed, lower, upper, color='green', alpha=.5)

        # plt.xticks(percent_observed)
        # plt.xticks([.02,.1,.2,.3,.4,.5])
        # plt.title("Side-Info (mean over {:d} runs with 95% CI)".format(num_runs))
        plt.xlabel("Percent observed (side tables)")
        plt.ylabel("Loss")
        # plt.axes().set_aspect(2/3, 'datalim')
        plt.savefig(image_path + 'side_info_loss_mean.pdf', bbox_inches='tight')

        # path = os.path.join('checkpoints', 'side_info_experiment', 'side_info_loss.npz')
        # file = open(path, 'wb')
        # np.savez(file, loss_ts=loss_ts, loss_mean=loss_mean)
        # file.close()


    if experiment == 'sparsity_varied':
        ## Sparsity experiment

        inductive = True

        checkpoint_path = 'checkpoints/sparsity_varied_experiment/'
        image_path = 'img/sparsity_varied_experiment/'

        # percent_observed = np.logspace(0, -1.6, num=11, endpoint=True) # log_10(-1.6) corresponds to 2.5% sparsity level, which ensures at least 3 entries per row and column. Include 1. just for constructing observed masks
        # percent_observed = percent_observed [1:]
        percent_observed = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
        num_runs = 10

        n_observed = len(percent_observed)
        # n_observed = 10
        loss_ts_ave = np.zeros((n_observed, n_observed))
        loss_mean_ave = np.zeros((n_observed, n_observed))

        for k in range(num_runs):
            loss_file_ind = open(checkpoint_path + 'run_{:d}_loss_varied_inductive.npz'.format(k), 'rb')
            loss_file_tra = open(checkpoint_path + 'run_{:d}_loss_varied.npz'.format(k), 'rb')
            if inductive:
                loss_file = loss_file_ind
            else:
                loss_file = loss_file_tra

            loss_data_ind = np.load(loss_file_ind)
            loss_data_tra = np.load(loss_file_tra)
            loss_ind = loss_data_ind['loss_ts']
            loss_tra = loss_data_tra['loss_ts']
            AA = (loss_ind - loss_tra) / loss_tra

            loss_data = np.load(loss_file)
            loss_ts = loss_data['loss_ts']
            loss_mean = loss_data['loss_mean']

            # #####
            # if loss_ts.shape[0] > len(percent_observed):
            #     loss_ts = loss_ts[:-1, :-1]
            #     loss_mean = loss_mean[:-1, :-1]
            # #####

            loss_ts_ave += loss_ts
            loss_mean_ave += loss_mean

            plt.imshow(loss_ts, interpolation='nearest')
            if inductive:
                plt.savefig(image_path + 'inductive_loss_{:d}.pdf'.format(k), bbox_inches='tight')
            else:
                plt.savefig(image_path + 'loss_{:d}.pdf'.format(k), bbox_inches='tight')
            plt.clf()

        loss_ts_ave /= num_runs
        loss_mean_ave /= num_runs

        # loss_ts_ave = (loss_mean_ave - loss_ts_ave) / loss_mean_ave

        print(loss_ts_ave)

        plt.imshow(loss_ts_ave, interpolation='nearest')
        if inductive:
            plt.savefig(image_path + 'inductive_loss_ave.pdf', bbox_inches='tight')
        else:
            plt.savefig(image_path + 'loss_ave.pdf', bbox_inches='tight')
        plt.clf()



    if experiment == 'side_info_varied':

        inductive = False


        checkpoint_path = 'checkpoints/side_info_varied_experiment/'
        image_path = 'img/side_info_varied_experiment/'

        # percent_observed = np.logspace(0, -1.6, num=11, endpoint=True) # log_10(-1.6) corresponds to 2.5% sparsity level, which ensures at least 3 entries per row and column. Include 1. just for constructing observed masks
        # percent_observed = percent_observed [1:]
        percent_observed = [.9, .8, .7, .6, .5, .4, .3, .2, .1, .0]
        num_runs = 9

        n_observed = len(percent_observed)
        # n_observed = 10
        loss_ts_ave = np.zeros((n_observed, n_observed))
        loss_mean_ave = np.zeros((n_observed, n_observed))

        for k in range(num_runs):

            loss_file_ind = open(checkpoint_path + 'run_{:d}_loss_varied_inductive.npz'.format(k), 'rb')
            loss_file_tra = open(checkpoint_path + 'run_{:d}_loss_varied.npz'.format(k), 'rb')
            if inductive:
                loss_file = loss_file_ind
            else:
                loss_file = loss_file_tra

            loss_ind = np.load(loss_file_ind)['loss_ts']
            loss_tra = np.load(loss_file_tra)['loss_ts']
            AA = loss_file_ind - loss_file_tra

            loss_data = np.load(loss_file)
            loss_ts = loss_data['loss_ts']
            loss_mean = loss_data['loss_mean']

            loss_ts_ave += loss_ts
            loss_mean_ave += loss_mean

            plt.imshow(loss_ts, interpolation='nearest')
            if inductive:
                plt.savefig(image_path + 'inductive_loss_{:d}.pdf'.format(k), bbox_inches='tight')
            else:
                plt.savefig(image_path + 'loss_{:d}.pdf'.format(k), bbox_inches='tight')
            plt.clf()

        loss_ts_ave /= num_runs
        loss_mean_ave /= num_runs

        # #####
        loss_ts_ave = loss_ts_ave[:,:-1]
        loss_mean_ave = loss_mean_ave[:, :-1]
        # #####
        # loss_ts_ave = (loss_mean_ave - loss_ts_ave) / loss_mean_ave


        print(loss_ts_ave)

        plt.imshow(loss_ts_ave, interpolation='nearest')
        if inductive:
            plt.savefig(image_path + 'inductive_loss_ave.pdf', bbox_inches='tight')
        else:
            plt.savefig(image_path + 'loss_ave.pdf', bbox_inches='tight')
        plt.clf()
