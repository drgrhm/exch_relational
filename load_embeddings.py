# import numpy as np
# import matplotlib.pyplot as plt
import os
from util import *

if __name__ == "__main__":

    # subdir = 'tmp/checkpoints/'

    np.random.seed(9870112)
    seeds = np.random.randint(low=0, high=1000000, size=3)

    for seed in seeds:

        image_path = 'img/embedding_experiment/' + str(seed) + '/'
        os.mkdir(image_path)

        ## Embeddings experiment
        checkpoint_path = 'checkpoints/embedding_experiment/' + str(seed) + '/'

        loss_file = open(checkpoint_path + 'loss.npz', 'rb')
        loss_data = np.load(loss_file)

        embeds_file = open(checkpoint_path + 'embeddings_best.npz', 'rb')
        embeds_data = np.load(embeds_file)

        # losses_tr = loss_data['losses_tr']
        # losses_vl = loss_data['losses_vl']
        # loss_mean = loss_data['loss_mean_tr_vl']
        #
        # n_half = np.shape(losses_tr)[0] // 2
        #
        # plot_loss(losses_tr, losses_vl, loss_mean, 'Loss', image_path + 'loss.pdf')
        # plot_loss(losses_tr[n_half:], losses_vl[n_half:], loss_mean, 'Loss for last half', image_path + 'loss_last.pdf')
        plot_embeddings(embeds_data['student_embeds_in'], np.squeeze(embeds_data['student_embeds_out_vl_best']), 'Student embeddings', image_path + 'student_embeddings.pdf')
        plot_embeddings(embeds_data['course_embeds_in'], np.squeeze(embeds_data['course_embeds_out_vl_best']), 'Course embeddings', image_path + 'course_embeddings.pdf')
        plot_embeddings(embeds_data['prof_embeds_in'], np.squeeze(embeds_data['prof_embeds_out_vl_best']), 'Prof embeddings', image_path + 'prof_embeddings.pdf')

        # make_embeddings_plot(embeds_data)

        loss_file.close()
        embeds_file.close()




    # ## Sparsity experiment
    # checkpoint_path = 'checkpoints/sparsity_experiment'
    # loss_file = open(checkpoint_path + '/loss.npz', 'rb')
    # loss_data = np.load(loss_file)
    #
    # loss_ts = loss_data['loss_ts']
    # loss_mean = loss_data['loss_mean']
    #
    # # plt.imshow(loss_ts, cmap='plasma', interpolation='nearest')
    # plt.imshow(loss_ts, cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.savefig(image_path + 'heat_map', bbox_inches='tight')
    # plt.clf()
