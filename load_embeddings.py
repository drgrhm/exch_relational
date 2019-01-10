# import numpy as np
# import matplotlib.pyplot as plt
import os
from util import *

if __name__ == "__main__":

    ## Embeddings experiment
    # np.random.seed(9858776)
    # seeds = np.random.randint(low=0, high=1000000, size=5)

    seeds = [96793, 160762, 179138, 388154, 488463, 907634, 988205]

    for seed in seeds:

        image_path = 'img/embedding_experiment/' + str(seed) + '/'
        os.mkdir(image_path)
        checkpoint_path = 'checkpoints/embedding_experiment/' + str(seed) + '/'

        loss_file = open(checkpoint_path + 'loss.npz', 'rb')
        loss_data = np.load(loss_file)

        embeds_file = open(checkpoint_path + 'embeddings_best.npz', 'rb')
        embeds_data = np.load(embeds_file)

        losses_tr = loss_data['losses_tr']
        losses_vl = loss_data['losses_vl']
        plot_loss(losses_tr, losses_vl, None, 'Loss', image_path + 'loss.pdf')
        n_half = np.shape(losses_tr)[0] // 2
        plot_loss(losses_tr[n_half:], losses_vl[n_half:], None, 'Loss for last half', image_path + 'loss_last.pdf')


        # rmses_tr = loss_data['rmses_tr']
        # rmses_vl = loss_data['rmses_vl']
        # rmse_mean = loss_data['loss_mean_tr_vl']
        # plot_loss(rmses_tr, rmses_vl, rmse_mean, 'RMSE', image_path + 'rmse.pdf')
        # plot_loss(rmses_tr[n_half:], rmses_vl[n_half:], rmse_mean, 'RMSE for last half', image_path + 'rmse_last.pdf')

        plot_embeddings(embeds_data['student_embeds_in'], np.squeeze(embeds_data['student_embeds_out_vl_best']), 'Student embeddings', image_path + 'student_embeddings.pdf', remove_outliers=False)
        plot_embeddings(embeds_data['course_embeds_in'], np.squeeze(embeds_data['course_embeds_out_vl_best']), 'Course embeddings', image_path + 'course_embeddings.pdf', remove_outliers=False)
        plot_embeddings(embeds_data['prof_embeds_in'], np.squeeze(embeds_data['prof_embeds_out_vl_best']), 'Prof embeddings', image_path + 'prof_embeddings.pdf', remove_outliers=False)

        # make_embeddings_plot(embeds_data)

        loss_file.close()
        embeds_file.close()




    # ## Sparsity experiment
    # checkpoint_path = 'checkpoints/sparsity_experiment/'
    # image_path = 'img/sparsity_experiment/'
    # loss_file = open(checkpoint_path + 'loss.npz', 'rb')
    # loss_data = np.load(loss_file)
    #
    # loss_ts = loss_data['loss_ts']
    # loss_mean = loss_data['loss_mean']
    #
    # print(loss_ts)
    # print(loss_mean)
    #
    # # plt.imshow(loss_ts, cmap='plasma', interpolation='nearest')
    # plt.imshow(normalize(loss_ts), cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.savefig(image_path + 'heat_map', bbox_inches='tight')
    # plt.clf()


    ## Side info experiment
    checkpoint_path = 'checkpoints/side_info_experiment/'
    image_path = 'img/side_info_experiment/'

