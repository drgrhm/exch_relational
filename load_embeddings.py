# import numpy as np
# import matplotlib.pyplot as plt
from util import *

if __name__ == "__main__":

    subdir = 'tmp/checkpoints/'

    checkpoint_path = 'checkpoints/'
    image_path = 'img/'

    # loss_file = open(checkpoint_path + 'loss.npz', 'rb')
    # loss_data = np.load(loss_file)
    #
    # embeds_file = open(checkpoint_path + 'embeddings_best.npz', 'rb')
    # embeds_data = np.load(embeds_file)
    #
    # losses_tr = loss_data['losses_tr']
    # losses_vl = loss_data['losses_vl']
    #
    # plot_loss(losses_tr, losses_vl, None, 'Loss', image_path + 'loss')
    # plot_embeddings(embeds_data['student_embeds_in'], np.squeeze(embeds_data['student_embeds_out_vl_best']), 'Student embeddings', image_path + 'student_embeddings')
    # plot_embeddings(embeds_data['course_embeds_in'], np.squeeze(embeds_data['course_embeds_out_vl_best']), 'Course embeddings', image_path + 'course_embeddings')
    # plot_embeddings(embeds_data['prof_embeds_in'], np.squeeze(embeds_data['prof_embeds_out_vl_best']), 'Prof embeddings', image_path + 'prof_embeddings')
    #
    # loss_file.close()
    # embeds_file.close()


    # a = np.random.random((16, 16))

    a = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            a[i,j] = i * j

    # plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.imshow(a, cmap='plasma', interpolation='nearest')
    plt.show()
    plt.savefig(image_path + 'heat_map', bbox_inches='tight')
    plt.clf()