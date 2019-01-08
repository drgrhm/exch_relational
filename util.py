import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# For debugging
def sparse_array_to_dense(indices, values, shape, num_features):
    out = np.zeros(list(shape) + [num_features])
    inds = expand_indices(indices, num_features)
    inds = list(zip(*inds))
    out[inds] = values
    return out

def expand_indices(indices, num_features):    
    num_vals = indices.shape[0]
    inds_exp = np.reshape(np.tile(range(num_features), reps=[num_vals]), newshape=[-1, 1]) # expand dimension of mask indices
    inds = np.tile(indices, reps=[num_features,1]) # duplicate indices num_features times
    inds = np.reshape(inds, newshape=[num_features, num_vals, 2])
    inds = np.reshape(np.transpose(inds, axes=[1,0,2]), newshape=[-1,2])
    inds = np.concatenate((inds, inds_exp), axis=1)
    return inds



# def sparse_transpose(indices, values, shape, split):
#     trans = np.concatenate((indices, values[:,None], split[:,None]), axis=1)
#     trans[:,[0,1,2,3]] = trans[:,[1,0,2,3]]
#     trans = list(trans)
#     trans.sort(key=lambda row: row[0])
#     trans = np.array(trans)
#     inds = trans[:,0:2]
#     vals = trans[:,2]
#     split = trans[:,3]
#     shape[[0,1]] = shape[[1,0]]
#     return {'indices':inds, 'values':vals, 'shape':shape, 'split':split}

def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x)))

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

# def plot_feature(embeds, predicts, title, plot_name, item_name, sort=False):
#
#     assert embeds.shape == predicts.shape
#
#     n = embeds.shape[0]
#
#     if sort:
#         score = np.zeros((n, 2))
#         score[:,0] = embeds
#         score[:,1] = predicts
#         score.view('f8,f8').sort(order=['f0'], axis=0)
#         embeds = score[:,0]
#         predicts = score[:,1]
#
#     plt.title(title)
#     plt.plot(range(n), embeds, '.-', color='blue')
#     plt.plot(range(n), predicts, '.-', color='green')
#     plt.xlabel(item_name)
#     plt.ylabel('feature value')
#     plt.legend(('embeddings', 'predictions'))
#     plt.show()
#     plt.savefig('img/' + plot_name + '.pdf', bbox_inches='tight')
#     plt.clf()
#
# def plot_embeddings(embeds, predicts, title, plot_name, sort=False):
#
#     assert embeds.shape[1] == 2
#     assert embeds.shape == predicts.shape
#
#     if sort:
#         score = np.zeros((embeds.shape[0], 5))
#         score[:,0] = embeds[:,0] * embeds[:,1]
#         score[:,1:3] = embeds
#         score[:,3:] = predicts
#         score.view('f8,f8,f8,f8,f8').sort(order=['f0'], axis=0)
#         embeds = score[:, 1:3]
#         predicts = score[:, 3:]
#
#     plt.title(title)
#     plt.plot(embeds[:,0], embeds[:,1], '.', color='blue')
#     plt.plot(predicts[:,0], predicts[:,1], '.', color='green')
#     plt.xlabel('feature 0')
#     plt.ylabel('feature 1')
#     plt.legend(('embeddings', 'predictions'))
#     plt.show()
#     plt.savefig(plot_name + '.pdf', bbox_inches='tight')
#     plt.clf()


def plot_embeddings(embeds, predicts, title, path, sort=False, plot_rate=1.):

    assert embeds.shape[1] == 2
    assert embeds.shape == predicts.shape

    if sort:
        score = np.zeros((embeds.shape[0], 5))
        score[:,0] = embeds[:,0] * embeds[:,1]
        score[:,1:3] = embeds
        score[:,3:] = predicts
        score.view('f8,f8,f8,f8,f8').sort(order=['f0'], axis=0)
        embeds = score[:, 1:3]
        predicts = score[:, 3:]

    if plot_rate < 1.:
        mask = np.random.choice((0,1), size=embeds.shape[0], replace=True, p=(1-plot_rate, plot_rate))
        embeds = embeds[mask == 1,:]
        predicts = predicts[mask == 1, :]

    plt.title(title)
    # plt.locator_params(axis='x', nbins=5)
    # plt.locator_params(axis='y', nbins=5)
    s = 10 + 10 * np.exp(normalize(embeds[:, 0]))
    c = sigmoid(normalize(embeds[:, 1]))
    # s = 30 * normalize(embeds[:, 0])
    # c = normalize(embeds[:, 1])
    plt.scatter(predicts[:,0], predicts[:,1], s=s, c=c, cmap='plasma', alpha=.5)
    plt.xlabel('embedding[0]')
    plt.ylabel('embedding[1]')
    plt.show()
    plt.savefig(path, bbox_inches='tight')
    plt.clf()


# def make_embeddings_plot(embeddings_data):
#     # plot_embeddings(embeds_data['student_embeds_in'], np.squeeze(embeds_data['student_embeds_out_vl_best']), 'Student embeddings', image_path + 'student_embeddings.pdf')
#     plt.subplot(1, 3, 1)
#     s = 10 * np.exp(normalize(embeddings_data['student_embeds_in'][:, 0]))
#     c = sigmoid(normalize(embeddings_data['student_embeds_in'][:, 1]))
#     plt.scatter(predicts[:, 0], predicts[:, 1], s=s, c=c, alpha=.7)


def plot_loss(losses_tr, losses_vl, mean_tr, title, path):
    n = len(losses_tr)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(n), losses_tr, '.-', color='blue')
    plt.plot(range(n), losses_vl, '.-', color='green')
    # plt.plot(range(n), losses_ts, '.-', color='red')
    if mean_tr is not None:
        plt.plot(range(n), mean_tr * np.ones(n), '-', color='yellow')
        plt.legend(('training', 'validation', 'mean'))
    else:
        plt.legend(('training', 'validation'))
    plt.show()
    plt.savefig(path, bbox_inches='tight')
    plt.clf()


def gaussian_embeddings(embedding_size, n_embeddings):
    """Multivariate Gaussian feature embeddings."""
    means = np.random.normal(0, 10, embedding_size)
    stds = np.random.uniform(1, 10, embedding_size)
    embeds = np.random.multivariate_normal(means, np.diag(stds), size=n_embeddings)
    return embeds


def np_rmse_loss(values_in, values_out, noise_mask):
    diffs = ((values_in - values_out) ** 2) * noise_mask
    return np.sqrt(np.sum(diffs) / np.sum(noise_mask))


def update_observed(observed_old, p_keep, min_observed):

    inds_sc = np.array(np.nonzero(observed_old)).T

    n_keep = int(p_keep * inds_sc.shape[0])
    n_drop = inds_sc.shape[0] - n_keep

    inds_sc_keep = np.concatenate( (np.ones(n_keep), np.zeros(n_drop)) )
    np.random.shuffle(inds_sc_keep)
    inds_sc = inds_sc[inds_sc_keep == 1, :]

    observed_new = np.zeros_like(observed_old)
    observed_new[inds_sc[:,0], inds_sc[:,1]] = 1

    shape = observed_new.shape
    rows = np.sum(observed_new, axis=1)
    for i in np.array(range(shape[0]))[rows < min_observed]:
        diff = observed_old[i, :] - observed_new[i, :]
        resample_inds = np.array(range(shape[1]))[diff == 1]
        jj = np.random.choice(resample_inds, int(min_observed - rows[i]), replace=False)
        observed_new[i, jj] = 1

    cols = np.sum(observed_new, axis=0)
    for j in np.array(range(shape[1]))[cols < min_observed]:
        diff = observed_old[:, j] - observed_new[:, j]
        resample_inds = np.array(range(shape[0]))[diff == 1]
        ii = np.random.choice(resample_inds, int(min_observed - cols[j]), replace=False)
        observed_new[ii, j] = 1

    return observed_new


def choose_observed(tid, shape, sparsity, min_observed=1):
    """Which entries of the matrix to consider as observed."""

    obs = np.random.choice([0,1], shape, p=(1-sparsity, sparsity))

    rows = np.sum(obs, axis=1)
    for i in  np.array(range(shape[0]))[rows < min_observed]:
        jj = np.random.choice(range(shape[1]), min_observed, replace=False)
        obs[i, jj] = 1

    cols = np.sum(obs, axis=0)
    for j in  np.array(range(shape[1]))[cols < min_observed]:
        ii = np.random.choice(range(shape[0]), min_observed, replace=False)
        obs[ii, j] = 1

    print("final density of observed values in table ", tid,  ": ", np.sum(obs) / (shape[0] * shape[1]))

    return obs