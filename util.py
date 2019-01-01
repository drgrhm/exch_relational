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

def plot_feature(embeds, predicts, title, plot_name, item_name, sort=False):

    assert embeds.shape == predicts.shape

    n = embeds.shape[0]

    if sort:
        score = np.zeros((n, 2))
        score[:,0] = embeds
        score[:,1] = predicts
        score.view('f8,f8').sort(order=['f0'], axis=0)
        embeds = score[:,0]
        predicts = score[:,1]

    plt.title(title)
    plt.plot(range(n), embeds, '.-', color='blue')
    plt.plot(range(n), predicts, '.-', color='green')
    plt.xlabel(item_name)
    plt.ylabel('feature value')
    plt.legend(('embeddings', 'predictions'))
    plt.show()
    plt.savefig('img/' + plot_name + '.pdf', bbox_inches='tight')
    plt.clf()

def plot_embeddings(embeds, predicts, title, plot_name, sort=False):

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

    plt.title(title)
    plt.plot(embeds[:,0], embeds[:,1], '.', color='blue')
    plt.plot(predicts[:,0], predicts[:,1], '.', color='green')
    plt.xlabel('feature 0')
    plt.ylabel('feature 1')
    plt.legend(('embeddings', 'predictions'))
    plt.show()
    plt.savefig(plot_name + '.pdf', bbox_inches='tight')
    plt.clf()


def plot_features(embeds, predicts, title, plot_name, sort=False, plot_rate=1.):

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
    s = 30 * sigmoid(normalize(embeds[:,0]))
    c = sigmoid(normalize([embeds.shape[0] - i for i in embeds[:,1]]))
    plt.scatter(predicts[:,0], predicts[:,1], s=s, c=c)
    # plt.scatter(sigmoid(normalize(predicts[:, 0])), sigmoid(normalize(predicts[:, 1])), s=s, c=c)
    plt.xlabel('feature 0')
    plt.ylabel('feature 1')
    plt.show()
    plt.savefig('img/' + plot_name + '.pdf', bbox_inches='tight')
    plt.clf()


def plot_loss(losses_tr, losses_vl, mean_tr, title, file_name):
    n = len(losses_tr)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(n), losses_tr, '.-', color='blue')
    plt.plot(range(n), losses_vl, '.-', color='green')
    if mean_tr is not None:
        plt.plot(range(n), mean_tr * np.ones(n), '-', color='red')
        plt.legend(('training', 'validation', 'mean'))
    else:
        plt.legend(('training', 'validation'))
    plt.show()
    plt.savefig('img/' + file_name + '.pdf', bbox_inches='tight')
    plt.clf()


def gaussian_embeddings(embedding_size, n_embeddings):
    """Multivariate Gaussian feature embeddings."""
    means = np.random.normal(0, 10, embedding_size)
    stds = np.random.uniform(1, 10, embedding_size)
    embeds = np.random.multivariate_normal(means, np.diag(stds), size=n_embeddings)
    return embeds

