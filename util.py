import numpy as np

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

def sparse_transpose(indices, values, shape, split, num_features):
    vals = np.reshape(values, [-1,num_features])
    trans = np.concatenate((indices, vals, split[:,None]), axis=1)
    inds = np.arange(num_features + 3)
    inds_t = np.copy(inds)
    inds_t[[1,0]] = inds_t[[0,1]]
    trans[:,inds] = trans[:,inds_t]
    trans = list(trans)
    trans.sort(key=lambda row: row[0])
    trans = np.array(trans)
    inds = trans[:,0:2].astype(np.int32)
    vals = trans[:,2:(num_features+2)]
    vals = np.reshape(vals, [-1])
    split = trans[:,(num_features+2)]
    shp = shape[[1,0]]
    return {'indices':inds, 'values':vals, 'shape':shp, 'split':split}


