import numpy as np

class Table:

    def __init__(self, tid, indices, values, shape, split_sizes, num_features, embeddings, one_hot=False, split_mode='uniform'):
        tr, vl, ts = split_sizes
        assert tr + vl + ts == 1, "(" + str(tr) + ", " + str(vl) + ", " + str(ts) + ") is not a valid train/valid/test split"
        n = values.shape[0]

        if split_mode == 'uniform':
            n_ts = int(n * ts)
            n_vl = int(n * vl)
            n_tr = n - n_ts - n_vl
            split = np.concatenate((np.zeros(n_tr, np.int32), np.ones(n_vl, np.int32), 2 * np.ones(n_ts, np.int32)))
            np.random.shuffle(split)

        elif split_mode == 'by_col':  # So that the two entries for each match are in the same tr/vl/ts split
            n_cols = shape[1]
            n_cols_ts = int(n_cols * ts)
            n_cols_vl = int(n_cols * vl)
            n_cols_tr = n_cols - n_cols_ts - n_cols_vl
            col_split = np.concatenate((np.zeros(n_cols_tr, np.int32), np.ones(n_cols_vl, np.int32), 2 * np.ones(n_cols_ts, np.int32)))
            np.random.shuffle(col_split)
            split = np.take(col_split, indices[:, 1])

            n_ts = np.sum(split == 2)
            n_vl = np.sum(split == 1)
            n_tr = n - n_ts - n_vl

        # if one_hot:
        #     split = np.array([i for i in split for _ in range(num_features)])
        #     indices = expand_indices(indices, num_features)
        #     new_vals = []
        #     for val in values:
        #         if val > 0:
        #             new_vals.append([1,0,0])
        #         elif val < 0:
        #             new_vals.append([0,0,1])
        #         elif val == 0:
        #             new_vals.append([0,1,0])
        #     new_vals = np.reshape(np.array(new_vals), [-1])
        #     values = new_vals


        elif split_mode is None:
            n_tr = n
            n_vl, n_ts = 0, 0
            split = np.zeros(n)

        if num_features > 1:
            # split = np.array([i for i in split for _ in range(num_features)])
            # indices = expand_indices(indices, num_features)

            if one_hot:  # TODO fix so generalizes better i.e. not just 3 features
                new_vals = []
                for val in values:
                    if val > 0:
                        new_vals.append([1, 0, 0])
                    elif val < 0:
                        new_vals.append([0, 0, 1])
                    elif val == 0:
                        new_vals.append([0, 1, 0])
            else:
                new_vals = np.array([[val, 0, 0] for val in values])

            new_vals = np.reshape(np.array(new_vals), [-1])
            values = new_vals

        self._set_data_splits(indices, values, split, num_features)
        self.shape = shape
        self.mean_tr = np.mean(self.values_tr)
        self.num_values_tr = n_tr
        self.num_values = n
        self.embeddings = embeddings

    def _set_data_splits(self, indices, values, split, num_features):
        self.indices_all = indices
        self.indices_tr = indices[split == 0]
        self.indices_vl = indices[split == 1]
        self.indices_tr_vl = indices[split <= 1]
        self.indices_ts = indices[split == 2]
        self.split = split

        split = np.array([i for i in split for _ in range(num_features)])

        self.values_all = values
        self.values_tr = values[split == 0]
        self.values_vl = values[split == 1]
        self.values_tr_vl = values[split <= 1]
        self.values_ts = values[split == 2]




class ToyTable:

    def __init__(self, tid, indices, values, shape, num_features, embeddings):

        n = values.shape[0]

        self.shape = shape
        self.num_values = n
        self.embeddings = embeddings
        self.indices = indices
        self.values = values


