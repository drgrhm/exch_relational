import numpy as np

class Table:

    def __init__(self, tid, indices, values, shape, num_features, embeddings, split_sizes, split=None, split_mode='uniform'):
        n = values.shape[0]
        if split is None:
            tr, vl, ts = split_sizes
            assert tr + vl + ts == 1, "(" + str(tr) + ", " + str(vl) + ", " + str(ts) + ") is not a valid train/valid/test split"

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

            elif split_mode is None:
                n_tr = n
                n_vl, n_ts = 0, 0
                split = np.zeros(n)

        self.tid = tid
        self._set_data_splits(indices, values, split, num_features)
        self.shape = shape
        # self.mean_tr = np.mean(self.values_tr)
        # self.num_values_tr = n_tr
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




# class ToyTable:
#
#     def __init__(self, tid, indices, values, shape, num_features, embeddings):
#
#         n = values.shape[0]
#
#         self.shape = shape
#         self.num_values = n
#         self.embeddings = embeddings
#         self.indices = indices
#         self.values = values


