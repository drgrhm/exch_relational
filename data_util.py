import numpy as np
import sqlite3
from itertools import groupby


class DataLoader:

    def __init__(self, data_folder, num_features, split_sizes):

        self.data_folder = data_folder
        self.team_player, self.team_match = self._load_tables(num_features, split_sizes)


    def _load_tables(self, num_features, split_sizes):
        
        # data = 'sample'
        data = 'soccer'

        if data == 'sample':

            inds0 = np.array([ [0,0],[0,2],[0,3],[1,0],[1,2],[2,1],[2,3],[3,1],[3,2],[3,3],[4,0],[4,1],[4,2],[5,1],[5,3] ])
            vals0 = np.array([ 1,0,1,1,0,0,1,0,1,1,0,1,0,1,1 ])
            shape0 = np.array([6,4])
            table0 = Table(0, inds0, vals0, shape0, num_features, split_sizes)


            inds1 = np.array([ [0,0],[0,2],[1,3],[2,0],[2,4],[3,3],[3,4],[4,1],[5,1],[5,2] ])
            vals1 = np.array([ 3,2,0.1,-3,-1,-0.1,1,-2,2,-2 ])
            shape1 = np.array([6,5])
            table1 = Table(1, inds1, vals1, shape1, num_features, split_sizes, split_mode='by_col')

        elif data == 'soccer':

            entities_map = {'player':0, 'team':1, 'match':2}
            player_map = {} ## map SQL id's to rows/cols of data matrices 
            team_map = {}
            match_map = {}

            conn = sqlite3.connect(self.data_folder + '/database.sqlite')

            ## TODO better features ...

            players = conn.execute("SELECT player_api_id FROM Player ORDER BY player_api_id;").fetchall()
            np.random.shuffle(players) 
            for i, player in enumerate(players):
                player_map[player[0]] = i
            n_players = i + 1            

            teams = conn.execute("SELECT team_api_id FROM Team ORDER BY team_api_id;").fetchall()
            np.random.shuffle(teams)
            for i, team in enumerate(teams):
                team_map[team[0]] = i
            n_teams = i + 1

            matches = conn.execute("SELECT match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, home_player_1, home_player_2, home_player_3, home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11, away_player_1, away_player_2, away_player_3, away_player_4, away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11 FROM Match;").fetchall()            
            np.random.shuffle(matches)

            team_match = []
            team_player = []

            max_goals = 0

            for i, match in enumerate(matches):
                match_id, home_team_id, away_team_id, home_team_goal, away_team_goal = match[0:5]
                home_team_id = team_map[home_team_id]
                away_team_id = team_map[away_team_id]
                home_player_ids = match[5:16]
                away_player_ids = match[16:27]
                
                match_map[match_id] = i
                match_id = i

                # team_match.append( [home_team_id, match_id, home_team_goal-away_team_goal] )
                # team_match.append( [away_team_id, match_id, away_team_goal-home_team_goal] )

                # if home_team_goal > max_goals:
                #     max_goals = home_team_goal
                # if away_team_goal > max_goals:
                #     max_goals = away_team_goal


                team_match.append( [home_team_id, match_id, home_team_goal, 1] )
                team_match.append( [away_team_id, match_id, away_team_goal, 0] )

                for player_id in home_player_ids:
                    if player_id != None:
                        team_player.append( [home_team_id, player_map[player_id], 1] )

                for player_id in away_player_ids:
                    if player_id != None:
                        team_player.append( [away_team_id, player_map[player_id], 1] )
            n_matches = i + 1

            # print('Max goals: ', max_goals)

            team_match.sort()
            team_player.sort()

            temp = []
            for key, group in groupby(team_player, key=lambda x: x[0:2]):
                key.append(sum([g[2] for g in group]))
                temp.append(key)
            
            team_player = temp

            team_player = np.array(team_player)
            team_match = np.array(team_match)

            table0 = Table(0, team_player[:,0:2], team_player[:,2], np.array([n_teams, n_players]), num_features, [1., .0, .0])
            table1 = Table(1, team_match[:,0:2], team_match[:,2:4], np.array([n_teams, n_matches]), num_features, split_sizes, split_mode='by_col')

            conn.close()

        return table0, table1




class Table:

    def __init__(self, tid, indices, values, shape, num_features, split_sizes, split_mode='uniform'):
        tr, vl, ts = split_sizes
        assert tr+vl+ts == 1, "(" + str(tr) + ", " + str(vl) + ", " + str(ts) + ") is not a valid train/valid/test split"
        n = values.shape[0]

        if split_mode == 'uniform':
            n_ts = int(n*ts)
            n_vl = int(n*vl)
            n_tr = n - n_ts - n_vl
            split = np.concatenate((np.zeros(n_tr, np.int32), np.ones(n_vl, np.int32), 2*np.ones(n_ts, np.int32)))
            np.random.shuffle(split)
        
        elif split_mode == 'by_col': # So that the two entries for each match are in the same tr/vl/ts split 
            n_cols = shape[1]
            n_cols_ts = int(n_cols*ts)
            n_cols_vl = int(n_cols*vl)
            n_cols_tr = n_cols - n_cols_ts - n_cols_vl
            col_split = np.concatenate((np.zeros(n_cols_tr, np.int32), np.ones(n_cols_vl, np.int32), 2*np.ones(n_cols_ts, np.int32)))
            np.random.shuffle(col_split)
            split = np.take(col_split, indices[:,1])

            n_ts = np.sum(split == 2)
            n_vl = np.sum(split == 1)
            n_tr = n - n_ts - n_vl


        # if num_features > 1:

            # values = values[:,0]


        new_vals = []
        for val in values:
            new_val = np.zeros(num_features)
            if len(values.shape) > 1:
                new_val[0:2] = val[0:2]
            else:
                new_val[0] = val
            new_vals.append(new_val)

        new_vals = np.reshape(np.array(new_vals), [-1])
        values = new_vals







        self._set_data_splits(indices, values, split, num_features)
        self.shape = shape
        self.mean_tr = np.mean(self.values_tr)
        self.num_values_tr = n_tr
        self.num_values = n
        

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


    # def transpose(self):
    #     trans = np.concatenate((self.indices_all, self.values_all[:,None], self.split[:,None]), axis=1)
    #     trans[:,[0,1,2,3]] = trans[:,[1,0,2,3]]        
    #     trans = list(trans)
    #     trans.sort(key=lambda row: row[0])
    #     trans = np.array(trans)

    #     inds = trans[:,0:2]
    #     vals = trans[:,2]
    #     split = trans[:,3]

    #     self._set_data_splits(inds, vals, split)

    #     self.shape[[0,1]] = self.shape[[1,0]]









# # For debugging
# def sparse_array_to_dense(indices, values, shape, num_features):
#     out = np.zeros(list(shape) + [num_features])
#     inds = expand_array_indices(indices, num_features)
#     inds = list(zip(*inds))
#     out[inds] = values
#     return out

# # For debugging
# def expand_array_indices(indices, num_features):    
#     num_vals = indices.shape[0]
#     inds_exp = np.reshape(np.tile(range(num_features), reps=[num_vals]), newshape=[-1, 1]) # expand dimension of mask indices
#     inds = np.tile(indices, reps=[num_features,1]) # duplicate indices num_features times
#     inds = np.reshape(inds, newshape=[num_features, num_vals, 2])
#     inds = np.reshape(np.transpose(inds, axes=[1,0,2]), newshape=[-1,2])
#     inds = np.concatenate((inds, inds_exp), axis=1)
#     return inds

# def sparse_transpose(indices, values, shape):
#     trans = np.concatenate((indices, values[:,None]), axis=1)
#     trans[:,[0,1,2]] = trans[:,[1,0,2]]
#     trans = list(trans)
#     trans.sort(key=lambda row: row[0])
#     trans = np.array(trans)    
#     inds = trans[:,0:2]
#     vals = trans[:,2]
#     shape[[0,1]] = shape[[1,0]]
#     return {'indices':inds, 'values':vals, 'shape':shape}









