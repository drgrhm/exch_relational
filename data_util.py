# import numpy as np
import sqlite3
from itertools import groupby
from util import *
from table import Table


class DataLoader:

    def __init__(self, data_folder, data_set, split_sizes, num_features, team_match_one_hot=False):

        self.data_folder = data_folder
        self.data_set = data_set
        # self.team_player, self.team_match = self._load_tables(split_sizes, num_features, team_match_one_hot)
        self.tables = self._load_tables(split_sizes, num_features, team_match_one_hot)


    def _load_tables(self, split_sizes, num_features, team_match_one_hot=False):
        
        if self.data_set == 'debug':

            inds0 = np.array([ [0,0],[0,2],[0,3],[1,0],[1,2],[2,1],[2,3],[3,1],[3,2],[3,3],[4,0],[4,1],[4,2],[5,1],[5,3] ])
            vals0 = np.array([ 1,0,1,1,0,0,1,0,1,1,0,1,0,1,1 ])
            shape0 = np.array([6,4])
            table0 = Table(0, inds0, vals0, shape0, [1.0, 0.0, 0.0], num_features=num_features)


            inds1 = np.array([ [0,0],[0,2],[1,3],[2,0],[2,4],[3,3],[3,4],[4,1],[5,1],[5,2] ])
            vals1 = np.array([ 3,2,0,-3,-1,0,1,-2,2,-2 ])
            shape1 = np.array([6,5])
            table1 = Table(1, inds1, vals1, shape1, split_sizes, num_features=num_features, one_hot=team_match_one_hot, split_mode='by_col')

            # return table0, table1
            return {'team_player':table0, 'team_match':table1}

        elif self.data_set == 'soccer':

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

            for i, match in enumerate(matches):
                match_id, home_team_id, away_team_id, home_team_goal, away_team_goal = match[0:5]
                home_team_id = team_map[home_team_id]
                away_team_id = team_map[away_team_id]
                home_player_ids = match[5:16]
                away_player_ids = match[16:27]
                
                match_map[match_id] = i
                match_id = i

                team_match.append( [home_team_id, match_id, home_team_goal-away_team_goal] )
                team_match.append( [away_team_id, match_id, away_team_goal-home_team_goal] )

                for player_id in home_player_ids:
                    if player_id != None:
                        team_player.append( [home_team_id, player_map[player_id], 1] )

                for player_id in away_player_ids:
                    if player_id != None:
                        team_player.append( [away_team_id, player_map[player_id], 1] )
            n_matches = i + 1

            team_match.sort()
            team_player.sort()

            temp = []
            for key, group in groupby(team_player, key=lambda x: x[0:2]):
                key.append(sum([g[2] for g in group]))
                temp.append(key)
            
            team_player = temp

            team_player = np.array(team_player)
            team_match = np.array(team_match)

            # table0 = Table(0, team_player[:,0:2], team_player[:,2], np.array([n_teams, n_players]), 1. - vl/tr, vl/tr, .0)
            table0 = Table(0, team_player[:,0:2], team_player[:,2], np.array([n_teams, n_players]), [1.0, 0.0, 0.0], num_features)
            table1 = Table(1, team_match[:,0:2], team_match[:,2], np.array([n_teams, n_matches]), split_sizes, num_features, team_match_one_hot, split_mode='by_col')

            conn.close()

            # return table0, table1
            return {'team_player': table0, 'team_match': table1}


class ToyDataLoader:

    def __init__(self, sizes, sparsity, split_sizes, num_features, embedding_size, min_observed, embeddings=None, observed=None, predict_unobserved=False, predict=None):
        self.sizes = sizes
        self._n_students = sizes[0]
        self._n_courses = sizes[1]
        self._n_profs = sizes[2]
        self._split_sizes = split_sizes
        self._num_features = num_features
        self._embedding_size = embedding_size
        self._min_observed = min_observed
        if observed is not None:
            self._observed = observed
        else:
            self._observed = {'sc':None, 'sp':None, 'cp':None}
        if predict is not None:
            self._predict = predict
        else:
            self._predict = {'sc':None, 'sp':None, 'cp':None}
        self._predict_unobserved = predict_unobserved

        assert embedding_size == 2, 'Currently only embedding size of 2 is supported'

        self._sparsity = sparsity

        if embeddings is None:
            self.embeddings = self._make_embeddings()
        else:
            self.embeddings = embeddings

        self.tables = self._make_tables()

    def _make_embeddings(self):

        student_embeds = gaussian_embeddings(embedding_size=self._embedding_size, n_embeddings=self._n_students)
        course_embeds = gaussian_embeddings(embedding_size=self._embedding_size, n_embeddings=self._n_courses)
        prof_embeds = gaussian_embeddings(embedding_size=self._embedding_size, n_embeddings=self._n_profs)

        return {'student': student_embeds, 'course': course_embeds, 'prof':prof_embeds}


    def _make_tables(self):
        """Construct all tables in our toy problem, populated with random data"""
        embeds_s = self.embeddings['student']
        embeds_c = self.embeddings['course']
        embeds_p = self.embeddings['prof']

        # if self._observed is None:
        #     table_sc = self._make_table(embeds_s, embeds_c, tid=0)
        #     table_sp = self._make_table(embeds_s, embeds_p, tid=1)
        #     table_cp = self._make_table(embeds_c, embeds_p, tid=2)
        # else:
        table_sc = self._make_table(embeds_s, embeds_c, tid=0, observed=self._observed['sc'], predict=self._predict['sc'])
        table_sp = self._make_table(embeds_s, embeds_p, tid=1, observed=self._observed['sp'], predict=self._predict['sp'])
        table_cp = self._make_table(embeds_c, embeds_p, tid=2, observed=self._observed['cp'], predict=self._predict['cp'])

        return {'student_course': table_sc, 'student_prof':table_sp, 'course_prof':table_cp}


    def _make_table(self, row_embeds, col_embeds, tid, observed=None, predict=None):

        assert row_embeds.shape[1] == col_embeds.shape[1]

        n_rows = row_embeds.shape[0]
        n_cols = col_embeds.shape[0]
        embeds_size = row_embeds.shape[1]
        shape = (n_rows, n_cols)
        tab = np.zeros((n_rows, n_cols))

        if self._embedding_size == 2:
            for i in range(n_rows):
                for j in range(n_cols):
                    tab[i, j] = self._product_data(row_embeds[i, :], col_embeds[j, :])
                    # tab[i, j] = self._quadratic_data(row_embeds[i, :], col_embeds[j, :], alpha)
        else:
            raise Exception('invalid embedding size')

        if observed is None:
            observed = choose_observed(tid, shape, self._sparsity, min_observed=self._min_observed)

        if self._predict_unobserved:
            inds = np.array(np.nonzero(np.ones((n_rows, n_cols)))).T
            vals = tab.flatten()
            split = 1. - observed.flatten()

        elif predict is not None:
            in_vals = observed + predict
            inds = np.array(np.nonzero(in_vals)).T
            vals = tab.flatten()[in_vals.flatten() > 0]
            split = np.zeros_like(in_vals.flatten())
            split[predict.flatten() == 1] = 1
            split = split[in_vals.flatten() != 0]
        else:
            inds = np.array(np.nonzero(observed)).T
            vals = tab.flatten()[observed.flatten() == 1]
            split = None

        return Table(tid, inds, vals, shape, num_features=self._num_features, split_sizes=self._split_sizes, split=split, embeddings=self.embeddings)





    def _product_data(self, row_embed, col_embed):
        """Produce data values by a (weighted) inner product of row and column embeddings."""
        return np.dot(row_embed, col_embed)

    def _quadratic_data(self, row_embed, col_embed, alpha):
        """Produce data values as quadratic form: row_embed^T alpha col_embed."""
        return row_embed[None, :] @ alpha @ col_embed[:, None]

