# import numpy as np
import sqlite3
from itertools import groupby
from util import *
from table import Table, ToyTable


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

    def __init__(self, sizes, embedding_size, sparsity, alphas):
        self.sizes = sizes
        self._n_students = sizes[0]
        self._n_courses = sizes[1]
        self._n_profs = sizes[2]
        self._embedding_size = embedding_size

        assert embedding_size <= 2, 'Currently only embedding size of 1 or 2 are supported'

        self._sparsity = sparsity
        self._alphas = alphas

        self.embeddings = self._make_embeddings()
        self.tables = self._make_tables()

    def _make_embeddings(self):

        student_embeds = self._gaussian_embeddings(size=self._n_students)
        course_embeds = self._gaussian_embeddings(size=self._n_courses)
        prof_embeds = self._gaussian_embeddings(size=self._n_profs)

        return {'student': student_embeds, 'course': course_embeds, 'prof':prof_embeds}


    def _make_tables(self):
        """Construct all tables in our toy problem, populated with random data"""
        se = self.embeddings['student']
        ce = self.embeddings['course']
        pe = self.embeddings['prof']

        table_sc = self._make_table(se, ce, 0)
        table_sp = self._make_table(se, pe, 1)
        # table_cp = self._make_table(ce, pe, 2)

        return {'student_course': table_sc, 'student_prof':table_sp}



    def _make_table(self, row_embeds, col_embeds, tid):

        n_rows = row_embeds.shape[0]
        n_cols = col_embeds.shape[0]
        a = self._alphas

        tab = np.zeros((n_rows, n_cols))

        if self._embedding_size == 1:
            for i in range(n_rows):
                for j in range(n_cols):
                    tab[i, j] = a[0] * row_embeds[i] * col_embeds[j]

        elif self._embedding_size == 2:
            for i in range(n_rows):
                for j in range(n_cols):
                    tab[i, j] = a[0] * row_embeds[i, 0] * col_embeds[j, 0] + \
                                a[1] * row_embeds[i, 0] * col_embeds[j, 1] + \
                                a[2] * row_embeds[i, 1] * col_embeds[j, 0] + \
                                a[3] * row_embeds[i, 1] * col_embeds[j, 1]

        else:
            raise Exception('invalid embedding size')

        shape = (n_rows, n_cols)
        observed = self._choose_observed(shape)
        inds = np.array(np.nonzero(observed)).T
        vals = tab.flatten()[observed.flatten() == 1]

        return ToyTable(tid, inds, vals, shape, num_features=self._embedding_size, embeddings=self.embeddings)



    def _choose_observed(self, shape):
        """Which entries of the matrix to consider as observed. """
        return np.random.choice([0,1], shape, p=(1-self._sparsity, self._sparsity))



    def _gaussian_embeddings(self, size):
        """Multivariate Gaussian feature embeddings."""
        means = np.random.normal(0, 10, size=self._embedding_size)
        stds = np.random.uniform(1, 10, size=self._embedding_size)
        embeds = np.random.multivariate_normal(means, np.diag(stds), size=size)
        return embeds



    # def plot_embeds(self, embeds, predicts, title, plot_name, sort=False):
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
    #
    #
    # def plot_feature(self, embeds, predicts, title, plot_name, sort=True):
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
    #     plt.xlabel('item')
    #     plt.ylabel('feature')
    #     plt.legend(('embeddings', 'predictions'))
    #     plt.show()
    #     plt.savefig(plot_name + '.pdf', bbox_inches='tight')
    #     plt.clf()

