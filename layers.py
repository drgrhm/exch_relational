import numpy as np
import tensorflow as tf


class Layer:

    def __init__(self, units):
        self.units_in = units[0]
        self.units_out = units[1]


    ## table_values is a sparse tensor of shape [N,M,K], marginal is dense of shape [1,M,K], [N,1,K], or [1,1,K] (according to axis).
    ## Broadcast add y onto the sparse coordinates of x_sp.
    ## Produces a sparse tensor with the same shape as x_sp, and non-zero values corresponding to those of x_sp
    def broadcast_add_marginal(self, table_values, marginal, indices, axis=None):

        # TODO refactor 2 cases into 1 
        units_out = self.units_out
        inds = self.expand_tensor_indices(indices, units_out)
        num_vals = tf.shape(inds)[0]
        if axis == 0:
            temp_inds = tf.strided_slice(inds, begin=[0,0], end=[num_vals,2], strides=[units_out,1])
            temp_inds = tf.slice(temp_inds, begin=[0,1], size=[-1,1])
            new_vals = tf.cast(tf.gather_nd(tf.reshape(marginal, shape=[-1,units_out]), temp_inds), tf.float32)
            vals = tf.reshape(table_values, shape=[-1,units_out]) + new_vals
            vals = tf.reshape(vals, shape=[num_vals])
        elif axis == 1:
            temp_inds = tf.strided_slice(inds, begin=[0,0], end=[num_vals,2], strides=[units_out,1])
            temp_inds = tf.slice(temp_inds, begin=[0,0], size=[-1,1])
            new_vals = tf.cast(tf.gather_nd(tf.reshape(marginal, shape=[-1,units_out]), temp_inds), tf.float32)
            vals = tf.reshape(table_values, shape=[-1,units_out]) + new_vals
            vals = tf.reshape(vals, shape=[num_vals])
        elif axis == None:
            vals = tf.reshape(table_values, shape=[-1,units_out])
            vals = tf.reshape(tf.add(vals, marginal), shape=[-1])
        return vals


    # Given a list of indices in [N,M], return 'equivalent' indices in [N,M,units]
    def expand_tensor_indices(self, indices, units):

        num_vals = tf.shape(indices)[0]
        inds_exp = tf.reshape(tf.tile(tf.range(units, dtype=tf.float32), multiples=[num_vals]), shape=[-1, 1]) # expand dimension of mask indices
        indices = tf.cast(indices, dtype=tf.float32) # cast so computation can be done on gpu
        inds = tf.tile(indices, multiples=[units,1]) # duplicate indices units times
        inds = tf.reshape(inds, shape=[units, num_vals, 2])
        inds = tf.reshape(tf.transpose(inds, perm=[1,0,2]), shape=[-1,2])
        inds = tf.concat((inds, inds_exp), axis=1)
        inds = tf.cast(inds, dtype=tf.int32)
        return inds


    ## TODO For higher dimensional data this will have to change 
    def marginalize_table(self, table, pool_mode, axis=None, keep_dims=False):
        vals = tf.reshape(table['values'], shape=[-1, self.units_in])
        eps = 1e-10        
        out = None
        if axis == 0:
            inds = table['indices'][:,1]
            num_segments = table['shape'][1]
            if pool_mode == 'mean':
                count = self.count_entries(table, axis, keep_dims=keep_dims) + eps
                out = tf.unsorted_segment_sum(vals, inds, num_segments=num_segments)                
                if keep_dims:
                    out = tf.expand_dims(out, axis=0)
                out = out / count                
            elif pool_mode == 'max':
                out = tf.unsorted_segment_max(vals, inds, num_segments=num_segments)
                if keep_dims:
                    out = tf.expand_dims(out, axis=0)
        elif axis == 1:
            inds = table['indices'][:,0]            
            num_segments = table['shape'][0]
            if pool_mode == 'mean':
                count = self.count_entries(table, axis, keep_dims=keep_dims) + eps
                out = tf.unsorted_segment_sum(vals, inds, num_segments=num_segments)
                if keep_dims:
                    out = tf.expand_dims(out, axis=1)
                out = out / count
            elif pool_mode == 'max':
                out = tf.unsorted_segment_max(vals, inds, num_segments=num_segments)
                if keep_dims:
                    out = tf.expand_dims(out, axis=1)
        elif axis == None:
            if 'mean' in pool_mode:
                count = self.count_entries(table, axis, keep_dims=keep_dims)
                out = tf.reduce_sum(vals, axis=0, keep_dims=keep_dims)
                out = out / count
            elif 'max' in pool_mode:
                out = tf.reduce_max(vals, axis=0, keep_dims=keep_dims)
                if keep_dims:
                    out = tf.expand_dims(out, axis=1)
        else:
            raise ValueError("Unknown axis: %s" % axis)

        if pool_mode == 'max': ## Hack to avoid large negative number. How to deal with max over no entries? 
            out = tf.where(tf.greater(out, -200000000), out, tf.zeros_like(out))
        return out


    def count_entries(self, table, axis=None, keep_dims=False):
        """Count non-zero entires of table along axis."""
        count = None
        if axis == 0:
            inds = table['indices'][:,1]
            num_segments = table['shape'][1]
            count = tf.unsorted_segment_sum(tf.ones_like(inds, dtype=tf.float32), inds, num_segments)
            count = tf.cast(tf.expand_dims(count, axis=1), dtype=tf.float32)
            if keep_dims:
                count = tf.reshape(count, shape=[1,-1,1])
        elif axis == 1:
            inds = table['indices'][:,0]
            num_segments = table['shape'][0]
            count = tf.unsorted_segment_sum(tf.ones_like(inds, dtype=tf.float32), inds, num_segments)
            count = tf.cast(tf.expand_dims(count, axis=1), dtype=tf.float32)
            if keep_dims:
                count = tf.reshape(count, shape=[-1,1,1])
        elif axis is None:
            inds = table['indices']
            count = tf.cast(tf.shape(inds)[0], dtype=tf.float32)
            if keep_dims:
                count = tf.reshape(count, shape=[1,1,1])            
        return count




class ExchangeableLayer(Layer):

    def __init__(self, units, **kwargs):
        
        Layer.__init__(self, units)

        self.pool_mode = kwargs['pool_mode']
        self.activation = kwargs['activation']
        self.skip_connections = kwargs.get('skip_connections', False)
        self.regularization_rate = kwargs['regularization_rate']
        self.scope = kwargs['scope']
        self.params = {}


    def get_output(self, tables, reuse=None, is_training=True):
        units_in = self.units_in
        units_out = self.units_out
        params = self.params

        with tf.variable_scope(self.scope, 
                               initializer=tf.random_normal_initializer(0, .01),
                               # regularizer=tf.contrib.keras.regularizers.l2(self.regularization_rate),
                               reuse=reuse):

            tables_out = {}
            
            for t, table in tables.items():
                params[t] = {}

                params[t]['theta_00'] = tf.get_variable(name=(t + '_theta_00'), shape=[units_in, units_out])
                params[t]['theta_01'] = tf.get_variable(name=(t + '_theta_01'), shape=[units_in, units_out])
                params[t]['theta_10'] = tf.get_variable(name=(t + '_theta_10'), shape=[units_in, units_out])
                params[t]['theta_11'] = tf.get_variable(name=(t + '_theta_11'), shape=[units_in, units_out])

                params[t]['theta_x0'] = tf.get_variable(name=(t + '_theta_x0'), shape=[units_in, units_out])
                params[t]['theta_x1'] = tf.get_variable(name=(t + '_theta_x1'), shape=[units_in, units_out])

                vals = tf.reshape(tf.matmul(tf.reshape(tables[t]['values'], [-1, units_in]),  params[t]['theta_00']), [-1])

                marg_01 = self.marginalize_table(tables[t], pool_mode=self.pool_mode, axis=0, keep_dims=True)
                vals_01 = tf.tensordot(marg_01, params[t]['theta_01'], axes=1)
                vals = self.broadcast_add_marginal(vals, vals_01, tables[t]['indices'], axis=0)

                marg_10 = self.marginalize_table(tables[t], pool_mode=self.pool_mode, axis=1, keep_dims=True)
                vals_10 = tf.tensordot(marg_10, params[t]['theta_10'], axes=1)
                vals = self.broadcast_add_marginal(vals, vals_10, tables[t]['indices'], axis=1)

                marg_11 = self.marginalize_table(tables[t], pool_mode=self.pool_mode, axis=None, keep_dims=True)
                vals_11 = tf.tensordot(marg_11, params[t]['theta_11'], axes=1)
                vals = self.broadcast_add_marginal(vals, vals_11, tables[t]['indices'], axis=None)

                
                print(t, table)




                if self.activation is not None:
                    vals = self.activation(vals)
                    
                if self.skip_connections and units_in == units_out:
                    vals = vals + tables[t]['values']


                tables_out[t] = {'indices':tables[t]['indices'], 'values':vals, 'shape':tables[t]['shape']}
                    


            # theta = tf.get_variable(name='thetaaa', shape=[units_in, units_out])
            # params['thetaaa'] = theta

            # # table_0_vals = tf.matmul(tf.reshape(tables['table_0']['values'], [-1, units_in]), theta)
            # table_0_vals = tf.reshape(tables['table_0']['values'], [-1, units_in]) * theta
            # table_0_vals = tf.reshape(table_0_vals, [-1])


            # tables_out = {'table_0':{'indices':tables['table_0']['indices'], 'values':table_0_vals, 'shape':tables['table_0']['shape']},
            #               'table_1':tables['table_1'],
            #               'table_2':tables['table_2']
            #               }
            


            return tables_out



        # units_in = self.units_in
        # units_out = self.units_out
        # params = self.params

        # with tf.variable_scope(self.scope, 
        #                        initializer=tf.random_normal_initializer(0, .01),
        #                        regularizer=tf.contrib.keras.regularizers.l2(self.regularization_rate),
        #                        reuse=reuse):

        #     params['theta_team_player_00'] = tf.get_variable(name='theta_team_player_00', shape=[units_in, units_out])
        #     params['theta_team_player_01'] = tf.get_variable(name='theta_team_player_01', shape=[units_in, units_out])
        #     params['theta_team_player_10'] = tf.get_variable(name='theta_team_player_10', shape=[units_in, units_out])
        #     params['theta_team_player_11'] = tf.get_variable(name='theta_team_player_11', shape=[units_in, units_out])

        #     params['theta_team_player_inter_0'] = tf.get_variable(name='theta_team_player_inter_0', shape=[units_in, units_out])
        #     params['theta_team_player_inter_1'] = tf.get_variable(name='theta_team_player_inter_1', shape=[units_in, units_out])


        #     params['theta_team_match_00'] = tf.get_variable(name='theta_team_match_00', shape=[units_in, units_out])
        #     params['theta_team_match_01'] = tf.get_variable(name='theta_team_match_01', shape=[units_in, units_out])
        #     params['theta_team_match_10'] = tf.get_variable(name='theta_team_match_10', shape=[units_in, units_out])
        #     params['theta_team_match_11'] = tf.get_variable(name='theta_team_match_11', shape=[units_in, units_out])            

        #     params['theta_team_match_inter_0'] = tf.get_variable(name='theta_team_match_inter_0', shape=[units_in, units_out])
        #     params['theta_team_match_inter_1'] = tf.get_variable(name='theta_team_match_inter_1', shape=[units_in, units_out])


        #     team_player_vals = tf.reshape(tf.matmul(tf.reshape(team_player['values'], shape=[-1,units_in]), params['theta_team_player_00']), [-1])

        #     team_player_marg_01 = self.marginalize_table(team_player, pool_mode=self.pool_mode, axis=0, keep_dims=True) # team-player table, marginalized over teams [1 x N_players x units]       
        #     team_player_vals_01 = tf.tensordot(team_player_marg_01, params['theta_team_player_01'], axes=1) #[1 x N_players x K]
        #     team_player_vals = self.broadcast_add_marginal(team_player_vals, team_player_vals_01, team_player['indices'], axis=0)

        #     team_player_marg_10 = self.marginalize_table(team_player, pool_mode=self.pool_mode, axis=1, keep_dims=True) # team-player table, marginalized over players [N_teams x 1 x units]
        #     team_player_vals_10 = tf.tensordot(team_player_marg_10, params['theta_team_player_10'], axes=1) #[N_teams x 1 x K]
        #     team_player_vals = self.broadcast_add_marginal(team_player_vals, team_player_vals_10, team_player['indices'], axis=1)        

        #     team_player_marg_11 = self.marginalize_table(team_player, pool_mode=self.pool_mode, axis=None, keep_dims=True) # team-player table, marginalized over both [1 x 1 x units]
        #     team_player_vals_11 = tf.tensordot(team_player_marg_11, params['theta_team_player_11'], axes=1) #[1 x 1 x K]
        #     team_player_vals = self.broadcast_add_marginal(team_player_vals, team_player_vals_11, team_player['indices'], axis=None)
            
        #     team_match_marg_10 = self.marginalize_table(team_match, pool_mode=self.pool_mode, axis=1, keep_dims=True) # team-match table, marginalized over matches [N_teams x 1 x units]
        #     team_player_vals_inter_0 = tf.tensordot(team_match_marg_10, params['theta_team_player_inter_0'], axes=1) #[N_teams x 1 x K]
        #     team_player_vals = self.broadcast_add_marginal(team_player_vals, team_player_vals_inter_0, team_player['indices'], axis=1)

        #     team_match_marg_11 = self.marginalize_table(team_match, pool_mode=self.pool_mode, axis=None, keep_dims=True) # team-match table, marginalized over both [1 x 1 x units]
        #     team_player_vals_inter_1 = tf.tensordot(team_match_marg_11, params['theta_team_player_inter_1'], axes=1) #[1 x 1 x K]
        #     team_player_vals = self.broadcast_add_marginal(team_player_vals, team_player_vals_inter_1, team_player['indices'], axis=None)




        #     team_match_vals = tf.reshape(tf.matmul(tf.reshape(team_match['values'], shape=[-1,units_in]), params['theta_team_match_00']), [-1])
            
        #     team_match_marg_01 = self.marginalize_table(team_match, pool_mode=self.pool_mode, axis=0, keep_dims=True) # team-match table, marginalized over teams [1 x N_matches x units]       
        #     team_match_vals_01 = tf.tensordot(team_match_marg_01, params['theta_team_match_01'], axes=1) #[1 x 1 x K]
        #     team_match_vals = self.broadcast_add_marginal(team_match_vals, team_match_vals_01, team_match['indices'], axis=0)

        #     team_match_marg_10 = self.marginalize_table(team_match, pool_mode=self.pool_mode, axis=1, keep_dims=True) # team-match table, marginalized over matches [N_teams x 1 x units]
        #     team_match_vals_10 = tf.tensordot(team_match_marg_10, params['theta_team_match_10'], axes=1) #[N_teams x 1 x K]
        #     team_match_vals = self.broadcast_add_marginal(team_match_vals, team_match_vals_10, team_match['indices'], axis=1)

        #     team_match_marg_11 = self.marginalize_table(team_match, pool_mode=self.pool_mode, axis=None, keep_dims=True) # team-match table, marginalized over both [1 x 1 x units]
        #     team_match_vals_11 = tf.tensordot(team_match_marg_11, params['theta_team_match_11'], axes=1) #[1 x 1 x K]
        #     team_match_vals = self.broadcast_add_marginal(team_match_vals, team_match_vals_11, team_match['indices'], axis=None)

        #     team_player_marg_10 = self.marginalize_table(team_player, pool_mode=self.pool_mode, axis=1, keep_dims=True) # team-match table, marginalized over matches [N_teams x 1 x units]
        #     team_match_vals_inter_0 = tf.tensordot(team_player_marg_10, params['theta_team_match_inter_0'], axes=1) #[N_teams x 1 x K]
        #     team_match_vals = self.broadcast_add_marginal(team_match_vals, team_match_vals_inter_0, team_match['indices'], axis=1)

        #     team_player_marg_11 = self.marginalize_table(team_player, pool_mode=self.pool_mode, axis=None, keep_dims=True) # team-match table, marginalized over both [1 x 1 x units]
        #     team_match_vals_inter_1 = tf.tensordot(team_player_marg_11, params['theta_team_match_inter_1'], axes=1) #[1 x 1 x K]
        #     team_match_vals = self.broadcast_add_marginal(team_match_vals, team_match_vals_inter_1, team_match['indices'], axis=None)
            

        #     if self.activation is not None:
        #         team_player_vals = self.activation(team_player_vals)
        #         team_match_vals = self.activation(team_match_vals)


        #     if self.skip_connections and units_in == units_out:
        #         team_player_vals = team_player_vals + team_player['values'],
        #         team_match_vals = team_match_vals + team_match['values'],


        #     team_player_out = {'indices':team_player['indices'], 'values':team_player_vals, 'shape':team_player['shape']}
        #     team_match_out = {'indices':team_match['indices'], 'values':team_match_vals, 'shape':team_match['shape']}

            
        #     return team_player_out, team_match_out



class FeatureDropoutLayer(Layer):

    def __init__(self, units, **kwargs):
        Layer.__init__(self, units)
        self.dropout_rate = kwargs['dropout_rate']
        self.scope = kwargs['scope']


    def get_output(self, team_player, team_match, reuse=None, is_training=True):

        team_player_vals = tf.layers.dropout(tf.reshape(team_player['values'], [-1, self.units_in]), rate=self.dropout_rate, training=is_training)
        team_player_vals = tf.reshape(team_player_vals, [-1])

        team_match_vals = tf.layers.dropout(tf.reshape(team_match['values'], [-1, self.units_in]), rate=self.dropout_rate, training=is_training)
        team_match_vals = tf.reshape(team_match_vals, [-1])


        team_player_out = {'indices':team_player['indices'], 'values':team_player_vals, 'shape':team_player['shape']}
        # team_player_out = team_player_out
        team_match_out = {'indices':team_match['indices'], 'values':team_match_vals, 'shape':team_match['shape']}
        
        return team_player_out, team_match_out

    







