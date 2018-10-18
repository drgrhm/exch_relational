import numpy as np
import tensorflow as tf
from util import *


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
        # num_vals = tf.shape(inds)[0]
        num_vals = tf.shape(inds, out_type=tf.int64)[0]

        num_vals = tf.cast(num_vals, tf.int64)
        begin_ = tf.cast([0,0], tf.int64)
        end_ = tf.cast([num_vals, 2], tf.int64)
        strides_ = tf.cast([units_out, 1], tf.int64)

        num_vals = tf.cast(num_vals, tf.int64)
        units_out = tf.cast(units_out, tf.int64)
        temp_shape_ = tf.cast([-1,units_out], tf.int64)

        if axis == 0:
            # temp_inds = tf.strided_slice(inds, begin=[0,0], end=[num_vals,2], strides=[units_out,1])
            temp_inds = tf.strided_slice(inds, begin=begin_, end=end_, strides=strides_)
            temp_inds = tf.slice(temp_inds, begin=[0,1], size=[-1,1])
            new_vals = tf.cast(tf.gather_nd(tf.reshape(marginal, shape=temp_shape_), temp_inds), tf.float64)
            vals = tf.reshape(table_values, shape=temp_shape_) + new_vals
            vals = tf.reshape(vals, shape=[-1])
        elif axis == 1:
            # temp_inds = tf.strided_slice(inds, begin=[0,0], end=[num_vals,2], strides=[units_out,1])
            temp_inds = tf.strided_slice(inds, begin=begin_, end=end_, strides=strides_)
            temp_inds = tf.slice(temp_inds, begin=[0,0], size=[-1,1])
            new_vals = tf.cast(tf.gather_nd(tf.reshape(marginal, shape=temp_shape_), temp_inds), tf.float64)
            vals = tf.reshape(table_values, shape=temp_shape_) + new_vals
            vals = tf.reshape(vals, shape=[-1])
        elif axis == None:
            vals = tf.reshape(table_values, shape=temp_shape_)
            vals = tf.reshape(tf.add(vals, marginal), shape=[-1])
        return vals


    # Given a list of indices in [N,M], return 'equivalent' indices in [N,M,units]
    def expand_tensor_indices(self, indices, units):

        # num_vals = tf.shape(indices)[0]
        num_vals = tf.cast(tf.shape(indices)[0], tf.int64)
        shape_ = tf.cast([-1,1], tf.int64)
        units_ = tf.cast(units, tf.int64)
        multiples_ = tf.cast([num_vals], tf.int64)

        inds_exp = tf.reshape(tf.tile(tf.cast(tf.range(units_, dtype=tf.int64), tf.float64), multiples=multiples_), shape=shape_) # expand dimension of mask indices

        indices = tf.cast(indices, dtype=tf.float64) # cast so computation can be done on gpu
        inds = tf.tile(indices, multiples=tf.cast([units,1], tf.int64)) # duplicate indices units times
        inds = tf.reshape(inds, shape=tf.cast([units_, num_vals, 2], tf.int64))
        inds = tf.reshape(tf.transpose(inds, perm=tf.cast([1,0,2], tf.int64)), shape=tf.cast([-1,2], tf.int64))
        inds = tf.concat((inds, inds_exp), axis=1)
        inds = tf.cast(inds, dtype=tf.int64)
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
            count = tf.unsorted_segment_sum(tf.ones_like(inds, dtype=tf.float64), inds, num_segments)
            count = tf.cast(tf.expand_dims(count, axis=1), dtype=tf.float64)
            if keep_dims:
                count = tf.reshape(count, shape=tf.cast([1,-1,1], tf.int64))
        elif axis == 1:
            inds = table['indices'][:,0]
            num_segments = table['shape'][0]
            count = tf.unsorted_segment_sum(tf.ones_like(inds, dtype=tf.float64), inds, num_segments)
            count = tf.cast(tf.expand_dims(count, axis=1), dtype=tf.float64)
            if keep_dims:
                count = tf.reshape(count, shape=tf.cast([-1,1,1], tf.int64))
        elif axis is None:
            inds = table['indices']
            count = tf.cast(tf.shape(inds)[0], dtype=tf.float64)
            if keep_dims:
                count = tf.reshape(count, shape=tf.cast([1,1,1], tf.int64))           
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

            params['table_0'] = {}
            params['table_1'] = {}
            params['table_2'] = {}

            tables_out = {}

            marg_0_10 = self.marginalize_table(tables['table_0'], pool_mode=self.pool_mode, axis=0, keep_dims=True) # 1 x n_businesses x K
            marg_0_01 = self.marginalize_table(tables['table_0'], pool_mode=self.pool_mode, axis=1, keep_dims=True) # n_users x 1 x K
            marg_0_11 = self.marginalize_table(tables['table_0'], pool_mode=self.pool_mode, axis=None, keep_dims=True) # 1 x 1 x K

            marg_1 = self.marginalize_table(tables['table_1'], pool_mode=self.pool_mode, axis=0, keep_dims=False)
            marg_1_10 = tf.expand_dims(marg_1, axis=0)  # 1 x n_users x K
            # marg_1_01 = tf.expand_dims(marg_1, axis=1)  # n_users x 1 x K
            marg_1_11 = self.marginalize_table(tables['table_1'], pool_mode=self.pool_mode, axis=None, keep_dims=True) # 1 x 1 x K

            marg_2_10 = self.marginalize_table(tables['table_2'], pool_mode=self.pool_mode, axis=0, keep_dims=True) # 1 x n_categories x K
            marg_2_01 = self.marginalize_table(tables['table_2'], pool_mode=self.pool_mode, axis=1, keep_dims=True) # n_businesses x 1 x K
            marg_2_11 = self.marginalize_table(tables['table_2'], pool_mode=self.pool_mode, axis=None, keep_dims=True) # 1 x 1 x K


            params['theta_b'] = tf.get_variable(name=('theta_b'), shape=[units_out], dtype=tf.float64)


            params['table_0']['theta_00'] = tf.get_variable(name=('table_0_theta_00'), shape=[units_in, units_out], dtype=tf.float64)            
            params['table_0']['theta_10'] = tf.get_variable(name=('table_0_theta_10'), shape=[units_in, units_out], dtype=tf.float64)
            params['table_0']['theta_01'] = tf.get_variable(name=('table_0_theta_01'), shape=[units_in, units_out], dtype=tf.float64)
            params['table_0']['theta_11'] = tf.get_variable(name=('table_0_theta_11'), shape=[units_in, units_out], dtype=tf.float64)

            params['table_0']['theta_1x0_10'] = tf.get_variable(name=('table_0_theta_1x0_10'), shape=[units_in, units_out], dtype=tf.float64)    # Interaction with table 1
            params['table_0']['theta_1x0_11'] = tf.get_variable(name=('table_0_theta_1x0_11'), shape=[units_in, units_out], dtype=tf.float64)
            
            params['table_0']['theta_2x0_01'] = tf.get_variable(name=('table_0_theta_2x0_01'), shape=[units_in, units_out], dtype=tf.float64)    # Interaction with table 2
            params['table_0']['theta_2x0_11'] = tf.get_variable(name=('table_0_theta_2x0_11'), shape=[units_in, units_out], dtype=tf.float64)            

            vals_0 = tf.reshape(tf.matmul(tf.reshape(tables['table_0']['values'], [-1, units_in]),  params['table_0']['theta_00']), [-1])

            vals_0_10 = tf.tensordot(marg_0_10, params['table_0']['theta_10'], axes=1)
            vals_0 = self.broadcast_add_marginal(vals_0, vals_0_10, tables['table_0']['indices'], axis=0)

            vals_0_01 = tf.tensordot(marg_0_01, params['table_0']['theta_01'], axes=1)
            vals_0 = self.broadcast_add_marginal(vals_0, vals_0_01, tables['table_0']['indices'], axis=1)

            vals_0_11 = tf.tensordot(marg_0_11, params['table_0']['theta_11'], axes=1)
            vals_0 = self.broadcast_add_marginal(vals_0, vals_0_11, tables['table_0']['indices'], axis=None)

            vals_1x0_10 = tf.tensordot(marg_1_10, params['table_0']['theta_1x0_10'], axes=1)
            vals_0 = self.broadcast_add_marginal(vals_0, vals_1x0_10, tables['table_0']['indices'], axis=0)

            vals_1x0_11 = tf.tensordot(marg_1_11, params['table_0']['theta_1x0_11'], axes=1)
            vals_0 = self.broadcast_add_marginal(vals_0, vals_1x0_11, tables['table_0']['indices'], axis=None)

            vals_2x0_01 = tf.tensordot(marg_2_01, params['table_0']['theta_2x0_01'], axes=1)
            vals_0 = self.broadcast_add_marginal(vals_0, vals_2x0_01, tables['table_0']['indices'], axis=0)     

            vals_2x0_11 = tf.tensordot(marg_2_11, params['table_0']['theta_2x0_11'], axes=1)
            vals_0 = self.broadcast_add_marginal(vals_0, vals_2x0_11, tables['table_0']['indices'], axis=None)     

            vals_0 = tf.reshape(vals_0, [-1, units_out]) + params['theta_b']
            vals_0 = tf.reshape(vals_0, [-1])



            params['table_1']['theta_00'] = tf.get_variable(name=('table_1_theta_00'), shape=[units_in, units_out], dtype=tf.float64)
            params['table_1']['theta_10'] = tf.get_variable(name=('table_1_theta_10'), shape=[units_in, units_out], dtype=tf.float64)
            # params['table_1']['theta_01'] = tf.get_variable(name=('table_1_theta_01'), shape=[units_in, units_out])
            # params['table_1']['theta_01'] = params['table_1']['theta_10']
            params['table_1']['theta_11'] = tf.get_variable(name=('table_1_theta_11'), shape=[units_in, units_out], dtype=tf.float64)            

            params['table_1']['theta_0x1_01'] = tf.get_variable(name=('table_1_theta_0x1_01'), shape=[units_in, units_out], dtype=tf.float64)    # Interaction with table 0
            params['table_1']['theta_0x1_11'] = tf.get_variable(name=('table_1_theta_0x1_11'), shape=[units_in, units_out], dtype=tf.float64)

            vals_1 = tf.reshape(tf.matmul(tf.reshape(tables['table_1']['values'], [-1, units_in]),  params['table_1']['theta_00']), [-1])

            vals_1_10 = tf.tensordot(marg_1_10, params['table_1']['theta_10'], axes=1)
            vals_1 = self.broadcast_add_marginal(vals_1, vals_1_10, tables['table_1']['indices'], axis=0)

            vals_1_11 = tf.tensordot(marg_1_11, params['table_1']['theta_11'], axes=1)
            vals_1 = self.broadcast_add_marginal(vals_1, vals_1_11, tables['table_1']['indices'], axis=None)

            vals_0x1_01 = tf.tensordot(marg_0_01, params['table_1']['theta_0x1_01'], axes=1)
            vals_1 = self.broadcast_add_marginal(vals_1, vals_0x1_01, tables['table_1']['indices'], axis=1)

            vals_0x1_11 = tf.tensordot(marg_0_11, params['table_1']['theta_0x1_11'], axes=1)
            vals_1 = self.broadcast_add_marginal(vals_1, vals_0x1_11, tables['table_1']['indices'], axis=None)

            vals_1 = tf.reshape(vals_1, [-1, units_out]) + params['theta_b']
            vals_1 = tf.reshape(vals_1, [-1])


            params['table_2']['theta_00'] = tf.get_variable(name=('table_2_theta_00'), shape=[units_in, units_out], dtype=tf.float64)            
            params['table_2']['theta_10'] = tf.get_variable(name=('table_2_theta_10'), shape=[units_in, units_out], dtype=tf.float64)
            params['table_2']['theta_01'] = tf.get_variable(name=('table_2_theta_01'), shape=[units_in, units_out], dtype=tf.float64)
            params['table_2']['theta_11'] = tf.get_variable(name=('table_2_theta_11'), shape=[units_in, units_out], dtype=tf.float64)

            params['table_2']['theta_0x2_10'] = tf.get_variable(name=('table_2_theta_0x2_10'), shape=[units_in, units_out], dtype=tf.float64)    # Interaction with table 0
            params['table_2']['theta_0x2_11'] = tf.get_variable(name=('table_2_theta_0x2_11'), shape=[units_in, units_out], dtype=tf.float64)            

            vals_2 = tf.reshape(tf.matmul(tf.reshape(tables['table_2']['values'], [-1, units_in]),  params['table_2']['theta_00']), [-1])
            
            vals_2_10 = tf.tensordot(marg_2_10, params['table_2']['theta_10'], axes=1)
            vals_2 = self.broadcast_add_marginal(vals_2, vals_2_10, tables['table_2']['indices'], axis=0)

            vals_2_01 = tf.tensordot(marg_2_01, params['table_2']['theta_01'], axes=1)
            vals_2 = self.broadcast_add_marginal(vals_2, vals_2_01, tables['table_2']['indices'], axis=1)

            vals_2_11 = tf.tensordot(marg_2_11, params['table_2']['theta_11'], axes=1)
            vals_2 = self.broadcast_add_marginal(vals_2, vals_2_11, tables['table_2']['indices'], axis=None)

            vals_0x2_10 = tf.tensordot(marg_0_10, params['table_2']['theta_0x2_10'], axes=1)
            vals_2 = self.broadcast_add_marginal(vals_2, vals_0x2_10, tables['table_2']['indices'], axis=0)

            vals_0x2_11 = tf.tensordot(marg_0_11, params['table_2']['theta_0x2_11'], axes=1)
            vals_2 = self.broadcast_add_marginal(vals_2, vals_0x2_11, tables['table_2']['indices'], axis=None)

            vals_2 = tf.reshape(vals_2, [-1,units_out]) + params['theta_b']
            vals_2 = tf.reshape(vals_2, [-1])



            if self.activation is not None:
                vals_0 = self.activation(vals_0)
                vals_1 = self.activation(vals_1)
                vals_2 = self.activation(vals_2)
                    
            if self.skip_connections and units_in == units_out:
                vals_0 = vals_0 + tables['table_0']['values']
                vals_1 = vals_1 + tables['table_1']['values']
                vals_2 = vals_2 + tables['table_2']['values']


            tables_out['table_0'] = {'indices':tables['table_0']['indices'], 'values':vals_0, 'shape':tables['table_0']['shape']}
            tables_out['table_1'] = {'indices':tables['table_1']['indices'], 'values':vals_1, 'shape':tables['table_1']['shape']}
            tables_out['table_2'] = {'indices':tables['table_2']['indices'], 'values':vals_2, 'shape':tables['table_2']['shape']}

            return tables_out
        


class FeatureDropoutLayer(Layer):

    def __init__(self, units, **kwargs):
        Layer.__init__(self, units)
        self.dropout_rate = kwargs['dropout_rate']
        self.scope = kwargs['scope']


    def get_output(self, tables, reuse=None, is_training=True):

        vals = tf.reshape(tables['table_0']['values'], [-1, self.units_in])
        vals = tf.layers.dropout(vals, rate=self.dropout_rate, training=is_training)
        vals = tf.reshape(vals, [-1])

        # for t, table in tables:           
        #     vals = tf.reshape(table['values'], [-1, self.units_in])
        #     vals = tf.layers.dropout(vals, rate=self.dropout_rate, training=is_training)
        #     vals = tf.reshape(team_player_vals, [-1])

        tables_out = {}
        tables_out['table_0'] = {'indices':tables['table_0']['indices'],'values':vals, 'shape':tables['table_0']['shape']}   
        tables_out['table_1'] = tables['table_1']
        tables_out['table_2'] = tables['table_2']

        return tables_out




