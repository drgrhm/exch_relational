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
        # self.output_embeddings = kwargs.get('output_embeddings', False)
        self.embedding_size = kwargs.get('embedding_size', 2)
        self.scope = kwargs['scope']
        self.params = {}


    def get_output(self, tables, reuse=None, is_training=True):
        units_in = self.units_in
        units_out = self.units_out
        params = self.params

        with tf.variable_scope(self.scope, initializer=tf.random_normal_initializer(0, .01), reuse=reuse):

            student_course = tables['student_course']
            student_prof = tables['student_prof']
            course_prof = tables['course_prof']

            student_embeds = student_course.get('row_embeds', None)
            course_embeds = student_course.get('col_embeds', None)
            prof_embeds = student_prof.get('col_embeds', None)


            if student_embeds is not None and course_embeds is not None and prof_embeds is not None:

                ## student_course
                params['theta_sc_row_embed'] = tf.get_variable(name='theta_sc_row_embed', shape=[units_in, units_out], trainable=is_training)
                params['theta_sc_col_embed'] = tf.get_variable(name='theta_sc_col_embed', shape=[units_in, units_out], trainable=is_training)
                params['theta_sc_row_embed_pool'] = tf.get_variable(name='theta_sc_row_embed_pool', shape=[units_in, units_out], trainable=is_training)
                params['theta_sc_col_embed_pool'] = tf.get_variable(name='theta_sc_col_embed_pool', shape=[units_in, units_out], trainable=is_training)

                student_course_vals_row = tf.tensordot(student_embeds, params['theta_sc_row_embed'], axes=1) # [N_students x 1 x units_out]
                student_course_vals_col = tf.tensordot(course_embeds, params['theta_sc_col_embed'], axes=1)  # [1 x N_courses x units_out]
                student_course_vals_row_pool = tf.tensordot(tf.reduce_sum(student_embeds, axis=0, keep_dims=True), params['theta_sc_row_embed_pool'], axes=1)
                student_course_vals_col_pool = tf.tensordot(tf.reduce_sum(course_embeds, axis=1, keep_dims=True), params['theta_sc_col_embed_pool'], axes=1)

                student_course_vals = tf.zeros([tf.shape(student_course['indices'])[0] * units_out], tf.float32)
                student_course_vals = self.broadcast_add_marginal(student_course_vals, student_course_vals_row, student_course['indices'], axis=1)
                student_course_vals = self.broadcast_add_marginal(student_course_vals, student_course_vals_col, student_course['indices'], axis=0)
                student_course_vals = self.broadcast_add_marginal(student_course_vals, student_course_vals_row_pool, student_course['indices'], axis=None)
                student_course_vals = self.broadcast_add_marginal(student_course_vals, student_course_vals_col_pool, student_course['indices'], axis=None)

                ## student_prof
                params['theta_sp_row_embed'] = tf.get_variable(name='theta_sp_row_embed', shape=[units_in, units_out], trainable=is_training)
                params['theta_sp_col_embed'] = tf.get_variable(name='theta_sp_col_embed', shape=[units_in, units_out], trainable=is_training)
                params['theta_sp_row_embed_pool'] = tf.get_variable(name='theta_sp_row_embed_pool', shape=[units_in, units_out], trainable=is_training)
                params['theta_sp_col_embed_pool'] = tf.get_variable(name='theta_sp_col_embed_pool', shape=[units_in, units_out], trainable=is_training)

                student_prof_vals_row = tf.tensordot(student_embeds, params['theta_sp_row_embed'], axes=1)  # [N_students x 1 x units_out]
                student_prof_vals_col = tf.tensordot(prof_embeds, params['theta_sp_col_embed'], axes=1)  # [1 x N_profs x units_out]
                student_prof_vals_row_pool = tf.tensordot(tf.reduce_sum(student_embeds, axis=0, keep_dims=True), params['theta_sp_row_embed_pool'], axes=1)
                student_prof_vals_col_pool = tf.tensordot(tf.reduce_sum(prof_embeds, axis=1, keep_dims=True), params['theta_sp_col_embed_pool'], axes=1)

                student_prof_vals = tf.zeros([tf.shape(student_prof['indices'])[0] * units_out], tf.float32)
                student_prof_vals = self.broadcast_add_marginal(student_prof_vals, student_prof_vals_row, student_prof['indices'], axis=1)
                student_prof_vals = self.broadcast_add_marginal(student_prof_vals, student_prof_vals_col, student_prof['indices'], axis=0)
                student_prof_vals = self.broadcast_add_marginal(student_prof_vals, student_prof_vals_row_pool, student_prof['indices'], axis=None)
                student_prof_vals = self.broadcast_add_marginal(student_prof_vals, student_prof_vals_col_pool, student_prof['indices'], axis=None)


                ## course_prof
                params['theta_cp_row_embed'] = tf.get_variable(name='theta_cp_row_embed', shape=[units_in, units_out], trainable=is_training)
                params['theta_cp_col_embed'] = tf.get_variable(name='theta_cp_col_embed', shape=[units_in, units_out], trainable=is_training)
                params['theta_cp_row_embed_pool'] = tf.get_variable(name='theta_cp_row_embed_pool', shape=[units_in, units_out], trainable=is_training)
                params['theta_cp_col_embed_pool'] = tf.get_variable(name='theta_cp_col_embed_pool', shape=[units_in, units_out], trainable=is_training)

                course_embeds = tf.expand_dims(tf.squeeze(course_embeds), 1)
                course_prof_vals_row = tf.tensordot(course_embeds, params['theta_cp_row_embed'], axes=1)  # [N_courses x 1 x units_out]
                course_prof_vals_col = tf.tensordot(prof_embeds, params['theta_cp_col_embed'], axes=1)  # [1 x N_profs x units_out]
                course_prof_vals_row_pool = tf.tensordot(tf.reduce_sum(course_embeds, axis=0, keep_dims=True), params['theta_cp_row_embed_pool'], axes=1)
                course_prof_vals_col_pool = tf.tensordot(tf.reduce_sum(prof_embeds, axis=1, keep_dims=True), params['theta_cp_col_embed_pool'], axes=1)

                course_prof_vals = tf.zeros([tf.shape(course_prof['indices'])[0] * units_out], tf.float32)
                course_prof_vals = self.broadcast_add_marginal(course_prof_vals, course_prof_vals_row, course_prof['indices'], axis=1)
                course_prof_vals = self.broadcast_add_marginal(course_prof_vals, course_prof_vals_col, course_prof['indices'], axis=0)
                course_prof_vals = self.broadcast_add_marginal(course_prof_vals, course_prof_vals_row_pool, course_prof['indices'], axis=None)
                course_prof_vals = self.broadcast_add_marginal(course_prof_vals, course_prof_vals_col_pool, course_prof['indices'], axis=None)

            else:
                ## Params
                params['theta_sc_sel'] = tf.get_variable(name='theta_sc_sel', shape=[units_in, units_out], trainable=is_training)
                params['theta_sc_row'] = tf.get_variable(name='theta_sc_row', shape=[units_in, units_out], trainable=is_training)
                params['theta_sc_col'] = tf.get_variable(name='theta_sc_col', shape=[units_in, units_out], trainable=is_training)
                params['theta_sc_all'] = tf.get_variable(name='theta_sc_all', shape=[units_in, units_out], trainable=is_training)
                params['theta_sc_sp_col'] = tf.get_variable(name='theta_sc_sp_col', shape=[units_in, units_out], trainable=is_training)
                params['theta_sc_sp_all'] = tf.get_variable(name='theta_sc_sp_all', shape=[units_in, units_out], trainable=is_training)
                params['theta_sc_cp_col'] = tf.get_variable(name='theta_sc_cp_col', shape=[units_in, units_out], trainable=is_training)
                params['theta_sc_cp_all'] = tf.get_variable(name='theta_sc_cp_all', shape=[units_in, units_out], trainable=is_training)

                params['theta_sp_sel'] = tf.get_variable(name='theta_sp_sel', shape=[units_in, units_out], trainable=is_training)
                params['theta_sp_row'] = tf.get_variable(name='theta_sp_row', shape=[units_in, units_out], trainable=is_training)
                params['theta_sp_col'] = tf.get_variable(name='theta_sp_col', shape=[units_in, units_out], trainable=is_training)
                params['theta_sp_all'] = tf.get_variable(name='theta_sp_all', shape=[units_in, units_out], trainable=is_training)
                params['theta_sp_sc_col'] = tf.get_variable(name='theta_sp_sc_col', shape=[units_in, units_out], trainable=is_training)
                params['theta_sp_sc_all'] = tf.get_variable(name='theta_sp_sc_all', shape=[units_in, units_out], trainable=is_training)
                params['theta_sp_cp_row'] = tf.get_variable(name='theta_sp_cp_row', shape=[units_in, units_out], trainable=is_training)
                params['theta_sp_cp_all'] = tf.get_variable(name='theta_sp_cp_all', shape=[units_in, units_out], trainable=is_training)

                params['theta_cp_sel'] = tf.get_variable(name='theta_cp_sel', shape=[units_in, units_out], trainable=is_training)
                params['theta_cp_row'] = tf.get_variable(name='theta_cp_row', shape=[units_in, units_out], trainable=is_training)
                params['theta_cp_col'] = tf.get_variable(name='theta_cp_col', shape=[units_in, units_out], trainable=is_training)
                params['theta_cp_all'] = tf.get_variable(name='theta_cp_all', shape=[units_in, units_out], trainable=is_training)
                params['theta_cp_sp_row'] = tf.get_variable(name='theta_cp_sp_row', shape=[units_in, units_out], trainable=is_training)
                params['theta_cp_sp_all'] = tf.get_variable(name='theta_cp_sp_all', shape=[units_in, units_out], trainable=is_training)
                params['theta_cp_sc_row'] = tf.get_variable(name='theta_cp_sc_row', shape=[units_in, units_out], trainable=is_training)
                params['theta_cp_sc_all'] = tf.get_variable(name='theta_cp_sc_all', shape=[units_in, units_out], trainable=is_training)


                ## Marginalizations
                student_course_marg_row = self.marginalize_table(student_course, pool_mode=self.pool_mode, axis=0, keep_dims=True)  # student_course table, marginalized over students [1 x N_courses x units_in]
                student_course_marg_col = self.marginalize_table(student_course, pool_mode=self.pool_mode, axis=1, keep_dims=True)  # student_course table, marginalized over courses [N_students x 1 x units_in]
                student_course_marg_all = self.marginalize_table(student_course, pool_mode=self.pool_mode, axis=None, keep_dims=True)  # student_course table, marginalized over both [1 x 1 x units_in]

                student_prof_marg_row = self.marginalize_table(student_prof, pool_mode=self.pool_mode, axis=0, keep_dims=True)  # student_prof table, marginalized over students [1 x N_profs x units_in]
                student_prof_marg_col = self.marginalize_table(student_prof, pool_mode=self.pool_mode, axis=1, keep_dims=True)  # student_prof table, marginalized over profs [N_students x 1 x units_in]
                student_prof_marg_all = self.marginalize_table(student_prof, pool_mode=self.pool_mode, axis=None, keep_dims=True)  # student_prof table, marginalized over both [1 x 1 x units_in]

                course_prof_marg_row = self.marginalize_table(course_prof, pool_mode=self.pool_mode, axis=0, keep_dims=True)  # course_prof table, marginalized over courses [1 x N_profs x units_in]
                course_prof_marg_col = self.marginalize_table(course_prof, pool_mode=self.pool_mode, axis=1, keep_dims=True)  # course_prof table, marginalized over profs [N_courses x 1 x units_in]
                course_prof_marg_all = self.marginalize_table(course_prof, pool_mode=self.pool_mode, axis=None, keep_dims=True)  # course_prof table, marginalized over both [1 x 1 x units_in]


                ## Tensor products
                student_course_vals_row = tf.tensordot(student_course_marg_row, params['theta_sc_row'], axes=1)  # [1 x N_courses x units_out]
                student_course_vals_col = tf.tensordot(student_course_marg_col, params['theta_sc_col'], axes=1)  # [N_students x 1 x units_out]
                student_course_vals_all = tf.tensordot(student_course_marg_all, params['theta_sc_all'], axes=1)  # [1 x 1 x units_out]
                sc_sp_vals_mix_col = tf.tensordot(student_prof_marg_col, params['theta_sc_sp_col'], axes=1)  # [N_students x 1 x units_out]
                sc_sp_vals_mix_all = tf.tensordot(student_prof_marg_all, params['theta_sc_sp_all'], axes=1)  # [1 x 1 x units_out]
                sc_cp_vals_mix_row = tf.expand_dims(tf.squeeze(tf.tensordot(course_prof_marg_col, params['theta_sc_cp_col'], axes=1)), 0) # [1 x N_courses x units_out]
                sc_cp_vals_mix_all = tf.tensordot(course_prof_marg_all, params['theta_sc_cp_all'], axes=1)  # [1 x 1 x units_out]

                student_prof_vals_row = tf.tensordot(student_prof_marg_row, params['theta_sp_row'], axes=1)  # [1 x N_profs x units_out]
                student_prof_vals_col = tf.tensordot(student_prof_marg_col, params['theta_sp_col'], axes=1)  # [N_students x 1 x units_out]
                student_prof_vals_all = tf.tensordot(student_prof_marg_all, params['theta_sp_all'], axes=1)  # [1 x 1 x units_out]
                sp_sc_vals_mix_col = tf.tensordot(student_course_marg_col, params['theta_sp_sc_col'], axes=1)  # [N_students x 1 x units_out]
                sp_sc_vals_mix_all = tf.tensordot(student_course_marg_all, params['theta_sp_sc_all'], axes=1)  # [1 x 1 x units_out]
                sp_cp_vals_mix_row = tf.tensordot(course_prof_marg_row, params['theta_sp_cp_row'], axes=1)  # [1 x N_profs x units_out]
                sp_cp_vals_mix_all = tf.tensordot(course_prof_marg_all, params['theta_sp_cp_all'], axes=1)  # [1 x 1 x units_out]

                course_prof_vals_row = tf.tensordot(course_prof_marg_row, params['theta_cp_row'], axes=1)  # [1 x N_profs x units_out]
                course_prof_vals_col = tf.tensordot(course_prof_marg_col, params['theta_cp_col'], axes=1)  # [N_courses x 1 x units_out]
                course_prof_vals_all = tf.tensordot(course_prof_marg_all, params['theta_cp_all'], axes=1)  # [1 x 1 x units_out]
                cp_sp_vals_mix_row = tf.tensordot(student_prof_marg_row, params['theta_cp_sp_row'], axes=1)  # [1 x N_profs x units_out]
                cp_sp_vals_mix_all = tf.tensordot(student_prof_marg_all, params['theta_cp_sp_all'], axes=1)  # [1 x 1 x units_out]
                cp_sc_vals_mix_col = tf.expand_dims(tf.squeeze(tf.tensordot(student_course_marg_row, params['theta_cp_sc_row'], axes=1)), 1)  # [N_courses x 1 x units_out]
                cp_sc_vals_mix_all = tf.tensordot(student_course_marg_all, params['theta_cp_sc_all'], axes=1)  # [1 x 1 x units_out]


                ## Broadcast add
                student_course_vals = tf.reshape(tf.matmul(tf.reshape(student_course['values'], shape=[-1, units_in]), params['theta_sc_sel']), [-1])
                student_course_vals = self.broadcast_add_marginal(student_course_vals, student_course_vals_row, student_course['indices'], axis=0)
                student_course_vals = self.broadcast_add_marginal(student_course_vals, student_course_vals_col, student_course['indices'], axis=1)
                student_course_vals = self.broadcast_add_marginal(student_course_vals, student_course_vals_all, student_course['indices'], axis=None)
                student_course_vals = self.broadcast_add_marginal(student_course_vals, sc_sp_vals_mix_col, student_course['indices'], axis=1)
                student_course_vals = self.broadcast_add_marginal(student_course_vals, sc_sp_vals_mix_all, student_course['indices'], axis=None)
                student_course_vals = self.broadcast_add_marginal(student_course_vals, sc_cp_vals_mix_row, student_course['indices'], axis=0)
                student_course_vals = self.broadcast_add_marginal(student_course_vals, sc_cp_vals_mix_all, student_course['indices'], axis=None)

                student_prof_vals = tf.reshape(tf.matmul(tf.reshape(student_prof['values'], shape=[-1, units_in]), params['theta_sp_sel']), [-1])
                student_prof_vals = self.broadcast_add_marginal(student_prof_vals, student_prof_vals_row, student_prof['indices'], axis=0)
                student_prof_vals = self.broadcast_add_marginal(student_prof_vals, student_prof_vals_col, student_prof['indices'], axis=1)
                student_prof_vals = self.broadcast_add_marginal(student_prof_vals, student_prof_vals_all, student_prof['indices'], axis=None)
                student_prof_vals = self.broadcast_add_marginal(student_prof_vals, sp_sc_vals_mix_col, student_prof['indices'], axis=1)
                student_prof_vals = self.broadcast_add_marginal(student_prof_vals, sp_sc_vals_mix_all, student_prof['indices'], axis=None)
                student_prof_vals = self.broadcast_add_marginal(student_prof_vals, sp_cp_vals_mix_row, student_prof['indices'], axis=0)
                student_prof_vals = self.broadcast_add_marginal(student_prof_vals, sp_cp_vals_mix_all, student_prof['indices'], axis=None)

                course_prof_vals = tf.reshape(tf.matmul(tf.reshape(course_prof['values'], shape=[-1, units_in]), params['theta_cp_sel']), [-1])
                course_prof_vals = self.broadcast_add_marginal(course_prof_vals, course_prof_vals_row, course_prof['indices'], axis=0)
                course_prof_vals = self.broadcast_add_marginal(course_prof_vals, course_prof_vals_col, course_prof['indices'], axis=1)
                course_prof_vals = self.broadcast_add_marginal(course_prof_vals, course_prof_vals_all, course_prof['indices'], axis=None)
                course_prof_vals = self.broadcast_add_marginal(course_prof_vals, cp_sp_vals_mix_row, course_prof['indices'], axis=0)
                course_prof_vals = self.broadcast_add_marginal(course_prof_vals, cp_sp_vals_mix_all, course_prof['indices'], axis=None)
                course_prof_vals = self.broadcast_add_marginal(course_prof_vals, cp_sc_vals_mix_col, course_prof['indices'], axis=1)
                course_prof_vals = self.broadcast_add_marginal(course_prof_vals, cp_sc_vals_mix_all, course_prof['indices'], axis=None)



            if self.activation is not None:
                student_course_vals = self.activation(student_course_vals)
                student_prof_vals = self.activation(student_prof_vals)
                course_prof_vals = self.activation(course_prof_vals)

            if self.skip_connections and units_in == units_out:
                student_course_vals = student_course_vals + student_course['values']
                student_prof_vals = student_prof_vals + student_prof['values']
                course_prof_vals = course_prof_vals + course_prof['values']

            student_course_out = {'indices': student_course['indices'],
                                  'values': student_course_vals,
                                  'shape': student_course['shape']}

            student_prof_out = {'indices':student_prof['indices'],
                                'values':student_prof_vals,
                                'shape':student_prof['shape']}

            course_prof_out = {'indices':course_prof['indices'],
                               'values':course_prof_vals,
                               'shape':course_prof['shape']}



            return {'student_course':student_course_out, 'student_prof':student_prof_out, 'course_prof':course_prof_out}




class FeatureDropoutLayer(Layer):

    def __init__(self, units, **kwargs):
        Layer.__init__(self, units)
        self.dropout_rate = kwargs['dropout_rate']
        self.scope = kwargs['scope']


    def get_output(self, tables, reuse=None, is_training=True):

        student_course = tables['student_course']
        student_prof = tables['student_prof']
        course_prof = tables['course_prof']

        # Todo: Dropout same channels in each table?

        student_course_vals = tf.layers.dropout(tf.reshape(student_course['values'], [-1, self.units_in]), noise_shape=[1, self.units_in], rate=self.dropout_rate, training=is_training)
        student_course_vals = tf.reshape(student_course_vals, [-1])

        student_prof_vals = tf.layers.dropout(tf.reshape(student_prof['values'], [-1, self.units_in]), noise_shape=[1, self.units_in], rate=self.dropout_rate, training=is_training)
        student_prof_vals = tf.reshape(student_prof_vals, [-1])

        course_prof_vals = tf.layers.dropout(tf.reshape(course_prof['values'], [-1, self.units_in]), noise_shape=[1, self.units_in], rate=self.dropout_rate, training=is_training)
        course_prof_vals = tf.reshape(course_prof_vals, [-1])

        student_course_out = {'indices':student_course['indices'],
                              'values':student_course_vals,
                              'shape':student_course['shape'],
                              'row_embeds':student_course.get('row_embeds', None),
                              'col_embeds':student_course.get('col_embeds', None)}

        student_prof_out = {'indices':student_prof['indices'],
                            'values':student_prof_vals,
                            'shape':student_prof['shape'],
                            'row_embeds':student_prof.get('row_embeds', None),
                            'col_embeds':student_prof.get('col_embeds', None)}

        course_prof_out = {'indices':course_prof['indices'],
                           'values':course_prof_vals,
                           'shape':course_prof['shape'],
                           'row_embeds':course_prof.get('row_embeds', None),
                           'col_embeds':course_prof.get('col_embeds', None)}

        return {'student_course':student_course_out, 'student_prof':student_prof_out, 'course_prof':course_prof_out}

    

class PoolingLayer(Layer):

    def __init__(self, units, **kwargs):
        Layer.__init__(self, units)
        self._pool_mode = kwargs['pool_mode']
        self.scope = kwargs['scope']


    def get_output(self, tables, reuse=None, is_training=True):

        student_course = tables['student_course']
        student_prof = tables['student_prof']
        course_prof = tables['course_prof']

        student_embeds_sc = self.marginalize_table(student_course, pool_mode=self._pool_mode, axis=1, keep_dims=True)  # student_course table, marginalized over courses [N_students x 1 x units_in]
        course_embeds_sc = self.marginalize_table(student_course, pool_mode=self._pool_mode, axis=0, keep_dims=True)  # student_course table, marginalized over students [1 x N_courses x units_in]

        student_embeds_sp = self.marginalize_table(student_prof, pool_mode=self._pool_mode, axis=1, keep_dims=True)  # student_prof table, marginalized over profs [N_students x 1 x units_in]
        prof_embeds_sp = self.marginalize_table(student_prof, pool_mode=self._pool_mode, axis=0, keep_dims=True)  # student_prof table, marginalized over students [1 x N_profs x units_in]

        course_embeds_cp = self.marginalize_table(course_prof, pool_mode=self._pool_mode, axis=1, keep_dims=True)  # course_prof table, marginalized over profs [N_courses x 1 x units_in]
        prof_embeds_cp = self.marginalize_table(course_prof, pool_mode=self._pool_mode, axis=0, keep_dims=True)  # course_prof table, marginalized over courses [1 x N_profs x units_in]

        student_embeds = student_embeds_sc + student_embeds_sp
        course_embeds = course_embeds_sc + tf.expand_dims(tf.squeeze(course_embeds_cp), 0)
        prof_embeds = prof_embeds_sp + prof_embeds_cp

        tables['student_course']['row_embeds'] = student_embeds
        tables['student_course']['col_embeds'] = course_embeds

        tables['student_prof']['row_embeds'] = student_embeds
        tables['student_prof']['col_embeds'] = prof_embeds

        tables['course_prof']['row_embeds'] = course_embeds
        tables['course_prof']['col_embeds'] = prof_embeds

        return tables




