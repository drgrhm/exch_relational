import numpy as np
import tensorflow as tf

class Model:

    def __init__(self, **kwargs):
        
        pool_mode = kwargs.get('pool_mode', None)
        dropout_rate = kwargs.get('dropout_rate', None)
        side_info = kwargs.get('side_info', True)
        layers = kwargs['layers']
        # variational = kwargs['variational']
        self.num_layers = len(layers)
        self.layers = {}
        units_in = kwargs['units_in']

        for l, layer in enumerate(layers):
            name = 'layer_' + str(l)
            units_out = layer['units_out']
            activation = layer.get('activation', None)
            save_embeddings = layer.get('save_embeddings', False)
            skip_connections = layer.get('skip_connections', False)
            self.layers[name] = layer['type'](units=[units_in, units_out],
                                              pool_mode=pool_mode,
                                              dropout_rate=dropout_rate,
                                              activation=activation,
                                              skip_connections=skip_connections,
                                              side_info=side_info,
                                              save_embeddings=save_embeddings,
                                              # variational=variational,
                                              scope=name)
            units_in = units_out


    def get_output(self, tables, reuse=None, is_training=True):
        tables_out = tables
        for l in range(self.num_layers):
            name = 'layer_' + str(l)
            tables_out = self.layers[name].get_output(tables_out, reuse=reuse, is_training=is_training)
        return tables_out
        


