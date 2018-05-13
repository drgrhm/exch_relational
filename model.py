import numpy as np
import tensorflow as tf

class Model:

    def __init__(self, **kwargs):
        
        pool_mode = kwargs.get('pool_mode', None)
        dropout_rate = kwargs.get('dropout_rate', None)
        layers = kwargs['layers']
        self.num_layers = len(layers)
        self.layers = {}
        units_in = kwargs['units_in']

        for l, layer in enumerate(layers):
            name = 'layer_' + str(l)
            units_out = layer['units_out']
            activation = layer.get('activation', None)
            skip_connections = layer.get('skip_connections', False)
            self.layers[name] = layer['type'](units=[units_in, units_out], pool_mode=pool_mode, dropout_rate=dropout_rate, activation=activation, skip_connections=skip_connections, scope=name)
            units_in = units_out


    def get_output(self, team_player, team_match, reuse=None, is_training=True):
        tp, tm = team_player, team_match
        for l in range(self.num_layers):
            name = 'layer_' + str(l)
            tp, tm = self.layers[name].get_output(tp, tm, reuse=reuse, is_training=is_training)
        return tp, tm
        


