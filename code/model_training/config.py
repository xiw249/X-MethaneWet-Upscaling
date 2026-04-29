import torch

class Config:
    def __init__(self, batch_size=128, enc_in=15, n_hidden=8, ):
        self.device = "cuda" 
        self.batch_size = batch_size
        self.enc_in = enc_in
        self.hidden_size = n_hidden
        self.batch_first = True 
        self.num_layers = 3
        self.input_size = 15
        self.output_size = 1
        self.seq_len = 365
        self.initial_forget_bias = 0
        self.input_size_dyn = 5
        self.input_size_stat = 10
        self.dropout = 0.2
        self.concat_static = False
        self.no_static = False
        self.num_channels = [1,1,1]
        self.kernel_size = 5
        self.d_model = 8
        self.e_layers = 3
        self.d_layers = 1
        self.n_heads = 4
        self.factor = 3
        self.embed = "time"
        self.freq = 'd'
        self.d_ff = 4*self.d_model
        self.c_out = 1
        self.activation = 'relu'
        self.output_attention = False
        self.pred_len = 365
        self.task_name = 'long_term_forecast'
        self.patch_channels = 15
        self.point_input_size = 15
        self.patch_size = 3