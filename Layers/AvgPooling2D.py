import numpy as np

class Layer_AvgPool2D:
    def __init__(self, kernel_size = (2, 2), stride =None, padding=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.k_h, self.k_w = self.kernel_size
        self.s_h, self.s_w = self.stride if self.stride is not None else self.kernel_size
        
    def create_padding(self, input):
        self.input = input.transpose(2, 0, 1)
        if isinstance(self.padding, tuple) and len(self.padding) == 2:
            self.p_h_top, self.p_h_bottom = self.padding
            self.p_w_left, self.p_w_right = self.padding
            
        elif self.padding == 'same':
            if self.k_h % 2 and self.k_w % 2 == 1:
                self.p_h = (self.k_h - 1) // 2
                self.p_w = (self.k_w - 1) // 2
                self.p_h_top, self.p_h_bottom = self.p_h, self.p_h
                self.p_w_left, self.p_w_right = self.p_w, self.p_w
            else:
                self.p_h_top = (self.k_h - 1) // 2
                self.p_h_bottom = self.k_h // 2
                self.p_w_left = (self.k_w - 1) // 2
                self.p_w_right = self.k_w // 2
        padded_input = np.pad(self.input, mode='constant', pad_width=((0, 0), (self.p_h_top, self.p_h_bottom), (self.p_w_left, self.p_w_right)))
        return padded_input
    
    def forward(self, input):
        self.input = self.create_padding(input) if self.padding is not None else input.transpose(2, 0, 1)
        
        windows = np.lib.stride_tricks.sliding_window_view(self.input, self.kernel_size, axis=(1, 2))[:, ::self.s_h, ::self.s_w]
        self.wins = windows
        self.output = windows.mean(axis=(3, 4)).transpose(1, 2, 0)
    
    def backward(self, dvalues):
        dvalues_reshaped = dvalues.transpose(2, 0, 1)[:, np.newaxis, np.newaxis, :, :]
        pool_size = 2 * 2
        dinputs = np.repeat(np.repeat(dvalues_reshaped, 2, axis=1), 2, axis=2) / pool_size
        dinputs_padded = np.zeros_like(self.input)

        h_out, w_out, c = dvalues.shape
        i, j = np.meshgrid(np.arange(h_out), np.arange(w_out), indexing ='ij')
        h_start = i * 2
        w_start = j * 2
        
        di, dj = np.meshgrid(np.arange(2), np.arange(2), indexing='ij')
        h_indices = (h_start[:, :, np.newaxis, np.newaxis] + di).reshape(-1)
        w_indices = (w_start[:, :, np.newaxis, np.newaxis] + dj).reshape(-1)
        c_indices = np.arange(c)[:, np.newaxis].repeat(h_out * w_out * 2 * 2, axis=1)
        
        np.add.at(dinputs_padded, (c_indices, h_indices, w_indices), dinputs.reshape(c, -1))
        
        if self.padding is not None:
            self.dinputs = dinputs_padded[:, self.p_h_top:-self.p_h_bottom or None, self.p_w_left:-self.p_w_right or None]
        else:
            self.dinputs = dinputs_padded
        
        self.dinputs = self.dinputs.transpose(1, 2, 0)
            