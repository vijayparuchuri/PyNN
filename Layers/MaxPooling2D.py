import numpy as np


class Layer_MaxPooling2D:
    def __init__(self, kernel_size=(2, 2), stride=None, padding=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.k_h, self.k_w = self.kernel_size
        self.s_h, self.s_w = self.stride if self.stride is not None else self.kernel_size
        
        
    def create_padding(self, input):
        input = input.transpose(2, 0, 1)
        if isinstance(self.padding, tuple) and len(self.padding)==2:
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
        padded_input = np.pad(input, mode='constant', pad_width=((0, 0), (self.p_h_top, self.p_h_bottom), (self.p_w_left, self.p_w_right)))
        return padded_input
    
    def forward(self, input):
        self.input = np.array([self.create_padding(input) if self.padding is not None else input.transpose(2, 0, 1) for input in input])
        self.max_indices = []
        self.output= np.stack([self.single_input_pooling(single_input) for single_input in self.input])
        
    def single_input_pooling(self, single_input):
        windows = np.lib.stride_tricks.sliding_window_view(single_input, self.kernel_size, (1, 2))[:, ::self.s_h, ::self.s_w]
        windows_reshaped = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], -1)
        self.max_indices.append(np.argmax(windows_reshaped, axis=3))
        output = windows.max(axis=(3, 4)).transpose(1, 2, 0)
        return output
    
    def backward(self, dvalues):
        self.dinputs = np.stack([self.calculate_single_dinput(single_input, single_dvalue, max_index) for single_input, single_dvalue, max_index in zip(self.input, dvalues, self.max_indices)])
        
    def calculate_single_dinput(self, single_input, single_dvalue, max_index):
        dinput = np.zeros_like(single_input)
        single_dvalue = single_dvalue.transpose(2, 0, 1)
        c, h_out, w_out = single_dvalue.shape
        max_pos = np.unravel_index(max_index, (self.k_h, self.k_w))

        row_indices = np.arange(h_out)[:, None] * self.s_h + max_pos[0]
        col_indices = np.arange(w_out)[None, :] * self.s_w + max_pos[1]

        row_indices = np.clip(row_indices, 0, self.input.shape[1] - 1)
        col_indices = np.clip(col_indices, 0, self.input.shape[2] - 1)

        for i in range(c):
            dinput[i, row_indices[i], col_indices[i]] += single_dvalue[i]

        if self.padding:
            dinput = dinput[:, self.p_h_top:dinput.shape[1] - self.p_h_bottom,
                                        self.p_w_left:dinput.shape[2] - self.p_w_right]
        dinput = dinput.transpose(1, 2, 0)
        return dinput