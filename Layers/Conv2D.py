import numpy as np
from scipy.signal import convolve2d


class Layer_Conv2D:
    def __init__(self, input_shape, kernel_size, filters = 1 ,padding='valid', stride=(1, 1)):
        self.k_h, self.k_w = kernel_size
        self.input_height, self.input_width , self.input_channels = input_shape
        self.filters = filters
        self.weights = 0.01 * np.random.randn(self.filters, self.input_channels, self.k_h, self.k_w)
        self.padding = padding
        self.s_h, self.s_w = stride
        self.biases = np.zeros(self.filters)
    
    def forward(self, input):
        self.input = input
        self.output = np.stack([self.process_single_input(single_input) for single_input in input], 0)
        
        
    def process_single_input(self, single_input):
        single_input = single_input.transpose(2, 0, 1)
        output = np.stack([self.calculate_convolutions(single_input, self.weights[i], self.biases[i]) for i in range(self.filters)], 0)
        return output.transpose(1, 2, 0)
    
    def calculate_convolutions(self, inputs, kernels, bias):
        output = [convolve2d(input, np.flipud(np.fliplr(weight)), mode=self.padding, fillvalue=0) for input, weight in zip(inputs, kernels)]
        output = np.sum(output, axis=0) + bias
        if self.s_h > 1 or self.s_w > 1:
            output = output[::self.s_h, ::self.s_w]
        return output
        
        
    def create_padding(self, input):
        if self.padding == 'same':
            if self.k_h % 2 == 1 and self.k_h % 2 == 1:
                self.p_h = (self.k_h - 1) // 2
                self.p_w = (self.k_w - 1) // 2
                self.p_h_top, self.p_h_bottom = self.p_h, self.p_h
                self.p_w_left, self.p_w_right = self.p_w, self.p_w
            else:
                self.p_h_top = (self.k_h - 1) // 2
                self.p_h_bottom = self.k_h // 2
                self.p_w_left = (self.k_w - 1) // 2
                self.p_w_right = self.k_w // 2
                
            input = np.array([np.pad(input, pad_width=((self.p_h_top, self.p_h_bottom), 
                                          (self.p_w_left, self.p_w_right)), mode='constant') for input in input])
        return input
    
    def backward(self, dvalues):
        self.dweights=np.zeros_like(self.weights)
        if self.padding == 'valid' and (self.s_h > 1 or self.s_w >1):
            self.dinputs = np.zeros_like(self.input)
        elif self.padding == 'same' and (self.s_h >1 or self.s_w > 1):
            self.dinputs = np.zeros_like(self.input)
        else:
            self.dinputs = np.zeros((self.input.shape[0], dvalues.shape[1], dvalues.shape[2], self.input.shape[3])) 
        self.dbiases = np.zeros_like(self.biases)
        
        for i, (single_inputs, single_dvalues) in enumerate(zip(self.input, dvalues)):
            single_dvalue = single_dvalues.transpose(2, 0, 1)    
            single_input = self.create_padding(single_inputs.transpose(2, 0, 1)) if self.padding else single_inputs.transpose(2, 0, 1)
            self.dweights += np.stack([self.calculate_dweights(single_input, single_dvalue) for single_dvalue in single_dvalue], 0)
            single_dinput = np.sum([self.calculate_dinputs(single_input, single_dvalue[i], self.weights[i]) for i in range(self.filters)], axis=0)
            self.dinputs[i] = single_dinput
        self.dbiases += np.sum(single_dvalue, axis=(1, 2))
        
        if self.padding=='valid':
            pad_h = (self.input.shape[1] - self.dinputs.shape[1]) // 2
            pad_w = (self.input.shape[2] - self.dinputs.shape[2]) // 2
            self.dinputs = np.pad(self.dinputs, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    
        self.dweights /= len(self.input)
        self.dbiases /= len(self.input)
    
    def calculate_dweights(self, single_input, dvalues):
        if self.s_h> 1 or self.s_w > 1:
            full_dvalues = np.array([np.zeros((input.shape[-2] - self.k_h + 1, input.shape[-1] - self.k_w + 1)) for input in single_input])
            full_dvalues[:, ::self.s_h, ::self.s_w] = dvalues
            dweights = np.array([convolve2d(input, np.flipud(np.fliplr(full_dvalue)), mode='valid') for input, full_dvalue in zip(single_input, full_dvalues)])
        else:
            dweights = np.array([convolve2d(input, np.flipud(np.fliplr(dvalues)), mode='valid') for input in single_input])
        return dweights
    
    def calculate_dinputs(self, single_input, dvalues, weights):
        if self.s_h > 1 or self.s_w > 1:
            full_dvalues = np.array([np.zeros((input.shape[-2] - self.k_h + 1, input.shape[-1] - self.k_w + 1)) for input in single_input])
            full_dvalues[:, ::self.s_h, ::self.s_w] = dvalues
            padded_dvalues = np.array([np.pad(full_dvalue, ((self.k_h-1, self.k_h-1), (self.k_w-1, self.k_w-1)), mode='constant') for full_dvalue in full_dvalues])
            dinputs = np.array([convolve2d(padded_dvalue, weight, mode='valid') for padded_dvalue, weight in zip(padded_dvalues, weights)])
            dinputs = dinputs.transpose(1,2,0)
        else:
            dinputs = np.array([convolve2d(dvalues , np.flipud(np.fliplr(weight)), mode='same') for weight in weights])
            dinputs = dinputs.transpose(1, 2, 0)
        if self.padding == 'same' and (self.s_h > 1 or self.s_w > 1):
            dinputs = dinputs[self.p_h_top:-self.p_h_bottom, self.p_w_left:-self.p_w_right, :]
        return dinputs