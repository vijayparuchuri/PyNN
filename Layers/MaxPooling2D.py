import numpy as np


class Layer_MaxPooling2D:
    def __init__(self, kernel_size=(2, 2), stride=(2, 2)):
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, input):
        if self.stride == (2, 2):
            self.s_h, self.s_w = self.kernel_size
        else:
            self.s_h, self.s_w = self.stride
        self.k_h, self.k_w = self.kernel_size
        self.h, self.w = input.shape
        
        self.output = np.zeros((self.h - self.k_h + 1, self.w - self.k_w + 1))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                self.output[i, j] = input[i: i+ self.k_h, j:j+self.k_w].max()
                self.output = self.output[::self.s_h, ::self.s_w]