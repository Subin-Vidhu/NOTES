# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 07:19:34 2021

@author: ASUS
"""

import tensorflow as tf
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
