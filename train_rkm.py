import numpy as np
import idx2numpy
import kpca_std
from math_custom import *
import reduced_set

def main(input_data):
    input_data /= 255.0
    # model = kpca_std.KpcaStd((input_data.shape[0], 100))
    # results = model.fit(input_data, eta=1, c_stab=0.5, sig2=0.2)

    reduced_set_kernel = reduced_set.ReducedSetMethod(input_data.shape[0], 50)
    reduced_set_kernel.fit(input_data, {'sig2': 0.2})
    return


# MNIST dataset for debugging
image_file = r'./train-images-idx3-ubyte/train-images.idx3-ubyte'
image_array = idx2numpy.convert_from_file(image_file)

# flatten
image_train = np.array(image_array[:1000, :, :], dtype=np.float64)
image_dim = image_train.shape
image_train = image_train.reshape((image_dim[0], image_dim[1] * image_dim[2]))

# run
ans = main(image_train)
print(ans)
