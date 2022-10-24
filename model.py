import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU
from segmentation_models import Unet

def create_model(input_shape=(256, 256, 1)):
    # Network Architecture

    # Input shape is (256, 256, 1)
    input = Input(input_shape, name="Input")
    # Eight convolutional layers (along with batch normalization and ReLU activation functions) with zero padding,
    # Each convolutional layer (except for the last convolutional layer) consists of 64 filters with stride of one

    # First layer hasn't BN
    x = Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", name = "Layer1-Conv2D")(input)
    x = ReLU(name = "Layer1-ReLU")(x)
    # Layers 2-7
    for i in range(2,8):
        x = Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", name = f"Layer{i}-Conv2D")(x)
        x = BatchNormalization(name = f"Layer{i}-BN")(x)
        x = ReLU(name = f"Layer{i}-ReLU")(x)
    # Last layer, 1 filter, no BN
    x = Conv2D(filters = 1, kernel_size = (3, 3), padding = "same", name = "Layer8-Conv2D")(x)
    x = ReLU(name = "Layer8-ReLU")(x)
    # The division residual layer with skip connection divides the input image by the estimated speckle.
    x = tf.math.divide(input, x)
    # A hyperbolic tangent layer is stacked at the end of the network, which serves as a nonlinear function
    x = tf.math.tanh(x)
    return tf.keras.Model(inputs=input, outputs=x)
