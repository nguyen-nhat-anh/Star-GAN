import tensorflow as tf
from tensorflow.keras import layers


def make_discriminator_model(label_dim, d_conv_dim):
    input_layer = layers.Input(shape=(128, 128, 3), name='disc_input')  # (None, 128, 128, 3)
    
    x = layers.Conv2D(d_conv_dim, (4, 4), strides=(2, 2), padding='same',
                      use_bias=True, name='disc_conv1')(input_layer)  # (None, 64, 64, d_conv_dim)
    x = layers.LeakyReLU(0.01)(x)
    
    x = layers.Conv2D(d_conv_dim * 2, (4, 4), strides=(2, 2), padding='same',
                      use_bias=True, name='disc_conv2')(x)  # (None, 32, 32, d_conv_dim*2)
    x = layers.LeakyReLU(0.01)(x)
    
    x = layers.Conv2D(d_conv_dim * 4, (4, 4), strides=(2, 2), padding='same',
                      use_bias=True, name='disc_conv3')(x)  # (None, 16, 16, d_conv_dim*4)
    x = layers.LeakyReLU(0.01)(x)
    
    x = layers.Conv2D(d_conv_dim * 8, (4, 4), strides=(2, 2), padding='same',
                      use_bias=True, name='disc_conv4')(x)  # (None, 8, 8, d_conv_dim*8)
    x = layers.LeakyReLU(0.01)(x)
    
    x = layers.Conv2D(d_conv_dim * 16, (4, 4), strides=(2, 2), padding='same',
                      use_bias=True, name='disc_conv5')(x)  # (None, 4, 4, d_conv_dim*16)
    x = layers.LeakyReLU(0.01)(x)
    
    x = layers.Conv2D(d_conv_dim * 32, (4, 4), strides=(2, 2), padding='same',
                      use_bias=True, name='disc_conv6')(x)  # (None, 2, 2, d_conv_dim*32)
    x = layers.LeakyReLU(0.01)(x)
    
    out_src = layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same', 
                            use_bias=False, name='disc_conv_src')(x)  # (None, 2, 2, 1)
    
    out_cls = layers.Conv2D(label_dim, (128//64, 128//64), strides=(1, 1), padding='valid', 
                            use_bias=False, name='disc_conv_cls')(x) # (None, 1, 1, label_dim)
    out_cls = layers.Reshape((label_dim,))(out_cls)  # (None, label_dim)
    
    return tf.keras.Model(input_layer, [out_src, out_cls])