import tensorflow as tf
from tensorflow.keras import layers


class ResidualBlock(layers.Layer):
    def __init__(self, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """
        input_shape = (batch_size, height, width, channels)
        """
        channels = input_shape.as_list()[-1]
        self.conv1 = layers.Conv2D(channels, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv1')
        self.conv2 = layers.Conv2D(channels, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv2')
        self.instance_norm1 = layers.LayerNormalization(axis=(1, 2), epsilon=1e-5, name='i_norm1')
        self.instance_norm2 = layers.LayerNormalization(axis=(1, 2), epsilon=1e-5, name='i_norm2')
        super(ResidualBlock, self).build(input_shape)  # Be sure to call this at the end
                
    def call(self, inputs, **kwargs):
        """
        inputs - shape = (batch_size, height, width, channels)
        """
        x = self.conv1(inputs)  # (batch_size, height, width, channels)
        x = self.instance_norm1(x)
        x = layers.ReLU()(x)
        x = self.conv2(x)  # (batch_size, height, width, channels)
        x = self.instance_norm2(x)
        
        x0 = inputs 
        return x + x0  # (batch_size, height, width, channels)
    
    
def make_generator_model(label_dim, g_conv_dim):
    # Concatenate inputs
    input_img = layers.Input(shape=(128, 128, 3), name='gen_input_img')  # (None, 128, 128, 3)
    
    input_lbl = layers.Input(shape=(label_dim,), name='gen_input_lbl')  # (None, label_dim)
    lbl_reshape = layers.Reshape((1, 1, label_dim))(input_lbl)  # (None, 1, 1, label_dim)
    lbl_stack = layers.Lambda(lambda x: tf.tile(x, (1, 128, 128, 1)))(lbl_reshape)  # (None, 128, 128, label_dim)
    
    input_concat = layers.Concatenate(name='gen_input_concat')([input_img, lbl_stack])  # (None, 128, 128, 3+label_dim)
    
    # Downsampling part
    
    x = layers.Conv2D(g_conv_dim, (7, 7), strides=(1, 1), padding='same',
                      use_bias=False, name='gen_conv1')(input_concat)  # (None, 128, 128, g_conv_dim)
    x = layers.LayerNormalization(axis=(1, 2), epsilon=1e-5, name='gen_i_norm1')(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(g_conv_dim * 2, (4, 4), strides=(2, 2), padding='same',
                      use_bias=False, name='gen_conv2')(x)  # (None, 64, 64, g_conv_dim*2)
    x = layers.LayerNormalization(axis=(1, 2), epsilon=1e-5, name='gen_i_norm2')(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(g_conv_dim * 4, (4, 4), strides=(2, 2), padding='same',
                      use_bias=False, name='gen_conv3')(x)  # (None, 32, 32, g_conv_dim*4)
    x = layers.LayerNormalization(axis=(1, 2), epsilon=1e-5, name='gen_i_norm3')(x)
    x = layers.ReLU()(x)
    
    # Bottleneck part
    x = ResidualBlock(name='gen_res_block1')(x) 
    x = ResidualBlock(name='gen_res_block2')(x)
    x = ResidualBlock(name='gen_res_block3')(x)
    x = ResidualBlock(name='gen_res_block4')(x)
    x = ResidualBlock(name='gen_res_block5')(x)
    x = ResidualBlock(name='gen_res_block6')(x)  # (None, 32, 32, g_conv_dim*4)
    
    # Upsampling part
    x = layers.Conv2DTranspose(g_conv_dim * 2, (4, 4), strides=(2, 2), padding='same',
                               use_bias=False, name='gen_deconv1')(x)  # (None, 64, 64, g_conv_dim*2)
    x = layers.LayerNormalization(axis=(1, 2), epsilon=1e-5, name='gen_i_norm4')(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(g_conv_dim, (4, 4), strides=(2, 2), padding='same',
                               use_bias=False, name='gen_deconv2')(x)  # (None, 128, 128, g_conv_dim)
    x = layers.LayerNormalization(axis=(1, 2), epsilon=1e-5, name='gen_i_norm5')(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(3, (7, 7), strides=(1, 1), padding='same', 
                      use_bias=False, name='gen_last_conv')(x)  # (None, 128, 128, 3)
    output_layer = layers.Activation('tanh')(x)

    return tf.keras.Model([input_img, input_lbl], output_layer)