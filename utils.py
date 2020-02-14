import os
import tensorflow as tf
import numpy as np
import ntpath
import matplotlib.pyplot as plt
from matplotlib import lines


def decode_img(img_path):
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, [128, 128])  # resize image
    img = (tf.cast(img, tf.float32) - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return img


class TrainingDataset:
    def __init__(self, label_df, selected_attributes):
        self.label_df = label_df
        self.selected_attributes = selected_attributes

    def get_label(self, img_path):
        file_name = ntpath.basename(img_path).decode('utf-8')
        label = (self.label_df[self.label_df['image_id'] == file_name][self.selected_attributes].values.squeeze() > 0).\
            astype('float32')
        return label

    def preprocess(self, img_path):
        img = decode_img(img_path)
        label = tf.py_function(func=lambda x: self.get_label(x.numpy()), inp=[img_path], Tout=tf.float32)
        label.set_shape([len(self.selected_attributes)])  # tf.py_function returns a tensor with unknown shape
                                                          # so we have to manually set the shape
        return img, label  # (128, 128, 3), (len(selected_attributes),)
    
    
def display_samples(samples, sample_dir, index, nrows=5):
    """
    Display the first `nrows` of `batch_size` samples
    samples - np.array with shape (n_attributes + 1, batch_size, 128, 128, 3)
    nrows must be less than or equal to batch_size
    """
    ncols, batch_size = samples.shape[0], samples.shape[1]
    assert nrows <= batch_size, 'nrows must be less than or equal to batch_size'
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2, nrows*2))
    
    for i, r in enumerate(axes):
        for j, c in enumerate(r):
            c.imshow((samples[j, i, :, :, :] * 127.5 + 127.5).astype(int))
            c.axis('off')
        
    column_titles = ['Original', 'Gender', 'Aged']
    assert len(column_titles) == ncols, "Column titles must match the selected attributes"
    for ax, column_title in zip(axes[0], column_titles):
        ax.set_title(column_title)
    
    X = np.array([0.3375, 0.3375])
    Y = np.array([0.0, 1.0])
    line = lines.Line2D(X, Y, color='r', lw=2, ls='--', transform=fig.transFigure)
    fig.lines.append(line)
        
    fig.tight_layout()
    plt.savefig(os.path.join(sample_dir, 'iter-{}.png'.format(str(index))))


def display_test_result(original_img, generated_img, selected_attrs, attr_values, test_img_path):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))

    ax1.imshow((original_img * 127.5 + 127.5).astype(int))
    ax1.set_title('original', fontsize='xx-large')
    ax1.axis('off')

    ax2.imshow((generated_img * 127.5 + 127.5).astype(int))
    ax2.set_title(','.join(['{}{}'.format(attr, value) for attr, value in zip(selected_attrs, attr_values)]),
                  fontsize='xx-large')
    ax2.axis('off')

    fig.tight_layout()

    test_img_dir, test_img_fname = os.path.split(test_img_path)
    name, ext = os.path.splitext(test_img_fname)
    generated_img_fname = name + '_result' + ext
    save_path = os.path.join(test_img_dir, generated_img_fname)
    plt.savefig(save_path)
    print('Generated image is saved at {}'.format(save_path))