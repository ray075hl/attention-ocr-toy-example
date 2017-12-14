import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.layers.core import Dense
from data_generator_ctc import *
from tensorflow.contrib.keras import backend as K

IMAGE_HEIGHT = 32
RNN_UNITS = 256
N_CLASS = 10+1
TRAIN_STEP = 100000
BATCH_SIZE  = 128
print(N_CLASS)

image = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, None, 1), name='img_data')
label = tf.sparse_placeholder(tf.int32, name='label')
feature_length = tf.placeholder(tf.int32, shape=[None], name='feature_length')


def encoder_net(_image, scope, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        convolution1 = layers.conv2d(inputs=_image,
                                     num_outputs=64,
                                     kernel_size=[3, 3],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)
        pool1 = layers.max_pool2d(inputs=convolution1, kernel_size=[2, 2], stride=[2, 2])

        convolution2 = layers.conv2d(inputs=pool1,
                                     num_outputs=128,
                                     kernel_size=[3, 3],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)
        pool2 = layers.max_pool2d(inputs=convolution2, kernel_size=[2, 2], stride=[2, 2])

        convolution3 = layers.conv2d(inputs=pool2,
                                     num_outputs=256,
                                     kernel_size=[3, 3],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)

        convolution4 = layers.conv2d(inputs=convolution3,
                                     num_outputs=256,
                                     kernel_size=[3, 3],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)
        pool3 = layers.max_pool2d(inputs=convolution4, kernel_size=[2, 1], stride=[2, 1])

        convolution5 = layers.conv2d(inputs=pool3,
                                     num_outputs=512,
                                     kernel_size=[3, 3],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)
        n1 = layers.batch_norm(convolution5)

        convolution6 = layers.conv2d(inputs=n1,
                                     num_outputs=512,
                                     kernel_size=[3, 3],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)
        n2 = layers.batch_norm(convolution6)
        pool4 = layers.max_pool2d(inputs=n2, kernel_size=[2, 1], stride=[2, 1])

        convolution7 = layers.conv2d(inputs=pool4,
                                     num_outputs=512,
                                     kernel_size=[2, 2],
                                     padding='VALID',
                                     activation_fn=tf.nn.relu)
        cnn_out = tf.squeeze(convolution7, axis=1)
        sequence_length = tf.reshape(cnn_out, [-1])
        print('cnn_out:', sequence_length.get_shape())
        cell = tf.contrib.rnn.GRUCell(num_units=RNN_UNITS)
        enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                                 cell_bw=cell,
                                                                 inputs=cnn_out,
                                                                 dtype=tf.float32)
        encoder_outputs = tf.concat(enc_outputs, -1)
        return encoder_outputs


def ctc_loss(_lstm_features, _label, _feature_length):
    project_output = layers.fully_connected(inputs=_lstm_features,
                                  num_outputs=N_CLASS,
                                  activation_fn=None)
    #[max_time x batch_size x num_classes].
    loss = tf.nn.ctc_loss(labels=_label, inputs=project_output, sequence_length=_feature_length, time_major=False)
    cost = tf.reduce_mean(loss)
    train_one_step = tf.train.AdadeltaOptimizer().minimize(cost)
    return cost, train_one_step


if __name__ == '__main__':
    features_lstm = encoder_net(image, 'encode')
    print(features_lstm.get_shape())
    cost, train_step = ctc_loss(features_lstm, label, feature_length)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # tensorboard visualization
    with tf.name_scope('summaries'):
        tf.summary.scalar("cost", cost)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs_path')
    # ------------------------------------------

    with sess.as_default():
        data_gen = name_training_data_generator(BATCH_SIZE)
        for i in range(TRAIN_STEP):
            input_data = data_gen.__next__()
            #print(input_data['input'].shape)
            sess.run(train_step, feed_dict={image: input_data['input'],
                                            label: input_data['label'],
                                            feature_length: input_data['feature_length']})
            print(sess.run(cost, feed_dict={image: input_data['input'],
                                            label: input_data['label'],
                                            feature_length: input_data['feature_length']}))
