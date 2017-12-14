import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.layers.core import Dense
from common import *
from data_generator_ctc_joint_attention import *


class CtcPlusAttModel(object):
    """
    Class CtcPlusAttModel
    """
    def __init__(self):
        """
        Initialize global variables and compute graph
        """
        # vocabulary parameters
        self.start_token = START_TOKEN
        self.end_token = END_TOKEN
        self.unk_token = UNK_TOKEN
        self.vocab_att = VOCAB_ATT
        self.vocab_att_size = VOCAB_ATT_SIZE
        self.vocab_ctc = VOCAB_CTC
        self.vocab_ctc_size = VOCAB_CTC_SIZE

        # training parameters
        self.batch_size = BATCH_SIZE
        self.rnn_units = RNN_UNITS
        self.max_train_steps = TRAIN_STEP
        self.image_height = IMAGE_HEIGHT
        self.att_embed_dim = ATT_EMBED_DIM
        self.max_dec_iteration = MAXIMUM__DECODE_ITERATIONS
        # loss weights refrencehttps://arxiv.org/pdf/1609.06773v1.pdf
        self.ctc_loss_weights = 0.8
        self.att_loss_weights = 1 - self.ctc_loss_weights
        # choose attention mode 0 is "Bahdanau" Attention, 1 is "Luong" Attention
        self.attention_mode = 1

        # visualization path and model saved path
        self.logs_path = LOGS_PATH
        self.save_model_dir = CKPT_DIR

        # input image
        self.input_image = tf.placeholder(tf.float32, shape=(None, self.image_height, None, 1), name='img_data')

        # attention part placeholder
        self.att_train_output = tf.placeholder(tf.int64, shape=[None, None], name='att_train_output')
        self.att_train_length = tf.placeholder(tf.int32, shape=[None], name='att_train_length')
        self.att_target_output = tf.placeholder(tf.int64, shape=[None, None], name='att_target_output')

        # ctc part placeholder
        self.ctc_label = tf.sparse_placeholder(tf.int32, name='ctc_label')
        self.ctc_feature_length = tf.placeholder(tf.int32, shape=[None], name='ctc_feature_length')

        #
        self.sess = tf.Session()

    def __shared_encoder(self):
        """
        Image features encoded by CNN and bidirectional GRU
        :return: encoded features
        """
        convolution1 = layers.conv2d(inputs=self.input_image,
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

        cell = tf.contrib.rnn.GRUCell(num_units=self.rnn_units)
        enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                                 cell_bw=cell,
                                                                 inputs=cnn_out,
                                                                 dtype=tf.float32)
        encoder_outputs = tf.concat(enc_outputs, -1)
        return encoder_outputs

    def __ctc_loss_branch(self, rnn_features):
        """
        Ctc loss compute graph
        :param rnn_features: encoded features and self.ctc_feature_length、self.ctc_label
        :return: loss matrix
        """
        project_output = layers.fully_connected(inputs=rnn_features,
                                                num_outputs=self.vocab_ctc_size + 1,
                                                activation_fn=None)
        # if time_major=True(default) the inputs must be the shape of [max_time x batch_size x num_classes].
        ctc_loss = tf.nn.ctc_loss(labels=self.ctc_label,
                                  inputs=project_output,
                                  sequence_length=self.ctc_feature_length,
                                  time_major=False)
        return ctc_loss

    def __attention_loss_branch(self, rnn_features):
        output_embed = layers.embed_sequence(self.att_train_output,
                                             vocab_size=self.vocab_att_size,
                                             embed_dim=self.att_embed_dim, scope='embed')
        #  with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.truncated_normal(shape=[self.vocab_att_size, self.att_embed_dim],
                                                     stddev=0.1), name='decoder_embedding')
        start_tokens = tf.zeros([self.batch_size], dtype=tf.int64)

        train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, self.att_train_length)
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                               start_tokens=tf.to_int32(start_tokens),
                                                               end_token=1)

        train_outputs = self.__att_decode(train_helper, rnn_features, 'decode')
        pred_outputs = self.__att_decode(pred_helper, rnn_features, 'decode', reuse=True)

        train_decode_result = train_outputs[0].rnn_output[0, :-1, :]
        pred_decode_result = pred_outputs[0].rnn_output[0, :, :]

        att_loss = tf.nn.softmax_cross_entropy_with_logits(logits=train_outputs[0].rnn_output[:, :-1, :],   # logits
                                                           labels=tf.one_hot(self.att_target_output,
                                                           depth=self.vocab_att_size))

        return att_loss

    def __att_decode(self, helper, rnn_features, scope, reuse=None):
        """
        Attention decode part
        :param helper: train or inference
        :param rnn_features: encoded features
        :param scope: name scope
        :param reuse: reuse or not
        :return: attention decode output
        """
        with tf.variable_scope(scope, reuse=reuse):
            if self.attention_mode == 1:
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_units,
                                                                        memory=rnn_features)
            else:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_units,
                                                                           memory=rnn_features)

            cell = tf.contrib.rnn.GRUCell(num_units=self.rnn_units)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                            attention_layer_size=self.rnn_units,
                                                            output_attention=True)
            output_layer = Dense(units=self.vocab_att_size)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell, helper=helper,
                initial_state=attn_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size),
                output_layer=output_layer)

            att_outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=self.max_dec_iteration)

            return att_outputs

    def build_model(self):
        """
        build compute graph
        :return: model
        """
        # share part
        encode_features = self.__shared_encoder()

        # attention part
        attention_loss = self.__attention_loss_branch(encode_features)

        # ctc part
        ctc_loss = self.__ctc_loss_branch(encode_features)

        # merge part
        t_loss = tf.reduce_mean(attention_loss)*self.att_loss_weights + tf.reduce_mean(ctc_loss)*self.ctc_loss_weights
        train_step = tf.train.AdadeltaOptimizer().minimize(t_loss)
        return train_step, t_loss

    def load_data(self):
        data_gen = gen_training_data(self.batch_size)
        return data_gen

    def train_process(self):
        train_step, loss = self.build_model()
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            data_gen = self.load_data()
            for step in range(self.max_train_steps):
                input_data = data_gen.__next__()
                self.sess.run(train_step, feed_dict={self.input_image: input_data['input_image'],
                                                     self.ctc_label: input_data['ctc_label'],
                                                     self.ctc_feature_length: input_data['ctc_feature_length'],
                                                     self.att_train_output: input_data['att_train_output'],
                                                     self.att_train_length: input_data['att_train_length'],
                                                     self.att_target_output: input_data['att_target_output']})
                if step % DISPLAY_STEPS == 0:
                    loss_print = self.sess.run(loss,
                                               feed_dict={self.input_image: input_data['input_image'],
                                                          self.ctc_label: input_data['ctc_label'],
                                                          self.ctc_feature_length: input_data['ctc_feature_length'],
                                                          self.att_train_output: input_data['att_train_output'],
                                                          self.att_train_length: input_data['att_train_length'],
                                                          self.att_target_output: input_data['att_target_output']})
                    print("step: {}\t loss:\t{}".format(step, loss_print))

    def visualize_log(self):
        pass


if __name__ == '__main__':
    ctc_att_model = CtcPlusAttModel()
    ctc_att_model.train_process()

    # loss = ctc_att_model.build_model()
    # print(loss.get_shape())
