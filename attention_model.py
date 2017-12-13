import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.layers.core import Dense
from data_generator_att import *

START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
VOCAB = {'<S>': 0, '</S>': 1, '<UNK>': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12}
VOCAB_SIZE = len(VOCAB)
BATCH_SIZE = 32
RNN_UNITS = 256
TRAIN_STEP = 1000000
IMAGE_HEIGHT = 32
MAXIMUM__DECODE_ITERATIONS = 20
DISPLAY_STEPS = 100
LOGS_PATH = 'logs_path'
CKPT_DIR = 'save_model'

image = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, None, 1), name='img_data')
train_output = tf.placeholder(tf.int64, shape=[None, None], name='train_output')
train_length = tf.placeholder(tf.int32, shape=[None], name='train_length')
target_output = tf.placeholder(tf.int64, shape=[None, None], name='target_output')


def encoder_net(_image, scope, reuse=None):
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

        cell = tf.contrib.rnn.GRUCell(num_units=RNN_UNITS)
        enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                                 cell_bw=cell,
                                                                 inputs=cnn_out,
                                                                 dtype=tf.float32)
        encoder_outputs = tf.concat(enc_outputs, -1)
        return encoder_outputs


def decode(helper, memory, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=RNN_UNITS, memory=memory)
        cell = tf.contrib.rnn.GRUCell(num_units=RNN_UNITS)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=RNN_UNITS, output_attention=True)
        output_layer = Dense(units=VOCAB_SIZE)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=attn_cell, helper=helper,
            initial_state=attn_cell.zero_state(dtype=tf.float32, batch_size=BATCH_SIZE),
            output_layer=output_layer)
        outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, output_time_major=False,
            impute_finished=True, maximum_iterations=MAXIMUM__DECODE_ITERATIONS)
        return outputs


def build_compute_graph():
    train_output_embed = encoder_net(image, scope='encode_features')
    pred_output_embed = encoder_net(image, scope='encode_features', reuse=True)

    output_embed = layers.embed_sequence(train_output, vocab_size=VOCAB_SIZE, embed_dim=VOCAB_SIZE, scope='embed')
    embeddings = tf.Variable(tf.truncated_normal(shape=[VOCAB_SIZE, VOCAB_SIZE], stddev=0.1), name='decoder_embedding')

    start_tokens = tf.zeros([BATCH_SIZE], dtype=tf.int64)

    train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, train_length)
    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)
    train_outputs = decode(train_helper, train_output_embed, 'decode')
    #pred_outputs = decode(pred_helper, pred_output_embed, 'decode', reuse=True)
    pred_outputs = decode(pred_helper, train_output_embed, 'decode', reuse=True)

    train_decode_result = train_outputs[0].rnn_output[0, :-1, :]
    pred_decode_result = pred_outputs[0].rnn_output[0, :, :]

    print(tf.one_hot(target_output, depth=VOCAB_SIZE).get_shape())
    loss = tf.div(
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=train_outputs[0].rnn_output[:, :-1, :], # logits
                                                        labels=tf.one_hot(target_output, depth=VOCAB_SIZE))),# targets
        tf.cast(train_length[0], tf.float32)
        )

    train_one_step = tf.train.AdadeltaOptimizer().minimize(loss)
    return loss, train_one_step, train_decode_result, pred_decode_result


def train_network(loss, train_one_step, train_decode_result, pred_decode_result):

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # # First let's load meta graph and restore weights
    # saver = tf.train.import_meta_graph('./save_model/attention_digit_ocr.model-100.meta')
    #
    # saver.restore(sess, tf.train.latest_checkpoint('./save_model'))

    # tensorboard visualization
    with tf.name_scope('summaries'):
        tf.summary.scalar("cost", loss)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGS_PATH)

    # train
    with sess.as_default():

        data_gen = name_training_data_generator(BATCH_SIZE)
        for i in range(TRAIN_STEP):
            input_data = data_gen.__next__()
            train_one_step.run(feed_dict={image: input_data['input'],
                                          train_output: input_data['train_output'],
                                          target_output: input_data['target_output'],
                                          train_length: input_data['train_length']})

            if i % DISPLAY_STEPS == 0:
                summary_loss, loss_result = sess.run([summary_op, loss],
                                                     feed_dict={image: input_data['input'],
                                                                train_output: input_data['train_output'],
                                                                target_output: input_data['target_output'],
                                                                train_length: input_data['train_length']})
                writer.add_summary(summary_loss, i)
                train_outputs_result = sess.run([train_decode_result],
                                                feed_dict={image: input_data['input'],
                                                           train_output: input_data['train_output'],
                                                           target_output: input_data['target_output'],
                                                           train_length: input_data['train_length']})
                pred_outputs_result = sess.run([pred_decode_result],
                                               feed_dict={image: input_data['input'],
                                                          train_output: input_data['train_output'],
                                                          target_output: input_data['target_output']})

                print("Step:{}, loss:{}, train_decode:{}, predict_decode:{}, ground_truth:{}".
                      format(i,
                             loss_result,
                             np.argmax(train_outputs_result[0], axis=1),
                             np.argmax(pred_outputs_result[0], axis=1),
                             input_data['target_output'][0]))
                # save model
                saver = tf.train.Saver()
                model_name = "attention_digit_ocr.model"
                if not os.path.exists(CKPT_DIR):
                    os.makedirs(CKPT_DIR)
                saver.save(sess, os.path.join(CKPT_DIR, model_name), global_step=i)

def main():
    loss, train_one_step, train_decode_result, pred_decode_result = build_compute_graph()
    train_network(loss, train_one_step, train_decode_result, pred_decode_result)


if __name__ == '__main__':
    main()
