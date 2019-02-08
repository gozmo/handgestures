from handsignals.dataset.tfrecords import read_tf_record
import tensorflow as tf
class AlexNet:
    def __init__(self):
        self.__setup_training()

    def _model(self, batch):
        conv_1 = tf.layers.conv2d(inputs=batch,
                                  filters=96,
                                  kernel_size=[11,11],
                                  padding="valid",
                                  strides=4)

        pool_1 = tf.layers.max_pooling2d(inputs=conv_1,
                                        pool_size=[5,5],
                                        strides=1,
                                        padding="valid")

        conv_2 = tf.layers.conv2d(inputs=pool_1,
                                  filters=256,
                                  kernel_size=[3,3],
                                  padding="valid",
                                  strides=1)

        pool_2 = tf.layers.max_pooling2d(inputs=conv_2,
                                        pool_size=[3,3],
                                        strides=1,
                                        padding="valid")

        conv_3 = tf.layers.conv2d(inputs=pool_2,
                                  filters=384,
                                  kernel_size=[3,3],
                                  padding="valid",
                                  strides=1)

        conv_4 = tf.layers.conv2d(inputs=conv_3,
                                  filters=384,
                                  kernel_size=[3,3],
                                  padding="valid",
                                  strides=1)

        conv_5 = tf.layers.conv2d(inputs=conv_4,
                                  filters=256,
                                  kernel_size=(3,3),
                                  padding="valid",
                                  strides=1)
        flatten = tf.layers.flatten(conv_5)

        dense_1 = tf.layers.dense(inputs=flatten,
                                  units=4096,
                                  activation=tf.nn.relu)
        dense_2 = tf.layers.dense(inputs=dense_1,
                                  units=4096,
                                  activation=tf.nn.relu)

        self.logits = tf.layers.dense(inputs=dense_2,
                                      units=4,
                                      activation=tf.nn.softmax)
        return self.logits

    def __setup_training(self):
        self.filenames = tf.placeholder(tf.string, [None], name="files")
        iterator = read_tf_record(self.filenames)
        next_example, next_label = iterator.get_next()
        logits = self._model(next_example)
        y = tf.one_hot(next_label, 4)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=y,
                                                    logits=self.logits)
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss)

    def train(self):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            self.__tensorboard_graph()

            feed_dict = {self.filenames : ["test.tfrecords"]}
            session.run(self.train_op, feed_dict=feed_dict)

    def __tensorboard_graph(self):
            summary_writer = tf.summary.FileWriter("tf_logs/", graph=tf.get_default_graph())

