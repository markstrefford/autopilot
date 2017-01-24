import os
import tensorflow as tf
from driving_data import DataReader
import model
import argparse

L2NormConst = 0.001


def train(load_model, data_dir, input_file, logdir, logs_path, save_model, udacity):
    sess = tf.InteractiveSession()
    train_vars = tf.trainable_variables()

    loss = tf.reduce_mean(tf.square(tf.sub(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    sess.run(tf.initialize_all_variables())

    # create a summary to monitor cost tensor
    tf.scalar_summary("loss", loss)
    # merge all summaries into a single op
    merged_summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver()
    if load_model != False:
        saver.restore(sess, load_model)  # Start with the pre-trainied autopilot model

    # op to write logs to Tensorboard
    #logs_path = './logs'
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

    epochs = 30
    batch_size = 100

    # def __init__(self, data_dir=DATA_DIR, data_csv=DATA_CSV, file_ext=FILE_EXT, sequential=False, udacity=True):
    data_reader = DataReader(data_dir=data_dir, data_csv=input_file, file_ext='png', sequential=False, udacity=udacity )

    # train over the dataset about 30 times
    for epoch in range(epochs):
        for i in range(int(data_reader.num_train_images/batch_size)):
            xs, ys = data_reader.LoadTrainBatch(batch_size)
            train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
            if i % 10 == 0:
                xs, ys = data_reader.LoadValBatch(batch_size)
                loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
                print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

            # write logs at every iteration
            summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
            summary_writer.add_summary(summary, epoch * batch_size + i)

            if i % batch_size == 0:
                if not os.path.exists(logdir):
                    os.makedirs(logdir)
                checkpoint_path = os.path.join(logdir, save_model)
                filename = saver.save(sess, checkpoint_path)
        print("Model saved in file: %s" % filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on test data and write to file')
    parser.add_argument('--csv', action='store', dest='input_file',
                        default='data.csv', help='Input model csv file name')
    parser.add_argument('--data', '-d', action='store', dest='data_dir',
                        default='/vol', help='Directory holding images')
    parser.add_argument('--load-model', action='store', dest='load_model',
                        default=False, help='Load a pre-trained model (assume that the model architecture is the same!)')
    parser.add_argument('--logdir', action='store', dest='logdir',
                        default='./save', help='Log directory')
    parser.add_argument('--logpath', action='store', dest='logpath',
                        default='./logs', help='Log path directory')
    parser.add_argument('--save-model', action='store', dest='save_model',
                        default='model.ckpt', help='Name to save model as')
    parser.add_argument('--udacity', action='store', dest='udacity',
                        default="False", help='Custom features to work with Udacity 640x480 images')
    args = parser.parse_args()

    train(args.load_model, args.data_dir, args.input_file, args.logdir, args.logpath, args.save_model, args.udacity)

    print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
