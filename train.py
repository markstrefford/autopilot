import os
import tensorflow as tf
import driving_data
import model
import argparse

L2NormConst = 0.001

def train(load_model):
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
  logs_path = './logs'
  summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

  epochs = 30
  batch_size = 100

  # train over the dataset about 30 times
  for epoch in range(epochs):
    for i in range(int(driving_data.num_images/batch_size)):
      xs, ys = driving_data.LoadTrainBatch(batch_size)
      train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
      if i % 10 == 0:
        xs, ys = driving_data.LoadValBatch(batch_size)
        loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
        print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

      # write logs at every iteration
      summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      summary_writer.add_summary(summary, epoch * batch_size + i)

      if i % batch_size == 0:
        if not os.path.exists(LOGDIR):
          os.makedirs(LOGDIR)
        checkpoint_path = os.path.join(LOGDIR, "udacity_model.ckpt")   # But remember to save out to udacity model!!
        filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on test data and write to file')
    parser.add_argument('--input', '-i', action='store', dest='input_file',
                        default='data.csv', help='Input model csv file name')
    parser.add_argument('--load-model', action='store', dest='load_model',
                        default=False, help='Load a pre-trained model (assume that the model architecture is the same!)')
    parser.add_argument('--logdir', action='store', dest='logdir',
                        default='./save', help='Log directory')


    parser.add_argument('--data-dir', '--data', action='store', dest='data_dir',
                        default='/vol/test/')
    parser.add_argument('--delimiter', action='store', dest='delimiter',
                        default=',', help='Delimeter')
    parser.add_argument('--units', action='store', dest='steering_units',
                        default='rad', help='Units: rad or deg')
    parser.add_argument('--file-ext', action='store', dest='file_ext',
                        default='', help='File extension for images if not in csv file, default is use what is in the csv file')
    parser.add_argument('--cut', action='store', dest='cut',
                        default='-150', help='Where to cut the image (Nvidia uses -150')
    args = parser.parse_args()

    train(args.load_model)

    print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
