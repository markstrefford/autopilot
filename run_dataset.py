#
# Predict steering angles based on a trained model
# The following commands are based on using a model trained on the SullyChen / nvidia dataset
#
# Nvidia dataset :
# python run_dataset.py -i data.csv --data /vol/driving_dataset --delimiter ' ' --units rad --cut -150
#
# Udacity dataset :
# python run_dataset.py -i train_center.csv --data /vol/data --delimiter ',' --units rad --cut -180
#
# Based on code by SullyChen (https://github.com/SullyChen/Autopilot-TensorFlow)
#
# Repurposed for Timelaps AI by Mark Strefford (January 2017)
#

import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import argparse
import csv

sess = tf.InteractiveSession()
saver = tf.train.Saver()
#saver.restore(sess, "save/model.ckpt")
#saver.restore(sess, "save/udacity_model.ckpt")
saver.restore(sess, "save/model.ckpt")

# Setup viewer
frame_size = (640,480)
frame_target = [256, 455]
deg_to_rad = scipy.pi / 180.0
rad_to_deg = 180.0 / scipy.pi
predicted_angles = []

def predict(input_file, image_directory, delim, steering_units, cut, show_predicted):
    img = cv2.imread('steering_wheel_image.jpg', 0)
    rows, cols = img.shape
    smoothed_angle = 0
    i = 0

    with open(image_directory + '/' + input_file) as f:
            reader = csv.DictReader(f, delimiter=delim)
            for row in reader:
                filename = row['filename']  # + '.jpg'
                steering_angle = float(row['steering_angle'])    # From CSV (deg for nvidia, rad for
                # If steering in csv is radians then convert to degrees for display purposes
                if steering_units == 'rad':
                    steering_angle = steering_angle * rad_to_deg
                #else:
                #    steering_angle = -steering_angle  # Angle is opposite to what you'd expect

                orig_image = scipy.misc.imread(image_directory + "/" + filename, mode="RGB")
                full_image = scipy.misc.imresize(orig_image, frame_target )

                image = scipy.misc.imresize(orig_image[190:400], [66, 200]) / 255.0
                #image = scipy.misc.imresize(orig_image[-150:], [66, 200]) / 255.0
                cv2.imshow('predict image', image)
                # Get CNN prediction (radians) and convert back into degrees
                degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * rad_to_deg  #180.0 / scipy.pi
                call("clear")
                print("Timestamp: " + str(i))
                print("Predicted steering angle: " + str(degrees) + " degrees")
                print("Actual steering angle   : " + str(steering_angle) + " degrees")
                print("-------------------------------------------")
                print("Delta                   : " + str(degrees-steering_angle) + " degrees")
                cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
                #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
                #and the predicted angle
                smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
                M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
                dst = cv2.warpAffine(img,M,(cols,rows))
                predicted_angles.append((i, steering_angle, degrees))
                cv2.imshow("steering wheel", dst)
                cv2.waitKey(1)
                i += 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on test data and write to file')
    parser.add_argument('--input', '-i', action='store', dest='input_file',
                        default='data.csv', help='Input model csv file name')
    parser.add_argument('--data-dir', '--data', action='store', dest='data_dir',
                        default='/vol/test/')
    parser.add_argument('--show_predictions', action='store', dest='show_predicted',
                        default=False, help='Show predicted steering angles (not used)')
    parser.add_argument('--delimiter', action='store', dest='delimiter',
                        default=',', help='Delimeter')
    parser.add_argument('--units', action='store', dest='steering_units',
                        default='rad', help='Units: rad or deg')
    parser.add_argument('--file-ext', action='store', dest='file_ext',
                        default='', help='File extension for images if not in csv file, default is use what is in the csv file')
    parser.add_argument('--cut', action='store', dest='cut',
                        default='-150', help='Where to cut the image (Nvidia uses -150')
    args = parser.parse_args()

    predict(args.input_file, args.data_dir, args.delimiter, args.steering_units, args.cut, args.show_predicted)

    # TODO - Write out predicted vs actual steering for graphing later...

