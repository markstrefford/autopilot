#!/usr/bin/env python
import argparse
import numpy as np
import rosbag
import cv2
from cv_bridge import CvBridge
import scipy.misc
from skimage import transform as tf
import pandas as pd

rad_to_deg = 180.0 / scipy.pi

# Load images from the ROS databag, resize accordingly and ensure orientation (w x h instead of h x w)
def load_rosbag_data(path="../2016-10-10/udacity-dataset_sensor_camera_center_2016-10-11-13-23-02_0.bag"):
    print 'Loading databag...'
    cvbridge = CvBridge()
    bag = rosbag.Bag(path)
    current_steering = 0
    current_speed = 0
    for topic, msg, t in bag.read_messages(
            topics=['/vehicle/steering_report',
                    '/center_camera/image_color',
                    '/center_camera/image_color/compressed']):

            if (topic == '/vehicle/steering_report'):
                current_steering = msg.steering_wheel_angle
                current_speed = msg.speed
            elif (topic == '/center_camera/image_color'):
                #img = cv2.resize(cvbridge.imgmsg_to_cv2(msg, "bgr8"), (200, 66))
                img = cvbridge.imgmsg_to_cv2(msg, "bgr8")
                yield (img, current_steering, current_speed)
            elif (topic == '/center_camera/image_color/compressed'):
                #img = cv2.resize(cvbridge.compressed_imgmsg_to_cv2(msg, "bgr8"), (200, 66))
                img = cvbridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                yield (img, current_steering, current_speed)
    bag.close()

# Based on code from comma.ai viewer
# ***** get perspective transform for images *****

rsrc = \
 [[43.45456230828867, 118.00743250075844],
  [104.5055617352614, 69.46865203761757],
  [114.86050156739812, 60.83953551083698],
  [129.74572757609468, 50.48459567870026],
  [132.98164627363735, 46.38576532847949],
  [301.0336906326895, 98.16046448916306],
  [238.25686790036065, 62.56535881619311],
  [227.2547443287154, 56.30924933427718],
  [209.13359962247614, 46.817221154818526],
  [203.9561297064078, 43.5813024572758]]
rdst = \
 [[10.822125594094452, 1.42189132706374],
  [21.177065426231174, 1.5297552836484982],
  [25.275895776451954, 1.42189132706374],
  [36.062291434927694, 1.6376192402332563],
  [40.376849698318004, 1.42189132706374],
  [11.900765159942026, -2.1376192402332563],
  [22.25570499207874, -2.1376192402332563],
  [26.785991168638553, -2.029755283648498],
  [37.033067044190524, -2.029755283648498],
  [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))   # *2 required due to viewer size (640x480)

def perspective_tform(x, y):
  p1, p2 = tform3_img((x,y))[0]
  return p2, p1

# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=1):
  row, col = perspective_tform(x, y)
  if row >= 0 and row < img.shape[0] and\
     col >= 0 and col < img.shape[1]:
    img[row-sz:row+sz, col-sz:col+sz] = color

def draw_path(img, path_x, path_y, color):
  for x, y in zip(path_x, path_y):
    draw_pt(img, x, y, color)

def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.0014 # slip factor obtained from real data
  steer_ratio = 14.8  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.85   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset) #* deg_to_rad (Udacity data already in rads)
  curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

def draw_path_on(img, speed_ms, angle_steers, color=(0,0,255)):
    path_x = np.arange(0., 20.1, 0.25)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color)


# TODO - Predict based on the image, the speed, or just get something from an array
# TODO - May have multiple versions of this function in the end...
def predict_steering_angle(i, img, speed):
    return 0    # Default to straight ahead for now


# Setup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on test data and write to file')
    parser.add_argument('--bag', '-b', action='store', dest='bag',
                        help='Path to rosbag')
    parser.add_argument('--save', '-s', action='store', dest='save_path',
                        default = '/vol/Ch2_002', help='Path to save images and csv')
    parser.add_argument('--csv', action='store', dest='csv',
                        default = 'data.csv', help='Filename for csv')
    parser.add_argument('--show', action='store', dest='show',
                        default = True, help='Show images during processing')
    args = parser.parse_args()

frame_size = (320,240)

# Main Loop
i=0
image_data_df = pd.DataFrame(columns=['idx', 'filename', 'steering_angle', 'steering_angle_rad', 'speed'])
#data = test_load_udacity_dataset(path)

for image, steering, speed in load_rosbag_data(path=args.bag):
    image_data={}
    steering_angle_deg = -steering * rad_to_deg   # Save in degrees
    #predicted_steering = predict_steering_angle(i, img, speed)
    #draw_path_on(img, speed, steering)
    #draw_path_on(img, speed, predicted_steering, (0, 255, 0))
    img = image[159:370]
    new_x = 455
    r = float(new_x) / img.shape[1]
    # dim = (100, int(image.shape[0] * r))
    dim = (new_x, int(img.shape[0] * r))
    img = cv2.resize(img, dim)    # Get the relevant part!!

    # Display image
    if args.show == True:
        cv2.imshow('Udacity challenge 2 - viewer', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    im_filename = args.save_path + "/" + str(i) + ".jpg"
    cv2.imwrite(im_filename, img)

    image_data={'idx': i, 'filename': im_filename, 'steering_angle': steering_angle_deg, 'steering_angle_rad': steering, 'speed': speed}
    image_data_df = image_data_df.append(image_data, ignore_index=True)  # TODO - Must be easier to do an inline append???

    print "{}: Steering angle: {} / Speed: {}".format(i,steering_angle_deg, speed)
    i += 1

cv2.destroyAllWindows()
image_data_df.to_csv(args.save_path + '/' + args.csv)    # TODO - Use os.path.join





