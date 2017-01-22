import csv
import numpy as np
import scipy.misc
import pandas as pd

def explore_udacity_data():
    xs = []
    ys = []

    # Read Udacity data
    with open("/vol/data/train_center.csv") as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            filename = '/vol/data/' + row['filename']  # + '.jpg'
            steering_angle = float(row['steering_angle'])
            xs.append(filename)
            ys.append(int(steering_angle * 180 / scipy.pi))
    return xs, ys

def explore_autopilot_data():
    xs = []
    ys = []
    with open("/vol/driving_dataset/data.txt") as f:
        for line in f:
            xs.append("/vol/driving_dataset/" + line.split()[0])
            #the paper by Nvidia uses the inverse of the turning radius,
            #but steering wheel angle is proportional to the inverse of turning radius
            #so the steering wheel angle in radians is used as the output
            ys.append(int(line.split()[1]))
    return xs, ys


print "Reading udacity dataset..."
#udacity_x, udacity_y = explore_udacity_data()
# udacity_angle_distribution = np.zeros(720)
# print "Calculating steering angle distribution..."
# for i in range(len(udacity_y)):
#     angle = int(udacity_y[i])
#     udacity_angle_distribution[angle] += 1
# print "Read {} lines... Done!".format(len(udacity_y))
udacity_data = pd.read_csv("/vol/driving_dataset/data.txt")


print "Reading autopilot dataset..."
autopilot_x, autopilot_y = explore_autopilot_data()
# autopilot_angle_distribution = np.zeros(720)
# print "Calculating steering angle distribution..."
# for i in range(len(autopilot_y)):
#     angle = int(autopilot_y[i])
#     autopilot_angle_distribution[angle] += 1
# print "Read {} lines... Done!".format(len(autopilot_y))


np.savetxt('save/udacity_distribution.csv', udacity_angle_distribution)
np.savetxt('save/autopilot_distribution.csv', autopilot_angle_distribution)
