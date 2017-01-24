import scipy.misc
import random
import csv
import cv2

# Defaults...
DATA_DIR = '/vol/data/'
FILE_EXT = '.png'
DATA_CSV = '/vol/data/train_center.csv'

deg_to_rad = scipy.pi / 180.0
rad_to_deg = 180.0 / scipy.pi

class DataReader(object):
    def __init__(self, data_dir=DATA_DIR, data_csv=DATA_CSV, file_ext=FILE_EXT, sequential=False, udacity=True):
        self.data_dir = data_dir
        self.data_csv = data_csv
        self.file_ext = file_ext
        self.sequential = sequential
        self.udacity = udacity
        self.load()

    def load(self):
        xs = []
        ys = []

        #points to the end of the last batch
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        #Read Udacity data (60x480)
        if self.udacity == True:
            with open(self.data_csv) as f:
                    reader = csv.DictReader(f, delimiter=',')
                    for row in reader:
                        filename = self.data_dir + row['filename']  # + '.jpg'
                        steering_angle = float(row['steering_angle'])
                        xs.append(filename)
                        ys.append(steering_angle)
        else:
            # Read data in autopilot format... (455x150, 455x210, etc.)
            with open(self.data_csv) as f:
                for line in f:
                        xs.append(self.data_dir + line.split()[0])
                        #the paper by Nvidia uses the inverse of the turning radius,
                        #but steering wheel angle is proportional to the inverse of turning radius
                        #so the steering wheel angle in radians is used as the output
                        ys.append(float(line.split()[1]) * deg_to_rad)

        #get number of images
        num_images = len(xs)

        #shuffle list of images
        c = list(zip(xs, ys))
        random.shuffle(c)
        xs, ys = zip(*c)

        self.train_xs = xs[:int(len(xs) * 0.8)]
        self.train_ys = ys[:int(len(xs) * 0.8)]

        self.val_xs = xs[-int(len(xs) * 0.2):]
        self.val_ys = ys[-int(len(xs) * 0.2):]

        self.num_train_images = len(self.train_xs)
        self.num_val_images = len(self.val_xs)

    def load_image(self, udacity, filename):
        if udacity:
            return scipy.misc.imresize(scipy.misc.imread(filename)[159:370], [66, 200]) / 255.0
        else:
            return scipy.misc.imresize(scipy.misc.imread(filename)[-150:], [66, 200]) / 255.0


    def LoadTrainBatch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            train_image = self.load_image(self.udacity, self.train_xs[(self.train_batch_pointer + i) % self.num_train_images])
            #cv2.imshow("Train", train_image)
            #cv2.waitKey(1)
            x_out.append(train_image)
            y_out.append([self.train_ys[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return x_out, y_out

    def LoadValBatch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            val_image = self.load_image(self.udacity, self.val_xs[(self.val_batch_pointer + i) % self.num_val_images])
            #cv2.imshow("Val", val_image)
            #cv2.waitKey(1)
            x_out.append(val_image)
            y_out.append([self.val_ys[(self.val_batch_pointer + i) % self.num_val_images]])
        self.val_batch_pointer += batch_size
        return x_out, y_out
