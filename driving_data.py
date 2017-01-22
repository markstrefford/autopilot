import scipy.misc
import random
import csv

xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#read Nvidia data (data.txt)
# with open("/vol/driving_dataset/data.txt") as f:
#     for line in f:
#         xs.append("/vol/driving_dataset/" + line.split()[0])
#         #the paper by Nvidia uses the inverse of the turning radius,
#         #but steering wheel angle is proportional to the inverse of turning radius
#         #so the steering wheel angle in radians is used as the output
#         ys.append(float(line.split()[1]) * scipy.pi / 180)
# cut = -150  (for Nvidia)

# Read Udacity data
with open("/vol/data/train_center.csv") as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            filename = '/vol/data/' + row['filename']  # + '.jpg'
            steering_angle = float(row['steering_angle'])
            xs.append(filename)
            ys.append(steering_angle)
cut = -180  # Estimate equivalent for Udacity

#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[cut:], [66, 200]) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[cut:], [66, 200]) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
