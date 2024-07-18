from model import prim_model
from image_loading import image_loader
from training import train_model, plot_training_curve, get_model_name

paths = ["Processed Images", "Validation Set", "Final Test Set"]
train_loader, val_loader, test_loader = image_loader(paths)

input_size = [176, 208]
kernel_sizes = [5, 5, 5, 5]
strides = [1, 1, 1, 1]
paddings = [0, 0, 0, 0]
batch_size = 16
learning_rate = 0.001
num_epoch = 15
num_epoch_plot = num_epoch - 1
condition = "Normal"

prim_model = prim_model(input_size = input_size, kernel_sizes = kernel_sizes, strides = strides, paddings = paddings)
classes = ["0", "0.5", "1"]
train_model(prim_model, train_loader, val_loader, batch_size, learning_rate, num_epoch)
model_path = get_model_name("prim_model",  batch_size, learning_rate, num_epoch_plot)
plot_training_curve(model_path, condition)