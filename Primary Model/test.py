from training import plot_training_curve, get_model_name

input_size = [176, 208]
kernel_sizes = [5, 5, 5]
strides = [1, 1, 1]
paddings = [0, 0, 0]
batch_size = 16
learning_rate = 0.001
num_epoch = 50
num_epoch_plot = num_epoch - 1
condition = "Normal"

model_path = get_model_name("prim_model",  batch_size, learning_rate, num_epoch_plot)
plot_training_curve(model_path, condition)