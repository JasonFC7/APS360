import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image

def train_model(model, train_loader, val_loader, batch_size = 16, learning_rate = 0.001, num_epochs = 50):
    criterion = nn.CrossEntropyLoss()                     
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_acc= np.zeros(num_epochs)
    train_err= np.zeros(num_epochs)
    train_loss= np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    for root, dirs, files in os.walk('Model Storage'):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    
    print("Training Started")
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        train_acc[epoch], train_err[epoch], train_loss[epoch] = evaluate(model, train_loader, criterion, device)
        val_acc[epoch], val_err[epoch], val_loss[epoch] = evaluate(model, val_loader, criterion, device)
        print(("Epoch {}: ").format(epoch))
        print(("Train acc: {} | " + "Train err: {} | " + "Train loss: {}").format(train_acc[epoch], train_err[epoch], train_loss[epoch]))
        print(("Val acc: {} | " + "Val err: {} | " + "Val loss: {}\n").format(val_acc[epoch], val_err[epoch], val_loss[epoch]))

        model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
        torch.save(model.state_dict(), model_path)

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    
    np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
    np.savetxt("{}_train_err.csv".format(model_path), train_err)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_acc.csv".format(model_path), val_acc)
    np.savetxt("{}_val_err.csv".format(model_path), val_err)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)
    
def get_model_name(name, batch_size, learning_rate, epoch):
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name, batch_size, learning_rate, epoch)
    path = "Model Storage/" + path
    return path

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_err = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total_err += (predicted != labels).sum().item()
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    err = float(total_err) / total_samples
    loss = total_loss / total_samples
    accuracy = float(total_correct) / total_samples
    model.train()
    return accuracy, err, loss

def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

def plot_training_curve(path, condition = "Normal"):
    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    n = len(train_err)
    
    plt.title("Train vs Validation Accuracy ({})".format(condition))
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    
    plt.title("Train vs Validation Error ({})".format(condition))
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    
    plt.title("Train vs Validation Loss ({})".format(condition))
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()
    
def load_images(folder):
    images = []
    labels = []
    label_names = sorted(os.listdir(folder))
    for label_folder in label_names:
        label_path = os.path.join(folder, label_folder)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('L') 
                    img_data = np.asarray(img).flatten()  
                    images.append(img_data)
                    labels.append(label_folder)  
            except Exception as e:
                print(f"Failed to process image {img_path}: {e}")
    return np.array(images), np.array(labels)