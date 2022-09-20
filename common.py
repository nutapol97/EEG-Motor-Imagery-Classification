import os
#comment this if you are not using puffer?
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
from mne.datasets import eegbci
import glob
from IPython.display import clear_output
import numpy as np
import torch
from torch import nn
import torch.optim as optim

from mne.datasets import eegbci
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader,SubsetRandomSampler
from scipy import signal

import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.autograd import Variable
import math
from torchvision.transforms import Compose, Normalize, ToTensor

from sklearn.model_selection import KFold
import numpy as np

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def train(model,
          gpu_num,
          train_loader,
          test_loader,
          
          optimizer,
          criterion,
          wand,
          vail_loader= None,
          cross = False,
          
         ):
    
    config = wand.config
    weights_name = config.weightname
    # Train the model
    num_epochs = config.epochs

    train_loss = []
    valid_loss = [10,11]
    train_accuracy = []
    valid_accuracy = []
    

    valid_loss_vail = []
    
    
    for epoch in range(config.epochs):
        iter_loss = 0.0
        correct = 0
        iterations = 0

        model.train()

        for i, (items, classes) in enumerate(train_loader):
            items = Variable(items)
            classes = classes.type(torch.LongTensor)
            classes = Variable(classes)

            if cuda.is_available():
                items = items.cuda(gpu_num)
                classes = classes.cuda(gpu_num)

            optimizer.zero_grad()
            outputs = model(items)
            loss = criterion(outputs, classes)

            iter_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            metrics = {"train/train_loss": loss}
            if i + 1 < config.num_step_per_epoch:
                # 🐝 Log train metrics to wandb 
                wand.log(metrics)
            
            #print(loss)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data).sum()
            iterations += 1

        train_loss.append(iter_loss/iterations)
        

        train_accuracy.append((100 * correct.float() / len(train_loader.dataset)))
        train_metrics = {"train/train_loss": iter_loss/iterations, 
                       "train/train_accuracy": (100 * correct.float() / len(train_loader.dataset))}
        
        wand.log({**metrics, **train_metrics})
        
        
        loss = 0.0
        correct = 0
        iterations = 0
        model.eval()

        for i, (items, classes) in enumerate(test_loader):
            classes = classes.type(torch.LongTensor)
            items = Variable(items)
            classes = Variable(classes)
            
            if cuda.is_available():
                items = items.cuda(gpu_num)
                classes = classes.cuda(gpu_num)


            outputs = model(items)
            loss += criterion(outputs, classes).item()

            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == classes.data).sum()
            #print("correct : {}".format(classes.data))
            #print("predicted : {}".format(predicted))
            iterations += 1

        valid_loss.append(loss/iterations)
        correct_scalar = np.array([correct.clone().cpu()])[0]
        valid_accuracy.append(correct_scalar / len(test_loader.dataset) * 100.0)
        
        test_metrics = {"Test/Test_loss": loss/iterations, 
                       "Test/Test_accuracy": correct_scalar / len(test_loader.dataset) * 100.0}
        wand.log({**metrics, **test_metrics})

        if epoch+1 > 2 and valid_loss[-1] < valid_loss[-2] and valid_accuracy[-2] <= valid_accuracy[-1] :
                newpath = r'./weight{}'.format(weights_name) 
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                torch.save(model.state_dict(), './weight{}/{}_{:.4f}_{:.4f}'.format(weights_name,weights_name,valid_loss[-1],valid_accuracy[-1]))
        if (epoch % 100) ==0:
            print ('Epoch %d/%d, Tr Loss: %.4f, Tr Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f'
                       %(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], valid_loss[-1], valid_accuracy[-1]))
            
        if cross :
            loss_vail = 0.0
            correct_vail = 0
            iterations_vail = 0
            model.eval()

            for i, (items, classes) in enumerate(vail_loader):
                classes = classes.type(torch.LongTensor)
                items = Variable(items)
                classes = Variable(classes)

                if cuda.is_available():
                    items = items.cuda(gpu_num)
                    classes = classes.cuda(gpu_num)


                outputs = model(items)
                loss_vail += criterion(outputs, classes).item()

                _, predicted = torch.max(outputs.data, 1)

                correct_vail += (predicted == classes.data).sum()
                #print("correct : {}".format(classes.data))
                #print("predicted : {}".format(predicted))
                iterations_vail += 1

            valid_loss_vail.append(loss_vail/iterations_vail)
            correct_scalar = np.array([correct_vail.clone().cpu()])[0]
            valid_accuracy.append(correct_scalar / len(vail_loader.dataset) * 100.0)
            vali_metrics = {"val/val_loss": loss_vail/iterations, 
                       "val/val_accuracy": correct_scalar / len(test_loader.dataset) * 100.0}
            wand.log({**metrics, **vali_metrics})
            if (epoch % 100) ==0:
                print ('Val Loss: {0}, Val Acc: {1}'.format(valid_loss_vail[-1], valid_accuracy[-1]))

    return train_loss,valid_loss,train_accuracy,valid_accuracy


def setup_dataflow(X_tensor,y_tensor, train_idx, val_idx):
    #train_sampler = SubsetRandomSampler(train_idx)
    #val_sampler = SubsetRandomSampler(val_idx)
    train_datasets = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_datasets = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
    
    train_loader = DataLoader(train_datasets, batch_size=32)
    val_loader = DataLoader(val_datasets, batch_size=32)

    return train_loader, val_loader

def vali(model,
          gpu_num,
          train_loader,
          criterion,
          wand
         ):
    
    config = wand.config
    weights_name = config.weightname
    # Train the model
    num_epochs = config.epochs

    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []
    old_acc = 0

    for epoch in range(config.epochs):
        iter_loss = 0.0
        correct = 0
        iterations = 0
        loss = 0.0
        model.eval()

        for i, (items, classes) in enumerate(train_loader):
            classes = classes.type(torch.LongTensor)
            items = Variable(items)
            classes = Variable(classes)
            
            if cuda.is_available():
                items = items.cuda(gpu_num)
                classes = classes.cuda(gpu_num)


            outputs = model(items)
            loss += criterion(outputs, classes).item()

            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == classes.data).sum()
            #print("correct : {}".format(classes.data))
            #print("predicted : {}".format(predicted))
            iterations += 1

        valid_loss.append(loss/iterations)
        correct_scalar = np.array([correct.clone().cpu()])[0]
        valid_accuracy.append(correct_scalar / len(train_loader.dataset) * 100.0)
        
        print ('Epoch  Val Loss: %.4f, Val Acc: %.4f'
                       %(valid_loss[-1], valid_accuracy[-1]))
        
    return valid_loss,valid_accuracy

class EEG:
    def __init__(self, path, base_url, subjects, runs):
        self.subpath = ''
        self.path = path
        self.base_url = base_url
        self.subjects = subjects
        self.runs = runs
        
        # download data if does not exist in path.
        # self.load_data()
        self.data_to_raw()
    
    def load_data(self):
        print(f">>> Start download from: {self.base_url}.")
        print(f"Downloading files to: {self.path}.")
        for subject in self.subjects:
            eegbci.load_data(subject,self.runs,path=self.path,base_url=self.base_url)
        print("Done.")
    
    
        
        print("Done.")
        return self.raw
    def filter(self, freq):
        raw = self.raw  
        low, high = freq
        print(f">>> Apply filter.")
        self.raw.filter(low, high, fir_design='firwin', verbose=20)
        return  raw
    def raw_ica(self):
        raw = self.raw
        ica = mne.preprocessing.ICA(n_components=64, max_iter=100)
        ica.fit(raw)
        ica.exclude = [1, 2]  # details on how we picked these are omitted here
        ica.plot_properties(raw, picks=ica.exclude)
        ica.apply(raw)
        print('ICA DONE ????')
        return  raw
        
    def get_events(self):
        event_id = dict(T1=0, T2=1) # the events we want to extract
        events, event_id = mne.events_from_annotations(self.raw, event_id=event_id)
        return events, event_id
    
    def get_epochs(self, events, event_id):
        picks = mne.pick_types(self.raw.info, eeg=True, exclude='bads')
        tmin = 0
        tmax = 4
        epochs = mne.Epochs(self.raw, events, event_id, tmin, tmax, proj=True, 
                            picks=picks, baseline=None, preload=True)
        return epochs
    
    def create_epochs(self):
        print(">>> Create Epochs.")
        events, event_id = self.get_events()
        self.epochs = self.get_epochs(events, event_id)
        return events , event_id
        
        print("Done.")
    
    def get_X_y(self):
        if self.epochs is None:
            events , event_id=self.create_epochs()
        self.X = self.epochs.get_data()
        self.y = self.epochs.events[:, -1]
        return self.X, self.y
    
    def get_X_y_ourdata(self,tmin=0,tmax=4):
        events = mne.find_events(raw)
        
        epochs = mne.Epochs(
        raw,
        events,
        event_id=[1,2,3],
        tmin=tmin,
        tmax=tmax,
        picks="data",
        on_missing='warn',
        baseline=None,
            preload=True
    )
        epochs=epochs.resample(160)
            #events , event_id=self.create_epochs()
        self.X = epochs.get_data()
        self.y = epochs.events[:, -1]
        return self.X, self.y 
    
    def data_to_raw(self):
        fullpath = os.path.join(self.path, *self.subpath.split(sep='/'))
        #print(f">>> Extract all subjects from: {fullpath}.")
        extension = "edf"
        raws = []
        count = 1
        for i, subject in enumerate(self.subjects):
            sname = f"S{str(subject).zfill(3)}".upper()
            
            for j, run in enumerate(self.runs):
                rname = f"{sname}R{str(run).zfill(2)}".upper()
                path_file = os.path.join(fullpath, sname, f'{rname}.{extension}')
                #print(path_file)
                #print(f"Loading file #{count}/{len(self.subjects)*len(self.runs)}: {f'{rname}.{extension}'}")
                raw = mne.io.read_raw_edf( path_file , preload=True, verbose='WARNING' )
                raws.append(raw)
                count += 1

        raw = mne.io.concatenate_raws(raws)
        eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage)
        self.raw = raw
        
        
        
def do_plot(train_loss, valid_loss,ty):
    if ty == "loss":
        plt.figure(figsize=(10,10))
        
        plt.plot(train_loss, label='train_loss')
        plt.plot(valid_loss, label='valid_loss')
        plt.title('loss {}'.format(iter))
        plt.legend()
        plt.show()
    if ty == "acc":
        plt.figure(figsize=(10,10))
        
        plot_ty=torch.tensor(train_loss, device = 'cpu')
        plat_va=torch.tensor(valid_loss, device = 'cpu')
        plt.plot(plot_ty, label='train_acc')
        plt.plot(plat_va, label='valid_acc')
        plt.title('acc {}'.format(iter))
        plt.legend()
        plt.show()

        
        
        
        
        
        
class EEG_fif:
    def __init__(self, path, base_url, subjects, runs):
        self.subpath = ''
        self.path = path
        print(path)
        self.base_url = base_url
        self.subjects = subjects
        self.runs = runs
        
        # download data if does not exist in path.
        # self.load_data()
        self.data_to_raw()
    
    def load_data(self):
        print(f">>> Start download from: {self.base_url}.")
        print(f"Downloading files to: {self.path}.")
        for subject in self.subjects:
            eegbci.load_data(subject,self.runs,path=self.path,base_url=self.base_url)
        print("Done.")
    
    
        
        print("Done.")
        return self.raw
    def filter(self, freq):
        raw = self.raw
        low, high = freq
        print(f">>> Apply filter.")
        self.raw.filter(low, high, fir_design='firwin', verbose=20)
        return  raw
    def raw_ica(self):
        raw = self.raw
        ica = mne.preprocessing.ICA(n_components=1, max_iter=100)
        ica.fit(raw)
        ica.exclude = [1, 2]  # details on how we picked these are omitted here
        ica.plot_properties(raw, picks=ica.exclude)
        ica.apply(raw)
        print('ICA DONE ????')
        return  raw
        
    def get_events(self):
        event_id = dict(T1=0, T2=1) # the events we want to extract
        events, event_id = mne.events_from_annotations(self.raw, event_id=event_id)
        return events, event_id
    
    def get_epochs(self, events, event_id):
        picks = mne.pick_types(self.raw.info, eeg=True, exclude='bads')
        tmin = 0
        tmax = 4
        epochs = mne.Epochs(self.raw, events, event_id, tmin, tmax, proj=True, 
                            picks=picks, baseline=None, preload=True)
        return epochs
    
    
        
    
    def data_to_raw(self):
        fullpath = os.path.join(self.path, *self.subpath.split(sep='/'))
        #print(f">>> Extract all subjects from: {fullpath}.")
        extension = "fif"
        raws = []
        count = 1
        for i, subject in enumerate(self.subjects):
            sname = f"S{str(subject).zfill(3)}".upper()
            
            for j, run in enumerate(self.runs):
                rname = f"{sname}R{str(run).zfill(2)}".upper()
                path_file = os.path.join(fullpath, sname, f'{rname}.{extension}')
                #print(path_file)
                #print(f"Loading file #{count}/{len(self.subjects)*len(self.runs)}: {f'{rname}.{extension}'}")
                raw = mne.io.read_raw_fif( path_file , preload=True, verbose='WARNING' )
                raws.append(raw)
                count += 1

        raw = mne.io.concatenate_raws(raws)
        eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage)
        self.raw = raw
    
    def create_epochs(self):
        print(">>> Create Epochs.")
        
        events, event_id = self.get_events()
        self.epochs = self.get_epochs(events, event_id)
        print("Done.")
        return events , event_id
# getepoch(raw,4, 10,reject_bad=False,on_missing='warn')    
    def get_X_y(self,raw,tmin=0,tmax=4):
        events = mne.find_events(raw)
        
        epochs = mne.Epochs(
        raw,
        events,
        event_id=[1,2,3],
        tmin=tmin,
        tmax=tmax,
        picks="data",
        on_missing='warn',
        baseline=None,
            preload=True
    )
        epochs=epochs.resample(160)
            #events , event_id=self.create_epochs()
        self.X = epochs.get_data()
        self.y = epochs.events[:, -1]
        return self.X, self.y 

        
def getepoch(raws,trial_duration, calibration_duration,reject_bad=False,on_missing='warn'):
    #reject_criteria = dict(eeg=100e-6)  # 100 µV
    #flat_criteria = dict(eeg=1e-6)  # 1 µV
    epochs_list = []
    raws = [raws]
    print(len(raws))
    for raw in raws:
        print(raw)
        events = mne.find_events(raw)
        epochs = mne.Epochs(
            raw,
            events,
            event_id=[1,2,3],
            tmin=-calibration_duration,
            tmax=trial_duration,
            picks="data",
            on_missing=on_missing,
            baseline=None
        )
        
        epochs_list.append(epochs)
    epochs = mne.concatenate_epochs(epochs_list)
    labels = epochs.events[:,-1]
    
    print(f'Found {len(labels)} epochs')
    print(epochs)
    print(labels)

    return epochs.get_data(),epochs,labels

        
        
def do_plot(train_loss, valid_loss,ty):
    if ty == "loss":
        plt.figure(figsize=(10,10))
        
        plt.plot(train_loss, label='train_loss')
        plt.plot(valid_loss, label='valid_loss')
        plt.title('loss {}'.format(iter))
        plt.legend()
        plt.show()
    if ty == "acc":
        plt.figure(figsize=(10,10))
        
        plot_ty=torch.tensor(train_loss, device = 'cpu')
        plat_va=torch.tensor(valid_loss, device = 'cpu')
        plt.plot(plot_ty, label='train_acc')
        plt.plot(plat_va, label='valid_acc')
        plt.title('acc {}'.format(iter))
        plt.legend()
        plt.show()
def create_dataloader(X, y, batch_size):
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).long()
    dataset_tensor = TensorDataset(X_tensor, y_tensor)
    dl = torch.utils.data.DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)
    return dl