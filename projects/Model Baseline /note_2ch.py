def extrack_dataset(dataset):
    for x, y, window_ind in dataset:
        x_shape = x.shape
        y_shape = len(dataset.get_metadata().target)
        break
    X = np.zeros((y_shape,x_shape[0],x_shape[1]))
    y_=[]
    i=0
    for x, y, window_ind in dataset:
        X[i]=x
        y_.append(y)
        i+=1
    X2 = X[:, 7:8, :]
    X3= X[:, 11:12, :]
    X = np.concatenate((X2,X3), axis=1)
    print(X.shape)
    return X,np.array(y_).T



run = neptune.init_run(
    project="AitBrainLab/BaseLine",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMTMyMzg0My02NzlhLTQ3N2ItYTdmMS0yNTcwNDBmM2QwM2QifQ==",
)

n_chans = X_train.shape[1]



params = {"Subject number":subject_id,
              "learning_rate": lr ,
              "optimizer": "AdamW" ,
              "Network":"EEGITNet",
              "Datasets":dataset_name,
              "sfreq":dataset.datasets[0].raw.info['sfreq'],
              "Class number":n_classes,
              "Channel number": train_set[0][0].shape[0],
              'low_cut_hz' : low_cut_hz,
              'high_cut_hz': high_cut_hz

              }