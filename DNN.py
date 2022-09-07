from __future__ import print_function

import os,sys
import numpy as np 
import scipy.io as scio
import tensorflow as tf
import keras
from keras.layers import Input, GRU, Dense, Flatten, Dropout, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, TimeDistributed
from keras.models import Model, load_model
import keras.backend as K
from sklearn.metrics import confusion_matrix
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split


use_existing_model = False
fraction_for_test = 0.1
data_dir = 'BVP/'
ALL_MOTION = [1,2,3,4,5,6]
N_MOTION = len(ALL_MOTION)
T_MAX = 0
n_epochs = 30
f_dropout_ratio = 0.5
n_gru_hidden_units = 128
n_batch_size = 4 #32
f_learning_rate = 0.001

def normalize_data(data_1):

    data_1_max = np.concatenate((data_1.max(axis=0),data_1.max(axis=1)),axis=0).max(axis=0)
    data_1_min = np.concatenate((data_1.min(axis=0),data_1.min(axis=1)),axis=0).min(axis=0)
    if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
        return data_1
    data_1_max_rep = np.tile(data_1_max,(data_1.shape[0],data_1.shape[1],1))
    data_1_min_rep = np.tile(data_1_min,(data_1.shape[0],data_1.shape[1],1))
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)
    return  data_1_norm

def zero_padding(data, T_MAX):

    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(data[i], ((0,0),(0,0),(T_MAX - t,0)), 'constant', constant_values = 0).tolist())
    return np.array(data_pad)

def onehot_encoding(label, num_class):
  
    label = np.array(label).astype('int32')
    label = np.squeeze(label)
    _label = np.eye(num_class)[label-1]     
    return _label

def load_data(path_to_data, motion_sel):
    global T_MAX
    data = []
    label = []
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        for data_file_name in data_files:

            file_path = os.path.join(data_root,data_file_name)
            try:
                data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
                label_1 = int(data_file_name.split('-')[1])
                location = int(data_file_name.split('-')[2])
                orientation = int(data_file_name.split('-')[3])
                repetition = int(data_file_name.split('-')[4])

               
                if (label_1 not in motion_sel):
                    continue

                data_normed_1 = normalize_data(data_1)
                
                
                if T_MAX < np.array(data_1).shape[2]:
                    T_MAX = np.array(data_1).shape[2]                
            except Exception:
                continue

            
            data.append(data_normed_1.tolist())
            label.append(label_1)
            
   
    data = zero_padding(data, T_MAX)

    
    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)   
    data = np.expand_dims(data, axis=-1)    

   
    label = np.array(label)

    
    return data, label
    
def assemble_model(input_shape, n_class):
    model_input = Input(shape=input_shape, dtype='float32', name='name_model_input')   

    
    x = TimeDistributed(Conv2D(16,kernel_size=(5,5),activation='relu',data_format='channels_last',
        input_shape=input_shape))(model_input)   
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)    
    x = TimeDistributed(Flatten())(x)   
    x = TimeDistributed(Dense(64,activation='relu'))(x) 
    x = TimeDistributed(Dropout(f_dropout_ratio))(x)
    x = TimeDistributed(Dense(64,activation='relu'))(x) 
    x = GRU(n_gru_hidden_units,return_sequences=False)(x)  
    x = Dropout(f_dropout_ratio)(x)
    model_output = Dense(n_class, activation='softmax', name='name_model_output')(x)  

    
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer=keras.optimizers.RMSprop(lr=f_learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
    return model

if len(sys.argv) < 2:
    print('Please specify GPU ...')
    exit(0)
if (sys.argv[1] == '1' or sys.argv[1] == '0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
  # config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    tf.random.set_seed(1)
else:
    print('Wrong GPU number, 0 or 1 supported!')
    exit(0)


data, label = load_data(data_dir, ALL_MOTION)
print('\nLoaded dataset of ' + str(label.shape[0]) + ' samples, each sized ' + str(data[0,:,:].shape) + '\n')


[data_train, data_test, label_train, label_test] = train_test_split(data, label, test_size=fraction_for_test)
print('\nTrain on ' + str(label_train.shape[0]) + ' samples\n' +\
    'Test on ' + str(label_test.shape[0]) + ' samples\n')


label_train = onehot_encoding(label_train, N_MOTION)


if use_existing_model:
    model = load_model('model_widar3_trained.h5')
    model.summary()
else:
    model = assemble_model(input_shape=(T_MAX, 20, 20, 1), n_class=N_MOTION)
    model.summary()
    model.fit({'name_model_input': data_train},{'name_model_output': label_train},
            batch_size=n_batch_size,
            epochs=n_epochs,
            verbose=1,
            validation_split=0.1, shuffle=True)
    print('Saving trained model...')
    model.save('model_widar3_trained.h5')


print('Testing...')
label_test_pred = model.predict(data_test)
label_test_pred = np.argmax(label_test_pred, axis = -1) + 1


cm = confusion_matrix(label_test, label_test_pred)
print(cm)
cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
cm = np.around(cm, decimals=2)
print(cm)


test_accuracy = np.sum(label_test == label_test_pred) / (label_test.shape[0])
print(test_accuracy)
