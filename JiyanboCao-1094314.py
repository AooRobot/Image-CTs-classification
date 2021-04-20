# Date 8/5/2020
import os
import cv2
import numpy as np
from tempfile import TemporaryDirectory
from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import pickle
# from collections import OrderedDict
import matplotlib.pyplot as plt
from keras import utils
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D

path = './Dataset/DS'
# labels = ['Abdomen','Chest','Head']


# Return 32x32 images and labels
def imgLabel(path):
    os.chdir(path)
    train_labels = []
    train_imgs = []
    for img in os.listdir('./'):
        train_labels.append(img.split('_')[0])
        image = cv2.imread(img)
        resized = cv2.resize(image,dsize=(32,32),interpolation=cv2.INTER_CUBIC)
        train_imgs.append(resized)
    return train_imgs, train_labels

def Cnn(input_shape,number_classes):
    model = Sequential() 
    model.add(Conv2D(64, 3, 3, input_shape = input_shape, padding='same',activation='relu')) 
    model.add(MaxPooling2D(pool_size=(1, 1)))
    # Layer 2
    model.add(Conv2D(64, 5, 5, padding='same',activation='relu')) 
    model.add(MaxPooling2D(pool_size=(1, 1),strides=2))
    # Layer 3
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(128, 3, 3, padding='same',activation='relu'))
    # Layer 4 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(192, 3, 3, padding='same',activation='relu')) 
    # Layer 5 
    model.add(ZeroPadding2D((1,1))) 
    model.add(Conv2D(128, 3, 3, padding='same',activation='relu')) 
    model.add(MaxPooling2D(pool_size=(1, 1)))
    # # Layer 6 
    model.add(GlobalAveragePooling2D()) 
    model.add(Dense(256, kernel_initializer='glorot_normal',activation='relu'))
    model.add(Dropout(0.5))
    # Layer 7 
    model.add(Dense(256, kernel_initializer='glorot_normal',activation='relu')) 
    model.add(Dropout(0.5)) 
    # Layer 8 
    model.add(Dense(number_classes, kernel_initializer='glorot_normal',activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model


if __name__ == "__main__":
    imgs, labels = imgLabel(path)
    imgs = np.array(imgs)
    # Get unique labels
    unique_labels = LabelEncoder().fit(labels)
    print('Unique Labels: ', unique_labels.classes_)
    n_category = len(unique_labels.classes_)
    # Transform labels to integer
    labels = LabelEncoder().fit_transform(labels)

    # ----display----
    print('Shape of X(images): ', imgs.shape)
    # print('Labels encoded: \n',ohe_labels)

    # Spilt dataset
    X_train, X_test, Y_train, Y_test = train_test_split(imgs, labels,test_size=0.2,random_state=42)
    cnn = Cnn((32,32,3),n_category)
    cnn.summary()
    print('Shape of Training data: ',X_train.shape)
    print('Shape of labels: ', Y_train.shape)
    

    # Training StratifiedKFold with 5 eoochs
    fold_cc = 0
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skf.split(X_train, Y_train):
        fold_cc += 1
        print('Number of Folder: {}'.format(fold_cc))
        
        x_train, x_test = X_train[train_index], X_train[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]

        # One-hot-encoded labels
        y_train = utils.to_categorical(y_train, n_category)
        y_test = utils.to_categorical(y_test, n_category)

        history = cnn.fit(x_train, y_train, validation_data = (x_test, y_test),
                                            epochs = 5,
                                            batch_size=128)
    
    # # 2 epochs without cross validation
    # history = cnn.fit(X_train, Y_train, epochs = 2, batch_size=128)

    # # Save model
    save_path = '../../'
    os.chdir(save_path)
    # cnn.save('JiyanboCao-1094314-cnn.h5')

    # Load model
    # cnn_test = load_model('JiyanboCao-1094314-cnn.h5')

    # Feature extraction by CNN
    layer = 'dense'
    feature_layer = Model(inputs=cnn.input, outputs=cnn.get_layer(layer).output)
    train_features = feature_layer.predict(X_train)
    train_features = np.array(train_features)
    print('Shape of train features: ', train_features.shape)
    print('Train features:\n',train_features)

    # K Nearest Neighbors, find the optimal value of K
    # Caching nearest neighbors
    # ------------------KNN----------------
    k_list = [1,2,3,4,5,6,7,8,9]
    knn_graph = KNeighborsTransformer(n_neighbors=max(k_list), mode='distance')
    knn = KNeighborsClassifier()
    with TemporaryDirectory(prefix="sklearn_graph_cache_") as tmpdir:
        full_model = Pipeline(
            steps=[('graph', knn_graph), ('classifier', knn)],
            memory=tmpdir)
    
    param_grid = {'classifier__n_neighbors': k_list}
    knn_grid = GridSearchCV(full_model, param_grid)
    knn_grid.fit(train_features, Y_train)

    # graph of knn
    fig,axes = plt.subplots()
    axes.errorbar(x=k_list,
                 y=knn_grid.cv_results_['mean_test_score'],
                 yerr=knn_grid.cv_results_['std_test_score'])
    axes.set(xlabel='n_neighbors', title='Classification accuracy')
    fig.tight_layout()
    plt.show()

    print('The best performance is the value ',knn_grid.best_params_)
    knn_grid = knn_grid.best_estimator_

    # Random Forest -find the optimal hyperparameters
    # Tune n_estimators(number of trees) and max_features
    # (size of random subset of variables)
    # ---------------RF------------------
    rf = RandomForestClassifier(random_state= 42)
    rf_grid = {
        'n_estimators':[20,50,100,125],
        'max_features':['sqrt','log2'],
    }
    rf_grid = GridSearchCV(estimator=rf, param_grid=rf_grid)
    rf_grid.fit(train_features, Y_train)

    print('The best parameters are:',rf_grid.best_params_)
    rf_grid = rf_grid.best_estimator_

    # rf = RandomForestClassifier(n_estimators=30, random_state= 42, max_features='sqrt',oob_score=True)
    # rf.fit(train_features, Y_train)

    # # Save models
    # # Save KNN model 
    with open('JiyanboCao-1094314-KNN.pkl','wb') as pickle_knn:
        pickle.dump(knn_grid, pickle_knn)
    # # Save RF model
    with open('JiyanboCao-1094314-RF.pkl','wb') as pickle_rf:
        pickle.dump(rf_grid, pickle_rf)


    # Test
    # Y_test = utils.to_categorical(Y_test)
    test_features = feature_layer.predict(X_test)
    test_features = np.array(test_features)
    print('Shape of test features: ',test_features.shape)
    print('Test features: \n',test_features)

    # KNN results
    with open('JiyanboCao-1094314-KNN.pkl','rb') as pickle_knn:
        load_knn = pickle.load(pickle_knn)
    print(load_knn)
    predictions_knn = load_knn.predict(test_features)
    # predictions_knn = knn_grid.predict(test_features)

    print('predicted knn: \n',predictions_knn)
    print('test: \n', Y_test)
    accuracy_knn = accuracy_score(predictions_knn, Y_test)
    print('Confusion matrix:\n', confusion_matrix(Y_test, predictions_knn, labels = range(n_category)))
    print(classification_report(Y_test, predictions_knn,target_names=unique_labels.classes_))
    print('Accuracy: ', accuracy_knn, '\n', accuracy_knn*100,'%.')

    # RF results
    with open('JiyanboCao-1094314-RF.pkl','rb') as pickle_rf:
        load_rf = pickle.load(pickle_rf)
    print(load_rf)
    predictions_rf = load_rf.predict(test_features)
    # predictions_rf = rf_grid.predict(test_features)

    print('predicted rf: \n',predictions_rf)
    print('test: \n', Y_test)
    accuracy_rf = accuracy_score(predictions_rf, Y_test)
    print(classification_report(Y_test, predictions_rf,target_names=unique_labels.classes_))
    print('Confusion matrix:\n', confusion_matrix(Y_test, predictions_rf, labels = range(n_category)))
    print('Accuracy: ', accuracy_rf, '\n', accuracy_rf*100,'%.')
    
    pass