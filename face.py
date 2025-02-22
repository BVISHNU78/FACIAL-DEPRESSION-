import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from PIL import Image
import keras,random
import seaborn as sns
from PIL import Image
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
train=r"D:\coding\facial depression\depression\acess\Tra"
test=r"D:\coding\facial depression\depression\acess\tes"
IMG_WIDTH=48
IMG_height=48
Batch_size=16
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rotation_range=20,
    width_shift_range=10,
    height_shift_range=10,
    validation_split=0.2
    )
test_datagen = ImageDataGenerator(rescale=1./255)
train_genrator=train_datagen.flow_from_directory(
    train,
    target_size=(IMG_WIDTH,IMG_height),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=Batch_size,
    shuffle=True,
    subset='training'
)
validation_genrator = train_datagen.flow_from_directory(
    train, target_size=(IMG_WIDTH,IMG_height ), color_mode='grayscale', 
      class_mode='binary', batch_size=Batch_size, shuffle=True, subset='validation'
)
test_genrator=train_datagen.flow_from_directory(
    test,
    target_size=(IMG_WIDTH,IMG_height),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=Batch_size,
    shuffle=False,
    
)
print("classes",train_genrator.class_indices)
train_labels=list(train_genrator.class_indices.keys())
test_labels=list(test_genrator.class_indices.keys())
labels={1:"Neutral",0:"Depression"}
img,label=train_genrator.__next__()
i=random.randint(0, (img.shape[0])-1)
image = img[i]
labl = labels[label[i].argmax()]
plt.imshow(image[:,:,0], cmap='gray')
plt.title(labl)
plt.show()
def plot_images(generator,labels):
    x, y = next(generator)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()
    for i in range(len(axes)):
        axes[i].imshow(x[i])  
        label_index = np.argmax(y[i])
        axes[i].set_title(f"Label: {labels[label_index]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
class_count=np.unique(train_genrator.classes,return_counts=True)
print("class",class_count[0])
print("Count",class_count[1])
plt.figure()
sns.barplot(x=train_labels,y=class_count[1])
plt.title("Training Data Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
sns.barplot(x=test_labels,y=class_count[1])
plt.title("validation Data Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
def class_weights(train_genrator):
    classes=np.unique(train_genrator.classes)
    class_weights=compute_class_weight(class_weight="balanced",classes=classes,y=train_genrator.classes)
    class_weight_dict=dict(enumerate(class_weights))
    print(class_weight_dict)
    return class_weight_dict
class_weight=class_weights(train_genrator)
def model_sequential():
    model=Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(48,48,1))),
    model.add(BatchNormalization()),
    model.add(Conv2D(64,(3,3),activation='relu',padding='same')),
    model.add(MaxPooling2D(2,2)),

    model.add(Conv2D(128,(3,3),activation='relu')),
    model.add(BatchNormalization()),
    model.add(MaxPooling2D(2,2)),

   
    model.add(Flatten()),
    model.add(Dense(512,activation='relu',kernel_regularizer=l2(0.001))),
    model.add(Dropout(0.5))

    model.add(Dense(256,activation='relu', kernel_regularizer=l2(0.01))),
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1,activation='sigmoid')),
    model.compile(optimizer=Adam(learning_rate=0.0005,clipnorm=1.0),loss='binary_crossentropy',metrics=["accuracy"])
    return model
model=model_sequential()
model.summary()
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True,verbose=1)
model.fit(train_genrator,epochs=50,validation_data=validation_genrator,batch_size=Batch_size,callbacks=[lr_scheduler,early_stopping],class_weight=class_weight)
scores = model.evaluate(test_genrator)
print("Accuracy: %.2f%%" % (scores[1]*100))
pred_y=model.predict(test_genrator)
pred_label=(pred_y>0.5).astype(int)
true_label=test_genrator.classes
print("Pred shape:", pred_y.shape)
print("Sample preds:", pred_y[:5])
if pred_label.ndim > 1:
    pred_label = pred_label.flatten()
pred_counts = np.bincount(pred_label)
print("Pred counts:", pred_counts)
print("Pred counts:", pred_label)
plt.figure()
cm=confusion_matrix(pred_label,true_label)
sns.heatmap(cm, annot=True,fmt='d',xticklabels=["Neutral", "Depression"], yticklabels=["Neutral", "Depression"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
print(classification_report(true_label,pred_label))
labels=['Neutral','Depression']
n=random.randint(0,img.shape[0]-1)
image=img[n]
org_label=labels[true_label[n]]
preds_label=labels[pred_label[n]]
plt.imshow(image[:,:,0],cmap='gray')
plt.title("original label is:" +org_label+ "predicted"+preds_label)
plt.show()
model.save("face.h5")
