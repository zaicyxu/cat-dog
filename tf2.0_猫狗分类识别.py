# -*- coding: utf-8 -*-
"""
第05讲 Keras卷积神经网络识别CIFAR-10图像
CIFAR-10数据集介绍
http://www.cs.toronto.edu/~kriz/cifar.html
"""

# Import Library，数据准备，标准化
import numpy as np
import os
import cv2
import tensorflow as tf
   

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 0代表使用第一块GPU显卡
tf.config.set_soft_device_placement = False # TensorFlow 自动选择一个现有且受支持的设备来运行操作，以避免指定的设备不存在
tf.config.experimental.set_memory_growth = True # 仅在需要时申请显存空间
gpus = tf.config.experimental.list_physical_devices('GPU') # 获取全部的GPU显卡
print("gpus:", gpus)
 
if gpus:
    # gpus[0] 代表设置第一块显卡的配置，代表建立了显存大小为1GB的“虚拟GPU”
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #print(len(gpus), len(logical_gpus), 'Logical gpus')
    

# 随机种子
np.random.seed(10)

train_path = r'train'

# 制作标签字典
label_dict = {i: j for i, j in enumerate(os.listdir(train_path))}

# 制作训练集和测试集
train_imgs = []
train_labels = []
test_imgs = []
test_labels = []

train_normalize_imgs = []
test_normalize_imgs = []

train_num = 10000       # 使用训练的图片数量
train_proportion = 0.9  # 训练集和测试集划分的比例
# 获取训练数据和训练标签
for label, path in enumerate(os.listdir(train_path)):
    images = os.listdir(os.path.join(train_path, path))
    images = images[:train_num]
    for index, img in enumerate(images):
        # 读取图片路径
        img = cv2.imread(os.path.join(train_path, path, img))
        img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        # 判断训练数量，按训练比例分成训练集和测试集
        if index < train_num * train_proportion:
            train_labels.append(label)
            train_imgs.append(img)
            train_normalize_imgs.append(img.astype('float64'))
        else:
            test_labels.append(label)
            test_imgs.append(img)
            test_normalize_imgs.append(img.astype('float64'))


# 转换为tensorflow比较好处理的numpy数组
train_imgs = np.array(train_imgs)
train_labels = np.array(train_labels) 
test_imgs = np.array(test_imgs)
test_labels = np.array(test_labels)

train_normalize_imgs = np.array(train_normalize_imgs)
test_normalize_imgs = np.array(test_normalize_imgs)

# 输出形状
print("train data:", 'images:', train_imgs.shape, " labels:", train_labels.shape)
print("test  data:", 'images:', test_imgs.shape, " labels:", test_labels.shape)


# 转为one-hot编码
from keras.utils import np_utils

train_labels_OneHot = np_utils.to_categorical(train_labels)
test_labels_OneHot = np_utils.to_categorical(test_labels)


# 建立建立3次卷积神经网络模型

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

model = Sequential()

# 卷积层1与池化层1
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(100, 100, 3),
                 activation='relu', padding='same'))

model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷积层2与池化层2
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷积层3与池化层3

model.add(Conv2D(filters=128, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(Conv2D(filters=128, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 建立神经网络(平坦层、隐藏层、输出层)
model.add(Flatten())
model.add(Dropout(0.15))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(2, activation='softmax'))

# 输出模型列表，需要调整近1000万的参数
print(model.summary())

# 训练模型
epochs = 20  # 训练次数
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

train_history = model.fit(train_normalize_imgs, train_labels_OneHot,
                          validation_split=0.0005,shuffle=True,
                          epochs=epochs, batch_size=256)

# 评估模型的准确率
import matplotlib.pyplot as plt


def show_train_history(train_history, train_acc, test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(test_normalize_imgs,
                        test_labels_OneHot, verbose=0)


print("scores:", scores)

# 进行预测
prediction = model.predict_classes(test_normalize_imgs)
from collections import Counter
result = Counter(prediction[1000:])
print('cat 识别数量：' , result)
result = Counter(prediction[1000:])
print('dog 识别数量：' , result)

"""
def plot_images_labels_prediction(images, labels, prediction,
                                  idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')

        title = str(i) + ',' + label_dict[labels[i]]
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[i]]
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]);
        ax.set_yticks([])
        idx += 1
    plt.show()


plot_images_labels_prediction(test_normalize_imgs, test_labels,
                              prediction, 0, 10)
"""
# 查看预测概率
Predicted_Probability = model.predict(test_normalize_imgs)
print("Predicted_Probability:", Predicted_Probability)


def show_Predicted_Probability(X_img, Predicted_Probability, i):
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(test_imgs[i], (100, 100, 3)))
    plt.show()
    for j in range(2):
        print(label_dict[j] + ' Probability:%1.9f' % (Predicted_Probability[i][j]))


show_Predicted_Probability(test_imgs, Predicted_Probability, 0)



# 加载要预测的图片
# Load images
img_names = ['test_cat.jpg', 'test_dog.jpg']

imgs = []
for img_name in img_names:
    img = cv2.imread(img_name)
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    img = img.astype('float32')
    imgs.append(img)

imgs = np.array(imgs) / 255

predictions = model.predict_classes(imgs)
print(predictions)


# 保存模型和json文件
open('Keras_Cifar_CNN_Conv3_architecture.json', 'w').write(model.to_json())
model.save('CNN.h5')
model.save_weights('Keras_Cifar_CNN_Conv3_weights.h5', overwrite=True)