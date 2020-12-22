from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Reshape, LeakyReLU
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import numpy as np

image_size = 64

im = Image.open('F:/DCGAN/anime-faces/img_align_celeba/1.png')
im = im.resize((image_size,image_size),Image.ANTIALIAS)
plt.imshow(im)
plt.show()

X_train = Image.open('F:/DCGAN/anime-faces/img_align_celeba/1.png')
X_train = X_train.resize((image_size,image_size),Image.ANTIALIAS)
X_train = np.asanyarray(X_train)
X_train = np.expand_dims(X_train, axis=0) # 将(64,64,3)维拓展为(1,64,64,3)
print(X_train.shape)

for dirname, _, filenames in os.walk('F:/DCGAN/anime-faces/img_align_celeba/'):
    for filename in filenames:
        if X_train.shape[0] > 5000:
            break
        try:
            im = Image.open(os.path.join(dirname, filename))
            im = im.resize((image_size,image_size),Image.ANTIALIAS)
            image_array = np.asanyarray(im)
            image_array = np.expand_dims(image_array, axis=0)
            X_train = np.concatenate((X_train, image_array), axis=0)
        except:
            pass
print(str(X_train.shape[0]))

character = 300 # 特征数

# # 生成器
# def generator_model():
#     model = Sequential()
#     model.add(Dense(int(image_size/8)*int(image_size/8)*256, input_shape=(character,)))
    
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Reshape((int(image_size/8),int(image_size/8),256))) # output: 8*8*256
    
#     model.add(Conv2DTranspose(128,5,strides=2,padding='SAME')) # output: (None,16,16,128)
    
#     model.add(Activation('relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2DTranspose(64,5,strides=2,padding='SAME')) # output: (None, 32, 32, 64)
    
#     model.add(Activation('relu'))
#     model.add(BatchNormalization())
    
#     model.add(Conv2DTranspose(3,5, strides=2,padding='SAME')) # output: (None, 64, 64, 3)

#     model.add(Activation('tanh'))
        
#     return model
    

# 第二种生成器架构
def generator_model():
    model = Sequential()
    model.add(Dense(int(image_size/4)*int(image_size/4)*128, input_shape=(character,)))
    # output: 16*16*64
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((int(image_size/4),int(image_size/4),128)))
    # output: (None,16,16,128)
    model.add(Conv2DTranspose(64,5,strides=2,padding='SAME'))
    # output: (None, 32, 32, 64)
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(3,5, strides=2,padding='SAME'))
    # (None, 64, 64, 1)
    model.add(Activation('tanh'))
    
    return model

g = generator_model()
g.summary()

# def discriminator_model():
#     model = Sequential()
#     model.add(Conv2D(64, padding='SAME',kernel_size=5,strides=2, input_shape=(image_size, image_size, 3)))
#     model.add(LeakyReLU())
#     model.add(BatchNormalization())
#     model.add(Conv2D(128,padding='SAME',kernel_size=5,strides=2))
#     model.add(LeakyReLU())
#     model.add(BatchNormalization())
#     model.add(Conv2D(256,padding='SAME',kernel_size=5,strides=2))
#     model.add(LeakyReLU())
#     model.add(BatchNormalization())
#     model.add(Flatten())
#     model.add(Dense(1024))
#     model.add(BatchNormalization())
#     model.add(Activation('tanh'))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
        
#     return model

# 第二种判别器架构
def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, padding='SAME',kernel_size=5,strides=2, activation='tanh', input_shape=(image_size, image_size, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(128,padding='SAME',kernel_size=5,strides=2, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
        
    return model

d = discriminator_model()
d.summary()

def combine(g,d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    
    return model

g = generator_model()
d = discriminator_model()
g_d = combine(g,d)
g_d.summary()

def combine_images(images):
    num = images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1], 3),
                    dtype = images.dtype)
    for index,img in enumerate(images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i+1) * shape[0], j * shape[1]:(j+1) * shape[1], 0:3] = img[:,:,:]
        
    return image

result_path = 'F:/DCGAN/anime-faces/generated/result/'

if os.path.exists(result_path)==False:
    os.makedirs(result_path)
    
def generated(noise_need, name):
    g = generator_model()
    try:
        g.load_weights(model_path+"generatorA")
        print("生成器权重导入成功")
    except:
        print("无权重")
    noise_need = np.random.normal(-1,1,size=(1,character))
    generated_image_need = g.predict(noise_need, verbose=0)
    image = combine_images(generated_image_need)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        result_path+name+".png")

model_path = 'F:/DCGAN/model/'
generated_image_path = 'F:/DCGAN/anime-faces/generated/'

if os.path.exists(model_path)==False:
    os.makedirs(model_path)
if os.path.exists(generated_image_path)==False:
    os.makedirs(generated_image_path)

def train(BATCH_SIZE, X_train):
    # 生成图片的连接图片数
    generated_image_size = 36
    # 读取图片
    X_train = ((X_train.astype(np.float32)) - 127.5) / 127.5 
    
    # 模型及其优化器
    d = discriminator_model()
    g = generator_model()
    g_d = combine(g,d)
    d_optimizer = RMSprop()
    g_optimizer = RMSprop()
    g.compile(loss='binary_crossentropy', optimizers='SGD') # 生成器
    g_d.compile(loss='binary_crossentropy',optimizers=g_optimizer) # 联合模型
    d.trainable = True
    d.compile(loss='binary_crossentropy',optimizers=d_optimizer) # 判别器
    
    # 导入权重
    try:
        d.load_weights(model_path+"discriminatorA")
        print("判别器权重导入成功")
        g.load_weights(model_path+"generatorA")
        print("生成器权重导入成功")
    except:
        print("无权重")
    
    for epoch in range(1000):
        # 每1轮打印一次当前轮数
        if epoch % 1 == 0:
            print('Epoch is ',epoch)
        for index in range(X_train.shape[0]//BATCH_SIZE):
             # 产生（-1，1）的正态分布的维度为（BATCH_SIZE, character）的矩阵
            noise = np.random.normal(-1,1,size=(BATCH_SIZE,character))
            train_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_image = g.predict(noise, verbose=0)
            
            if index % 50 == 0:
                # 每50次输出一次图片
                noise_need = np.random.normal(-1,1,size=(generated_image_size,character))
                generated_image_need = g.predict(noise_need, verbose=0)
                image = combine_images(generated_image_need)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    generated_image_path+str(epoch)+"_"+str(index)+".png")
            # 每运行一次训练一次判别器
            if index % 1 == 0:
                X = np.concatenate((train_batch,generated_image))
                Y = list((np.random.rand(BATCH_SIZE)*10+90)/100) + [0]*BATCH_SIZE
                d_loss = d.train_on_batch(X,Y)
            
            noise = np.random.normal(-1,1,size=(BATCH_SIZE,character))
            d.trainable = False
            g_loss = g_d.train_on_batch(noise, list((np.random.rand(BATCH_SIZE)*10+90)/100))
            d.trainable = True
            if index % 10 == 0:
                print('batch: %d, g_loss: %f, d_loss: %f' % (index, g_loss, d_loss))
            
            if index % 10 == 0:
                g.save_weights(model_path+'generatorA', True)
                print('Successfully save generatorA')
                d.save_weights(model_path+'discriminatorA', True)
                print('Successfully save discriminatorA')

train(BATCH_SIZE=128, X_train=X_train)
