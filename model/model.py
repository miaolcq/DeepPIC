# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:22:28 2022

@author: tianmiao
"""
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def load_data():
    x = [] 
    y = []    
    for ar in array_p:
        min_max_scaler = preprocessing.MinMaxScaler()
        arr = min_max_scaler.fit_transform(ar)
        x.append(arr)
    for al in array_pl:
        min_max_scaler = preprocessing.MinMaxScaler()
        arl = min_max_scaler.fit_transform(al)
        y.append(arl)
    x = np.expand_dims(np.array(x), axis=3)
    y = np.expand_dims(np.array(y), axis=3)
    np.random.seed(116)
    #备份原数据
    # x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.20, random_state=420)
    # x_train1, x_valid1, y_train1, y_valid1 = train_test_split(rt_train1, rt_train1, test_size=0.25, random_state=420)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=420)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=420)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def conv2d_block(input_tensor, n_filters=64, kernel_size=3, batchnorm=True, padding='same'):
    # the first layer
    x = Conv2D(n_filters, kernel_size, padding=padding, kernel_initializer='he_normal')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # the second layer
    x = Conv2D(n_filters, kernel_size, padding=padding, kernel_initializer='he_normal')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x 
   
def get_unet(input_data, n_filters=64, dropout=0.1, batchnorm=True, padding='same'):
    # contracting path
    c1 = conv2d_block(input_data, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)
    p1 = MaxPooling2D((2, 2))(c1)
  
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)
    p3 = MaxPooling2D((2, 2))(c3)
        
    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)
    #p4 = Dropout(dropout)(c4)
    p4 = MaxPooling2D((2, 2))(c4)
        
    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm, padding=padding)
    p5 = Dropout(dropout)(c5) 

    # extending path
    u6 = Conv2D(n_filters * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(p5))
    u6 = concatenate([c4, u6], axis=3)
    #u6 = concatenate([Conv2DTranspose(n_filters * 8, (2, 2), strides=(2, 2), padding='same')(p5), c4], axis=3)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)

    u7 = Conv2D(n_filters * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c6))
    u7 = concatenate([c3, u7], axis=3)
    #u7 = concatenate([Conv2DTranspose(n_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6), c3], axis=3)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)

    u8 = Conv2D(n_filters * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c7))
    u8 = concatenate([c2, u8], axis=3)
    #u8 = concatenate([Conv2DTranspose(n_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7), c2], axis=3)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)
        
    u9 = Conv2D(n_filters * 1, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c8))
    u9 = concatenate([c1, u9], axis=3)
    #u9 = concatenate([Conv2DTranspose(n_filters * 1, (2, 2), strides=(2, 2), padding='same')(c8), c1], axis=3)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)
    c9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    model = Model(inputs=[input_data], outputs=[outputs])

    return model


def get_pred_array():
    preds_array = []
    for pred in preds:
        preds_array.append(pred)
        for i in range(len(pred[:,:,0])):
            for j in range(len(pred[:,:,0])):
                if pred[i,j,0] <= 0.1:
                    pred[i,j,0] = 0
                else:
                    pred[i,j,0] = 1
    return preds_array

def pred_array():
    choose_spec2 = []
    preds_array = get_pred_array()
    for iu in range(len(x_test1)):
        #iu=60
        choose_spec11 = x_test1[iu]
        for i in range(256):
            for j in range(256):
                if preds_array[iu][i,j,0] == 0.0:
                    choose_spec11[i,j,0] = 0.0 
        choose_spec1 = choose_spec11[126:131,:] 
        choose_spec1 = np.squeeze(choose_spec1)#减少维度
        for i in range(256):
            if preds_array[iu][128,i,0] == 0.0:
                if preds_array[iu][126,i,0] == 1.0:
                    preds_array[iu][128,i,0] = preds_array[iu][126,i,0]
                if preds_array[iu][127,i,0] == 1.0:
                    preds_array[iu][128,i,0] = preds_array[iu][127,i,0]
                if preds_array[iu][129,i,0] == 1.0:
                    preds_array[iu][128,i,0] = preds_array[iu][129,i,0]
                if preds_array[iu][130,i,0] == 1.0:
                    preds_array[iu][128,i,0] = preds_array[iu][130,i,0]
        for i in range(256):
            if preds_array[iu][128,i,0] == 1 and choose_spec1[2,i] == 0.0:
                if len(choose_spec1[:,i][choose_spec1[:,i].nonzero()]) == 1:
                    choose_spec1[2,i] = choose_spec1[:,i][choose_spec1[:,i].nonzero()]
        for q1 in range(256):
            if preds_array[iu][128,q1] == 1:
                break
        for q3 in range(256):
            if preds_array[iu][128,q3] == 1:
                q2 = q3
        choose_spec1[2,0:q1] = 0
        choose_spec1[2,q2+1:] = 0
        choose_spec2.append(choose_spec1)
    return choose_spec2

def choose_label2():
    choose_label2 = []
    for iu in range(len(y_test1)):
        choose_label = y_test1[iu]
        choose_label = choose_label[126:131,:] 
        choose_label=np.squeeze(choose_label)
        for i in range(256):
            if choose_label[2,i] == 0.0 and choose_label[0,i] != 0.0:
                choose_label[2,i] = choose_label[0,i]
            if choose_label[2,i] == 0.0 and choose_label[1,i] != 0.0:
                choose_label[2,i] = choose_label[1,i]
            if choose_label[2,i] == 0.0 and choose_label[3,i] != 0.0:
                choose_label[2,i] = choose_label[3,i]     
            if choose_label[2,i] == 0.0 and choose_label[4,i] != 0.0:
                choose_label[2,i] = choose_label[4,i] 
        choose_label2.append(choose_label)
    return choose_label2

if __name__ == '__main__':
    path = 'D:/Dpic/data/train_1'
    #path = 'D:/Dpic/data/true/MM48/20L/zs_train'
    files= os.listdir(path)
    rt_p = []
    array_p= []
    ms_p = []
    for file in files:
        position = path+'\\'+ file
        #position = "D:/Dpic/data/PICS/157.5_288.12304_5815.359375_142.5_172.25_773.txt"
        choose_spec = np.loadtxt(position)
    
        B = np.zeros([256, 256],dtype=float)
        N = np.unique(choose_spec[:,0])#注意间隔要改
        if ((256-len(N)) % 2) == 0:
            a = np.array(0).repeat(int((256-len(N))/2))
            b = np.array(0).repeat(int((256-len(N))/2))
        else:
            a = np.array(0).repeat(int((256-len(N))/2))
            b = np.array(0).repeat(int((256-len(N))/2)+1)
        left = np.hstack((a,N,b)) 
        # for g in range(len(rt1)):
        #     if rt1[g] == float(choose_spec[np.argmax(choose_spec[:,2]),0]):#注意修改字符数
        #         break
        # left = np.roll(rt1, 128-g, axis = 0)
        C = np.around(choose_spec[np.argmax(choose_spec[:,2]),1],3)
        k1 = np.around(np.arange(C-0.01,C+0.01,0.005),3)
        k2 = np.around(np.arange(C-1.28,C+1.28,0.01),3)
        k3 = np.delete(np.unique(np.hstack((k1,k2))),np.searchsorted(np.unique(np.hstack((k1,k2))), C))
        rows = dict(zip(list(range(0, len(B))), list(left)))
        cols = dict(zip(list(range(0, len(B)+1)), list(k3)))
        for row in range(256):
            for i in range(len(choose_spec)):
                if rows[row] == choose_spec[i,0]:
                    for col in range(256):
                        if cols[col]<choose_spec[i,1]<=cols[col+1]:
                            B[col][row] = choose_spec[i,2]
        rt_p.append(left)
        array_p.append(B)
        ms_p.append(k3)
        
    path = 'D:/Dpic/data/label_1'
    files= os.listdir(path)
    choose_label = []
    for file in files:
        position = path+'\\'+ file
        # position = "D:/Dpic/data2/ninanjie/label/217.534_251.1002655029297_849915.0_177.84_256.89_213.txt"
        choose_label.append(np.loadtxt(position))
    array_pl = []
    for i in range(len(rt_p)):
        B = np.zeros([256, 256],dtype=float)
        rows = dict(zip(list(range(0, len(B))), list(rt_p[i])))
        cols = dict(zip(list(range(0, len(B)+1)), list(ms_p[i])))   
        for row in range(256):
            for label in choose_label:
                for i in range(len(label)):
                    if rows[row] == label[i,0]:
                        for col in range(256):
                            if cols[col]<label[i,1]<=cols[col+1]:
                                B[col][row] = label[i,2]
                # B=np.pad(B,((28,28),(28,28)),'constant',constant_values = (0,0))
        array_pl.append(B)
        
    array_zsl = []
    for i in range(100):
        f = np.zeros([256, 256],dtype=float)
        array_zsl.append(f)
        
    array_zs = []
    path = 'D:/Dpic/data/zs_train_array'
    files= os.listdir(path)
    for file in files:
        position = path+'\\'+ file
        array_zs.append(np.loadtxt(position))
        
    array_p.extend(array_zs)
    array_pl.extend(array_zsl)
    x_train,x_valid, x_test, y_train, y_valid, y_test = load_data()
    # 训练
    results = model.fit(x_train, y_train, batch_size=2, epochs=40,
                        callbacks=callbacks,validation_data=(x_valid, y_valid))
    model.save_weights('D:/Dpic/data/train/model/u-net.h5', overwrite=True)
    model=model.save('D:/Dpic/data/train/model')
    # 绘制损失曲线
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    metric = results.history['accuracy']
    val_metric = results.history["val_accuracy"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    x = np.linspace(0, len(loss), len(loss))  # 创建横坐标
    plt.subplot(121), plt.plot(x, loss, x, val_loss)
    plt.title("Loss curve"), plt.legend(['loss', 'val_loss'])
    plt.xlabel("Epochs"), plt.ylabel("loss")
    plt.subplot(122), plt.plot(x, metric, x, val_metric)
    plt.title("metric curve"), plt.legend(['accuracy', 'val_accuracy'])
    plt.xlabel("Epochs"), plt.ylabel("accuracy")
    plt.show()  # 会弹出显示框，关闭之后继续运行
    fig.savefig('D:/Dpic/data/train/model/curve.png', bbox_inches='tight', pad_inches=0.1)  # 保存绘制曲线的图片
    plt.close()
    
    model = get_unet(input_data, n_filters=64, dropout=0.5, batchnorm=True, padding='same')
    model.compile(optimizer=Adam(lr = 0.001), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    # 保存
    callbacks = [
        EarlyStopping(patience=60, verbose=1),
        ReduceLROnPlateau(factor=0.8, patience=3, min_lr=0.00005, verbose=1),
        ModelCheckpoint('D:/Dpic/data/train/model/best_unet.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)]
    model.load_weights('D:/Dpic/data/train/best_unet2_zz.h5')#加载模型
    preds = model.predict(x_test,batch_size=1,verbose=1)#测试
    pred_array_int = pred_array()#预测结果
    choose_label2 = choose_label2()#对应标签
    
    for index in range(120):
        # N = rt_b[index]
        #index = 13
        N = rt_p1[58]
        for x in range(len(N[:])):
            if N[x] != 0:
                break
        for z in range(len(N[:])):
            if N[z] != 0:
                v = z
        # right=np.pad(rt_b[index],((28,28)),'constant',constant_values = (0,0))   [rt_b[index].nonzero()]
        index=60
        rt_pi = N[x:v+1]
        rt_ti = N[x:v+1]
        int_pi = pred_array_int[index][2,x:v+1]
        int_ti = choose_label2[index][2,x:v+1]
        ##因为有重复的强度
        c=0
        for i in range(len(N[x:v+1])):
            if int_pi[i] == int_ti[i]:
                c=c+1
        d=0
        e=0
        for i in range(len(N[x:v+1])):
            if int_pi[i] != int_ti[i] and int_ti[i] == 0:
                d=d+1
            if int_pi[i] != int_ti[i] and int_pi[i] == 0:
                e=e+1
        ##
        # c=[x for x in int_pi if x in int_ti]
        # d=len([y for y in int_pi if y not in int_ti])
        # e=len([z for z in int_ti if z not in int_pi])
        recall = c/(c+e)
        precision = c/(c+d)
        F_score = 2*recall*precision/(recall+precision)
        print(recall,precision,F_score)
    
    t_iou = []
    for index in range(4):
        # N = rt_b[index]
        index = 58
        N = rt_p1[index]
        for x in range(len(N[:])):
            if N[x] != 0:
                break
        for z in range(len(N[:])):
            if N[z] != 0:
                v = z
        # right=np.pad(rt_b[index],((28,28)),'constant',constant_values = (0,0))   [rt_b[index].nonzero()]
        index=60
        for p1 in range(len(choose_spec2[index][2,x:v+1])):
            if choose_spec2[index][2,x:v+1][p1] != 0:
                break
        for p3 in range(len(choose_spec2[index][2,x:v+1])):
            if choose_spec2[index][2,x:v+1][p3] != 0:
                p2 = p3
        for t1 in range(len(choose_label2[index][2,x:v+1])):
            if choose_label2[index][2,x:v+1][t1] != 0:
                break
        for t3 in range(len(choose_label2[index][2,x:v+1])):
            if choose_label2[index][2,x:v+1][t3] != 0:
                t2 = t3
        rt_pi = N[x:v+1][p1:p2+1]
        rt_ti = N[x:v+1][t1:t2+1]
        int_pi = choose_spec2[index][2,x:v+1][p1:p2+1]
        int_ti = choose_label2[index][2,x:v+1][t1:t2+1]
        rt_p = dict(zip(list(rt_pi), list(int_pi)))
        rt_t = dict(zip(list(rt_ti), list(int_ti)))
        inter_rt_int = list(rt_p.items() & rt_t.items())
        union_rt_int = list(rt_p.items() | rt_t.items())
        def take_int(elem):
            return elem[0]
        inter_rt_int.sort(key=take_int)
        union_rt_int.sort(key=take_int)
        inter_rti = np.array([i[0] for i in inter_rt_int])
        inter_inti = np.array([i[1] for i in inter_rt_int])
        union_rti = np.array([i[0] for i in union_rt_int])
        union_inti = np.array([i[1] for i in union_rt_int])
        if rt_pi[0] >= rt_ti[0] and rt_pi[-1] <= rt_ti[-1]:
            inter = np.trapz(int_pi,rt_pi)
            union = np.trapz(int_ti,rt_ti)
            iou = inter/union
        if (rt_pi[0] < rt_ti[0] and rt_pi[-1] > rt_ti[-1]) or (rt_pi[0] == rt_ti[0] and rt_pi[-1] > rt_ti[-1]) or rt_pi[0] < rt_ti[0] and rt_pi[-1] == rt_ti[-1]:#注意等于号
            inter = np.trapz(int_ti,rt_ti)
            union = np.trapz(int_pi,rt_pi)
            iou = inter/union
        if inter_rti.shape == (0,) or inter_inti.shape == (0,) or union_rti.shape == (0,) or union_inti.shape == (0,):
            iou = 0
        if rt_pi[0] > rt_ti[0] and rt_pi[-1] > rt_ti[-1] and inter_rti.shape != (0,): 
            inter = np.trapz(inter_inti,inter_rti)
            union = np.trapz(union_inti,union_rti)
            iou = inter/union
        if rt_pi[0] < rt_ti[0] and rt_pi[-1] < rt_ti[-1] and inter_rti.shape != (0,): 
            inter = np.trapz(inter_inti,inter_rti)
            union = np.trapz(union_inti,union_rti)
            iou = inter/union
        t_iou.append(iou)

