# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:41:35 2022

@author: tianmiao
"""
import numpy as np

def pics(array,pred_array_int,pred_array_mz,choose_spec):
    pic_rt = []
    pic_mz = []
    pic_intensity = []
    pic_index = []
    pics = []
    peaks = []
    for index in range(len(array)):
        # N = rt_b[index]
        #index = 0
        N = array[index][1][0]
        for x in range(len(N[:])):
            if N[x] != 0:
                break
        for z in range(len(N[:])):
            if N[z] != 0:
                v = z   
        mz_1 = pred_array_mz[index][2,x:v+1]
        int_1 = pred_array_int[index][2,x:v+1]
        if int_1.any() == 0:
            continue
        else:
            for p1 in range(len(int_1)):
                if int_1[p1] != 0:
                    break
            for p3 in range(len(int_1)):
                if int_1[p3] != 0:
                    p2 = p3
            rt_2 = N[x:v+1][p1:p2+1]
            int_2 = int_1[p1:p2+1]
            mz_2 = mz_1[p1:p2+1]
            max_int_index = np.searchsorted(rt_2, choose_spec[index][1])
            if len(int_2)<=max_int_index or int_2[max_int_index]==0:
                continue
            else:
                pic_rt.append(rt_2)
                pic_intensity.append(int_2)
                pic_mz.append(mz_2)
                pic_index.append(max_int_index)
                pic_1 = np.transpose(np.array([rt_2, int_2, mz_2]))
                  
                for t1 in range(max_int_index):
                    if pic_1[:,1][abs(t1-max_int_index)] == 0:
                        break
                    else:
                        t1 = max_int_index+1
                for t2 in range(len(pic_1)-max_int_index):
                    if pic_1[:,1][abs(t2+max_int_index)] == 0:
                        break
                    else:
                        t2 = len(pic_1)-max_int_index
                pic_2 = np.delete(pic_1,range((t2+max_int_index),len(pic_1)),axis=0)
                pic_3 = np.delete(pic_2,range(0,(max_int_index-t1+1)),axis=0) 
                pics.append(pic_3)
    # for i in range(len(pics)):
    #     np.savetxt('%s/%s.txt' % ('D:/Dpic/data2/leaf_seed/pics20',i), pics[i])  
    # np.savetxt('D:/Dpic/data2/leaf_seed/scantime20/rt20.txt', rt)
    return pics

if __name__ == '__main__':
    path = "D:/Dpic/data2/leaf_seed/data/0_100_0_10.mzXML"
    choose_spec0, rt, rt_mean_interval = readms(path)
    choose_spec = get_range(choose_spec0, mass_inv = 1, rt_inv = 15, min_intensity=400)#ninanjie
    # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    p = Pool(5)
    array = p.map(get_array,choose_spec)
    model.load_weights('D:/Dpic/data/train/best_unet2_zz.h5')
    preds = model.predict(scaler(array),batch_size=4,verbose=1)
    pred_array_int = pred_array(0, preds, array)
    pred_array_mz = pred_array(2, preds, array)