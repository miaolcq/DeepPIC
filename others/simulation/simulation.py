# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:31:33 2022

@author: tianmiao
"""

import subprocess, os
import pandas as pd
import pyopenms

def simulation(fasta, contaminants, out, out_cntm, stddev,
               simulator = 'D:/openms/OpenMS-2.3.0/bin/MSSimulator.exe'):   
    """
        Should copy "C:\Program Files\OpenMS\share\OpenMS\examples" to working directory of Python
    """ 
   
    subprocess.call([simulator, '-in', fasta, '-out', out, '-out_cntm',out_cntm, 
               '-algorithm:MSSim:RawSignal:contaminants:file', contaminants,
               '-algorithm:MSSim:RawSignal:noise:detector:stddev', f'{stddev}',
               '-algorithm:MSSim:RawSignal:resolution:value', '5000',
               '-algorithm:MSSim:RawSignal:resolution:type', 'constant',
               '-algorithm:MSSim:Ionization:mz:lower_measurement_limit', '10',
               '-algorithm:MSSim:Ionization:mz:upper_measurement_limit', '1000',
               '-algorithm:MSSim:RT:total_gradient_time', '1000',
               '-algorithm:MSSim:RT:sampling_rate', '0.25',
               '-algorithm:MSSim:RT:scan_window:min', '0',
               '-algorithm:MSSim:RT:scan_window:max', '1000'])

def data2mzxml(path, converter = 'D:/openms/OpenMS-2.3.0/bin/FileConverter.exe'):
    if os.path.isfile(path):
        files = [path]
        path = ""
    elif os.path.isdir(path):
        files=os.listdir(path)
    for f in files:
        if f.lower().endswith(".mzdata"): 
            file_in  = path + f
            file_out = path + f[0:-6] + "mzxml"
            subprocess.call([converter, '-in', file_in, '-out', file_out])
        if f.lower().endswith(".mzml"):
            file_in  = path + f
            file_out = path + f[0:-4] + "mzxml"
            subprocess.call([converter, '-in', file_in, '-out', file_out])
            
def parse_featureXML_GT(feature_file):
    featuremap = pyopenms.FeatureMap()
    featurexml = pyopenms.FeatureXMLFile()
    featurexml.load(feature_file, featuremap) 
    hulls = pd.DataFrame(columns=['rt_min', 'rt_max', 'mz_min', 'mz_max', 'detected', 'pic_id'])   
    for i in range(featuremap.size()):
        feature = featuremap[i]
        chs = feature.getConvexHulls()
        for j in range(len(chs)):
            pts = chs[j].getHullPoints()
            hulls.loc[len(hulls)] = [pts.min(0)[0], pts.max(0)[0], pts.min(0)[1], pts.max(0)[1], False, -1]
    return hulls

def FeatureFindingMetabo(mzfile, noise_threshold_int, snr):
    finder = 'D:/openms/OpenMS-2.3.0/bin/FeatureFinderMetabo.exe'
    feature_file = 'tmp.featureXML'
    noise_threshold_int = noise_threshold_int / snr
    subprocess.call([finder, '-in', mzfile, '-out', feature_file, 
               '-algorithm:common:noise_threshold_int', f'{noise_threshold_int}',
               '-algorithm:common:chrom_peak_snr', f'{snr}',
               '-algorithm:common:chrom_fwhm', '10',
               '-algorithm:mtd:mass_error_ppm', '20',
               '-algorithm:mtd:reestimate_mt_sd', 'true',
               '-algorithm:mtd:min_sample_rate', '0',
               '-algorithm:mtd:min_trace_length', '2',
               '-algorithm:epd:width_filtering', 'off',
               '-algorithm:ffm:charge_lower_bound', '1',
               '-algorithm:ffm:charge_lower_bound', '5'])  
    featuremap = pyopenms.FeatureMap()
    featurexml = pyopenms.FeatureXMLFile()
    featurexml.load(feature_file, featuremap)
    os.remove(feature_file)
    return featuremap

def parse_featureXML_FFM(featuremap):   
    df = pd.DataFrame(columns=['rt', 'mz', 'intensity'])   
    for i in range(featuremap.size()):
        feature = featuremap[i]
        isotope_distances = feature.getMetaValue(b'isotope_distances')
        rt = feature.getRT()
        mz = feature.getMZ()
        intensity = feature.getIntensity()
        for j in range(feature.getMetaValue(b'num_of_masstraces')):
            if j == 0:
                df.loc[len(df)] = [rt, mz, intensity]
            else:
                mz_delta = isotope_distances[j-1]
                mz = mz + mz_delta
                df.loc[len(df)] = [rt, mz, intensity] 
    return df

def tic(name = ""):
    #Homemade version of matlab tic and toc functions
    import time
    global gname 
    gname = name
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print(gname, ": Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

def pics2df(pics):
    df = pd.DataFrame(columns=['rt', 'mz', 'intensity'])
    for i,pic in enumerate(pics):
        idx = pic[:,2].argmax()
        rt  = pic[idx,0]
        mz  = pic[idx,1]
        intensity = pic[idx,2]
        df.loc[len(df)] = [rt, mz, intensity] 
    return df

def peaks2df(peaks):
    df = pd.DataFrame(columns=['rt', 'mz', 'intensity'])
    for i in range(peaks.shape[0]):
        rt  = peaks[i,3]
        mz  =peaks[i,0]
        intensity = peaks[i,6]
        df.loc[len(df)] = [rt, mz, intensity] 
    return df

def match_features(ground_truths, df):
    for i in range(len(df)):
        rt  = df.at[i, 'rt']
        mz  = df.at[i, 'mz']
        for j in range(len(ground_truths)):
            if(rt >= ground_truths.at[j, 'rt_min'] and rt <= ground_truths.at[j, 'rt_max'] and
               mz >= ground_truths.at[j, 'mz_min']-0.01 and mz <= ground_truths.at[j, 'mz_max']+0.01
               ):
                ground_truths.at[j, 'detected'] = True
                ground_truths.at[j, 'pic_id'] = i

def metrics(TP, FN, FP):
    r = TP/(TP+FN)
    p = TP/(TP+FP)
    f1 = (2*r*p)/(r+p)
    return r, p, f1

mm48_all = pd.read_csv('D:/Dpic/data/MM48_annotations.csv')
mm48_all['charge'] = [1] * mm48_all.shape[0]
mm48_all['shape'] = ['gauss'] * mm48_all.shape[0]
mm48_all['source'] = ['ESI'] * mm48_all.shape[0]
mm48 = mm48_all[['Name', 'Formel','RT','RT2','Intensity','charge','shape','source']]
mm48.to_csv('D:/Dpic/data/MM48_MSSimulator.csv', header=False, index=False)

names = ['stddev','FFM_Recall', 'FFM_Precision', 'FFM_FScore']
results = pd.DataFrame(columns=names) 
parameters = [[0, 1.35],[1,13.5],[3, 40.5],[8, 108], [13, 175.5], [18, 243], [23, 310.5], [30, 405]]

openms_path = "D:/openms/OpenMS-2.3.0/bin/"

os.getcwd()
os.chdir('D:/Dpic/data')

for i,p in enumerate(parameters):   
    simulation('D:/Dpic/data/test.fasta','D:/Dpic/data/MM48_MSSimulator.csv', 'D:/Dpic/data/MM48_MSS_Profile.mzML', 'D:/Dpic/data/MM48_MSS.featureXML', p[0]) 
    peak_picker = 'D:/openms/OpenMS-2.3.0/bin/PeakPickerHiRes.exe'
    subprocess.call([peak_picker,'-in', 'D:/Dpic/data/MM48_MSS_Profile.mzML','-out', 
                     'D:/Dpic/data/MM48_MSS.mzML'])
    data2mzxml('D:/Dpic/data/MM48_MSS.mzML')
    ground_truths = parse_featureXML_GT('D:/Dpic/data/MM48_MSS.featureXML')
        
    mzfile =  'D:/Dpic/data/MM48_MSS.mzXML'
    mzMLfile =  'D:/Dpic/data/MM48_MSS.mzML'
    
    tic()
    feature_map = FeatureFindingMetabo(mzMLfile, p[1], 3)
    df_ffm = parse_featureXML_FFM(feature_map)
    toc()
    
    match_ffm = ground_truths.copy()
    match_features(match_ffm, df_ffm)
    m = match_ffm.detected.value_counts().values      
    m_r, m_p,m_f = metrics(m[0], m[1], df_ffm.shape[0]-m[0])

    results.loc[len(results)] = [p[0],m_r,m_p,m_f] 
    result_rd = results.round(4) 