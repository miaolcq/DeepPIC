# DeepPIC
Deep Learning-based Pure Ion Chromatogram Extraction for LC-MS.
Liquid chromatography coupled with mass spectrometry (LC-MS) can detect thousands of features in metabolomics samples. Traditional extracted ion chromatograms and region of interest (ROI) methods can split peaks due to fixed m/z bias. Pure ion chromatograms (PIC) are ion traces consisting of ions from the same analyte, and PIC-based methods have been proposed to address the split peak problem in traditional methods. A deep learning based pure ion chromatogram method (DeepPIC) has been developed to extract PICs directly from raw LC-MS files.
The method has a high sensitivity and precision, reliable quantification capability and wide adaptability. It provides the entire pipeline from raw data to discriminant model for metabolomic datasets.
![fig_1](https://user-images.githubusercontent.com/49331604/167531194-0f24c5e3-13f9-4edc-9ecd-f3262b71c0c7.png)
# Installation
Python and TensorFlow:

Python 3.8.8 and TensorFlow (version 2.2.0-GPU)

The main packages can be seen in requirements.txt

Install Anaconda https://www.anaconda.com/

Install main packages in requirements.txt with following commands

 conda create --name DeepPIC python=3.8.13
 conda activate DeepPIC
 python -m pip install -r requirements.txt
 pip install tqdm
# Clone the repo and run it directly
git clone at：https://github.com/miaolcq/DeepPIC.git

# Download the model and run directly
As the model and the data exceeded the limits, we have uploaded the model and all data (MM48, simulated MM48, quantitative and different instrument datasets) to Zenodo.

10.5281/zenodo.6535058

Training the model and extracting PICs from the LC-MS files.

Run the file 'model.py'. The model and these data have been uploaded at Zenodo. Download the model and these example data, DeepPIC can be reload and predict easily.

# Contact
Miao Tian: miaolcq@csu.edu.cn
