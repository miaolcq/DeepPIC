library(KPIC)
first_file_name <- list.files("D:/Dpic/data2/leaf_seed/pics/pics01")      
dir <- paste("D:/Dpic/data2/leaf_seed/pics/pics01/",first_file_name,sep = "")            
n <- length(dir) 
data1 <- read.table(file = dir[1])
data2 <- as.matrix(data1)
data<-list(data2)
for (i in 2:n){
  new_data1 = read.table(file = dir[i])
  new_data2 = as.matrix(new_data1)
  new_data = list(new_data2)
  data = c(data, new_data)
}
pics1 <- list(data)
path1 <- "D:/Dpic/data2/leaf_seed/data/1.mzXML"
path_scantime <- "D:/Dpic/data2/leaf_seed/scantime/scantime1/rt1.txt"
scantime11 <- read.table(path_scantime)
scantime22 <- as.matrix(scantime11)
scantime1 <- list(scantime22)
pics5 = list(path = path1,scantime = scantime1[[1]],
             pics=pics1[[1]])
# path_peaks <- "D:/Dpic/data2/leaf_seed/peaks/peaks1/peaks1.txt"
# peaks1 <- read.table(path_peaks)
# peakIndex = peaks1[1,1]
# snr = peaks1[1,2]
# signals = peaks1[1,3]
# peaks = list(peakIndex = peakIndex,snr=snr,signals=signals)
# peaks2 <- list(peaks)
# for (i in 2:(length(peaks1[,1]))){
#   peakIndex = peaks1[i,1]
#   snr = peaks1[i,2]
#   signals = peaks1[i,3]
#   peak = list(peakIndex = peakIndex,snr=snr,signals=signals)
#   peak2 <- list(peak)
#   peaks2 = cbind(peaks2, peak2)
# }
# peaks1 <- list(peaks2)
#####
first_file_name <- list.files("D:/Dpic/data2/leaf_seed/pics/pics02")      
dir <- paste("D:/Dpic/data2/leaf_seed/pics/pics02/",first_file_name,sep = "")            
n <- length(dir) 
data1 <- read.table(file = dir[1])
data2 <- as.matrix(data1)
data<-list(data2)
for (i in 2:n){
  new_data1 = read.table(file = dir[i])
  new_data2 = as.matrix(new_data1)
  new_data = list(new_data2)
  data = c(data, new_data)
}
pics2 <- list(data)
p<-list(path=path,pics=pics[[1]])
path2 <- "D:/Dpic/data2/leaf_seed/data/2.mzXML"
path_scantime <- "D:/Dpic/data2/leaf_seed/scantime/scantime2/rt2.txt"
scantime11 <- read.table(path_scantime)
scantime22 <- as.matrix(scantime11)
scantime2 <- list(scantime22)
pics4 = list(path = path2,scantime = scantime2[[1]],
             pics=pics2[[1]])
pics_1 = list(pics5, pics4)
#循环
filenames <- dir("D:/Dpic/data2/leaf_seed/pics", full.names = T)
path_scantime <- dir("D:/Dpic/data2/leaf_seed/scantime", full.names = T)
for(j in 3:length(filenames)){
  path <- paste("D:/Dpic/data2/leaf_seed/data/",j,".mzXML",sep = "")
  dir <-dir(filenames[j], full.names = T)
  rtdir <-dir(path_scantime[j], full.names = T)
  scantime11 <- read.table(rtdir[1])
  scantime22 <- as.matrix(scantime11)
  scantime <- list(scantime22)
  data1 <- read.table(file = dir[1])
  data2 <- as.matrix(data1)
  data<-list(data2)
  for(i in 2:length(dir)){
    n <- length(dir) 
    data1 <- read.table(file = dir[i])
    data2 <- as.matrix(data1)
    new_data<-list(data2)
    data = c(data, new_data)
  }
  pics4 = list(path = path,scantime = scantime[[1]],
               pics=data)
  pics_1[[j]] = pics4
}

PICS <- PICset.decpeaks(pics_1)
PICS <- PICset.split(PICS)
PICS <- PICset.getPeaks(PICS)
groups_raw <- PICset.group(PICS, tolerance = c(0.01, 10))
groups_align <- PICset.align(groups_raw, method='fftcc',move='loess')
groups_align <- PICset.group(groups_align$picset,  tolerance = c(0.01, 10))
groups_align <- PICset.align(groups_align, method='fftcc',move='direct')
groups_align <- groupCombine(groups_align, type='isotope')
data <- getDataMatrix(groups_align)
data <- fillPeaks.EIBPC(data)
labels <- c(rep('leaf',10), rep('seed',10))
analyst.RF(labels, data$data.mat)
analyst.OPLS(labels, data$data.mat)
write.csv(data$data.mat,file="D:/Dpic/data2/leaf_seed/s111.csv",
          quote=T,row.names = T)




