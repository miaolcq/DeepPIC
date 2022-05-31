
#峰检测内函数
getNoise <- function(peaks, cwt2d, ridges){
  row_one <- row_one_del <- cwt2d[1,]
  del <- which(abs(row_one) < 10e-5)
  if (length(del)>0){
    row_one_del <- row_one[-del]
  }
  
  t <- 3*median(abs(row_one_del-median(row_one_del)))/0.67
  row_one[row_one > t] <- t
  row_one[row_one < -t] <- -t
  
  noises <- sapply(1:length(peaks),function(s){
    hf_win <- length(ridges$ridges_rows)
    win_s <- max(1, peaks[s] - hf_win)
    win_e <- min(ncol(cwt2d), peaks[s] + hf_win)
    return(as.numeric(quantile(abs(row_one[win_s:win_e]),0.9)))
  })
  return(noises)
}

cwtft <- function(val) {
  .Call('_KPIC_cwtft', PACKAGE = 'KPIC', val)
}

ridgesDetection <- function(cwt2d, val) {
  .Call('_KPIC_ridgesDetection', PACKAGE = 'KPIC', cwt2d, val)
}

peaksPosition <- function(val, ridges, cwt2d) {
  .Call('_KPIC_peaksPosition', PACKAGE = 'KPIC', val, ridges, cwt2d)
}

getSignal <- function(cwt2d, ridges, peaks) {
  .Call('_KPIC_getSignal', PACKAGE = 'KPIC', cwt2d, ridges, peaks)
}

peak_detection <- function(vec, min_snr, level=0){
  cwt2d <- cwtft(vec)
  sca <- cwt2d$scales
  cwt2d <- cwt2d$cwt2d
  ridges <- ridgesDetection(cwt2d, vec)
  if (length(ridges$ridges_rows)<1){return(NULL)}
  peaks <- peaksPosition(vec, ridges, cwt2d)
  signals <- getSignal(cwt2d, ridges, peaks)
  lens <- signals$ridge_lens
  lens[lens<0] <- 0
  scales <- sca[1+lens]
  lens <- signals$ridge_lens
  signals <- signals$signals
  peaks <- peaks+1
  noises <- getNoise(peaks, cwt2d, ridges)
  snr <- (signals+10^-5)/(noises+10^-5)
  refine <- snr>min_snr & lens>3 & vec[peaks]>level
  
  info <- cbind(peaks, scales, snr)
  info <- info[refine,]
  info <- unique(info)
  if (length(info)==0){return(NULL)
  } else if (length(info)>3){
    info <- info[order(info[,1]),]
    peakIndex=info[,1]; peakScale=info[,2]; snr=info[,3]; signals=vec[info[,1]]
  } else {
    peakIndex=info[1]; peakScale=info[2]; snr=info[3]; signals=vec[info[1]]
  }
  return(list(peakIndex=peakIndex, peakScale=peakScale, snr=snr, signals=signals))
}

decPeak <- function(picss, min_snr=6, level=0){
  peaks <- lapply(picss$pics,function(pic){
    peak_detection(pic[,2], min_snr, level)
  })
  
  nps <- sapply(peaks,function(peaki){
    length(peaki$peakIndex)
  })
  pics <- picss[["pics"]][nps>0]
  peaks <- peaks[nps>0]
  gc()
  
  picss[["pics"]] <- pics
  picss[["peaks"]] <- peaks
  output <- list(path=picss[["path"]], scantime=picss[["scantime"]], 
                 pics=picss$pics, peaks=picss$peaks)
  # if (export){
  #   exportJSON <- toJSON(pics10[[1]])
  #   splitname <- strsplit(filename,"\\.")[[1]][1]
  #   outpath <- paste(splitname,'json',sep='.')
  #   write(exportJSON,outpath)
  # }
  return(output)
}

##峰检测
PICset.decpeaks <- function(picset, min_snr=6, level=0){
  for (i in 1:length(picset)){
    picset[[i]] <- decPeak(picset[[i]])
  }
  return(picset)
}

##得到峰表
integration <- function(x,yf){
  n <- length(x)
  integral <- 0.5*sum((x[2:n] - x[1:(n-1)]) * (yf[2:n] + yf[1:(n-1)]))
  return(integral)
}

getPeaks <- function(pics){
  mzinfo <- lapply(pics$pics,function(pic){
    mz <- mean(pic[,3], na.rm=TRUE)
    mzmin <- min(pic[,3], na.rm=TRUE)
    mzmax <- max(pic[,3], na.rm=TRUE)
    mzrsd <- sd(pic[,3], na.rm=TRUE)/mz*10^6
    c(mz,mzmin,mzmax,mzrsd)
  })
  
  rt <- sapply(pics$pics,function(pic){
    pic[which.max(pic[,2]),1]
  })
  
  snr <- sapply(pics$peaks,function(peaki){
    peaki$snr[which.max(peaki$signals)]
  })
  snr <- round(snr,2)
  
  maxo <- sapply(pics$pics,function(pic){
    max(pic[,2])
  })
  
  rtmin <- sapply(pics$pics,function(pic){
    pic[1,1]
  })
  rtmax <- sapply(pics$pics,function(pic){
    pic[nrow(pic),1]
  })
  
  area <- sapply(pics$pics,function(pic){
    round(integration(pic[,1],pic[,2]))
  })
  
  mzinfo <- round(do.call(rbind,mzinfo),4)
  colnames(mzinfo) <- c('mz','mzmin','mzmax','mzrsd')
  
  peakinfo <- cbind(rt,rtmin,rtmax,mzinfo,maxo,area,snr)
  pics$peakinfo <- peakinfo
  
  return(pics)
}

PICset.getPeaks <- function(picset){
  for (i in 1:length(picset)){
    picset[[i]] <- getPeaks(picset[[i]])
  }
  return(picset)
}