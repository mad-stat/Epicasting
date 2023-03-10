#' @title Ensemble Wavelet Neural Network for Time Series Forecasting

#' A wavelet based auto-regressive neural network architecture for forecasting univariate  
#' non-stationary, non-linear, and long-term dependent time series

#' Maximal overlap discrete wavelet transform (MODWT)-based additive decomposition is applied 
#' to the time series \code{y} using pyramid algorithm (specified by \code{FastFlag}) and 
#' \code{boundary} condition to decompose the series into \code{Waveletlevels} levels. 
#' A feed-forward neural network architecture is fitted to each of the \code{Waveletlevels}
#' decomposed series with \code{MaxARParam} lagged observations as input and a single
#' hidden layer having \code{size} nodes. A total of \code{repeats} networks are fitted
#' at random starting point to obtain stable results. The model generated one-step
#' ahead forecast, multi-step ahead forecast is computed recuursively 

#' @param y A processed univariate time series object that contains data to be analyzed, for training.
#' @param Waveletlevels A predefined \code{integer} specifying the level of wavelet decomposition. 
#' Option: floor(log (base e) length(train_set)) (default).
#' @param boundary A string indicating the boundary condition of wavelet decomposition.
#' Options: "periodic"-generates wavelet decomposed series of same length as that of \code{y} (default),
#' "reflection"-generates wavelet decomposed series twice the length of \code{y}.  
#' @param FastFlag A logical indicator denoting the application of pyramid algorithm.
#' Options. "TRUE" (default), "FALSE".
#' @param MaxARParam An \code{integer} indicating the maximum number of lagged observations modeled in 
#' the EWNet architecture.  
#' @param size An \code{integer} denoting the number of hidden nodes in the single hidden layer.
#' Default: \code{size} = (\code{MaxARParam}+1)/2, to ensure stable learning.
#' @param repeats An \code{integer} representing the number of repetations of the neural network.
#' Default: 500.
#' @param: NForecast Length of the forecast horizon, an \code{integer} value.
#' @param: PI (optinal)  A logical indicator indicating the probabilistic bands
#' Options. "TRUE", "FALSE"(default).
#' @param: NVal An \code{integer} denoting the size of the validation set used for hyper-parameter tuning.
#' @return Returns an object of class "\code{ts}" representing the forecast of \code{NForecast} horizon.

@export

WaveletFitting <- function(ts,Wvlevels,bndry,FFlag)
{
  mraout <- wavelets::modwt(ts, filter='haar', n.levels=Wvlevels,boundary=bndry, fast=FFlag)
  WaveletSeries <- cbind(do.call(cbind,mraout@W),mraout@V[[Wvlevels]])
  return(list(WaveletSeries=WaveletSeries,WVSeries=mraout))
}

WaveletFittingnar<- function(ts,Waveletlevels,MaxARParam,boundary,FastFlag,NForecast)
  
{
  WS <- WaveletFitting(ts=ts,Wvlevels=Waveletlevels,bndry=boundary,FFlag=FastFlag)$WaveletSeries
  AllWaveletForecast <- NULL;AllWaveletPrediction <- NULL
  
  for(WVLevel in 1:ncol(WS))
  {
    ts <- NULL
    ts <- WS[,WVLevel]
    WaveletNARFit <- forecast::nnetar(y=as.ts(ts), p = MaxARParam, repeats = 500)
    WaveletNARPredict <- WaveletNARFit$fitted
    WaveletNARForecast <- forecast::forecast(WaveletNARFit, h=NForecast)
    AllWaveletPrediction <- cbind(AllWaveletPrediction,WaveletNARPredict)
    AllWaveletForecast <- cbind(AllWaveletForecast,as.matrix(WaveletNARForecast$mean))
  }
  Finalforecast <- rowSums(AllWaveletForecast,na.rm = T)
  FinalPrediction <- rowSums(AllWaveletPrediction,na.rm = T)
  return(list(Finalforecast=Finalforecast,FinalPrediction=FinalPrediction))
}

#' EWNet model without validation set

ewnet <- function(ts,Waveletlevels,MaxARParam,boundary,FastFlag,NForecast, PI =FALSE){ 
  n_test = NForecast 
  fit_ewnet = WaveletFittingnar(ts(ts), Waveletlevels = floor(log(length(ts))), boundary = "periodic", 
                           FastFlag = TRUE, MaxARParam, NForecast)
  if (isTRUE(PI)){
    upper = fit_ewnet$Finalforecast + 1.5*sd(fit_ewnet$Finalforecast)
    lower = fit_ewnet$Finalforecast - 1.5*sd(fit_ewnet$Finalforecast)
    forecast = data.frame("Forecast" = fit_ewnet$Finalforecast, 
                          "Lower Interval" = lower,
                          "Upper Interval" = upper)
  }else{
    forecast = data.frame("Forecast" = fit_ewnet$Finalforecast)
  }
  return(forecast)
}
 

#' EWNet model with validation set

ewnet_val <- function(ts,Waveletlevels,MaxARParam,boundary,FastFlag,NForecast,NVal, PI =FALSE){  
  train_val = subset(ts(ts),  end= length(ts(ts))-NVal)
  val = subset(ts(ts),  start= length(ts)-NVal+1)
  n_val = length(val)
  n_test = NForecast
  model_smry <- data.frame()
  for(p in 1:MaxARParam){
    fit_ewnet_val = WaveletFittingnar(ts(train_val), Waveletlevels = floor(log(length(train_val))), boundary = "periodic", 
                                      FastFlag = TRUE, MaxARParam = p, NForecast = n_val)
    fore_ewnet_val = as.data.frame(fit_ewnet_val$Finalforecast, h = n_val)
    ewnet_val_SMAPE = Metrics::smape(val, fore_ewnet_val$`fit_ewnet_val$Finalforecast`)*100
    ewnet_val_MAE = Metrics::mae(val, fore_ewnet_val$`fit_ewnet_val$Finalforecast`)
    ewnet_val_MASE = Metrics::mase(val, fore_ewnet_val$`fit_ewnet_val$Finalforecast`)
    ewnet_val_RMSE = Metrics::rmse(val, fore_ewnet_val$`fit_ewnet_val$Finalforecast`)
    ewnet_val_evaluation = data.frame(SMAPE = ewnet_val_SMAPE, MAE = ewnet_val_MAE, MASE = ewnet_val_MASE, RMSE = ewnet_val_RMSE, p)
    print(p)
    model_smry <- rbind(model_smry, ewnet_val_evaluation)
  }
  final = model_smry[which.min(model_smry$MASE),]
  fit_ewnet = WaveletFittingnar(ts(ts), Waveletlevels = floor(log(length(ts))), boundary = "periodic", 
                                FastFlag = TRUE, MaxARParam = final$p, NForecast = n_test)
    if (isTRUE(PI)){
    upper = fit_ewnet$Finalforecast + 1.5*sd(fit_ewnet$Finalforecast)
    lower = fit_ewnet$Finalforecast - 1.5*sd(fit_ewnet$Finalforecast)
    forecast = data.frame("Forecast" = fit_ewnet$Finalforecast, 
                          "Lower Interval" = lower,
                          "Upper Interval" = upper)
  }else{
    forecast = data.frame("Forecast" = fit_ewnet$Finalforecast)
  }
  return(forecast)
}

#' Implementation example
#' Fit EWNet model to the given time series data with 5 lagged observations to generate forecast of next 3 time points
#' ewnet(data, MaxARParam = 5, NForecast = 3) 
