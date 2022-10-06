#' @title Implementation of different forecasting models from literature
#' @param \code{train} is a time series object for training the model
#' @param n an \code{integer} value indicating the desired forecast horizons
#' @library required - Metrics, forecast, tsdyn, nnfor, WaveletArima, bsts

#' Evaluation Matrix
performance = data.frame()
model_summary = function(test, output, model){
  SMAPE = smape(as.vector(test),output)*100
  MAE = mae(as.vector(test),output)
  MASE = mase(as.vector(test),output)
  RMSE = rmse(as.vector(test),output)
  Evaluation = data.frame(Model = model, SMAPE = SMAPE, MAE = MAE, MASE = MASE, RMSE = RMSE)
  return(Evaluation)
}

#' ARIMA 
fitARIMA = auto.arima(train) 
predARIMA = forecast::forecast(fitARIMA,h=n)
ARIMA_summary = model_summary(test, predARIMA$mean, model = "ARIMA")
performance = rbind(performance, ARIMA_summary)

#' ETS  
fitETS = ets(train)
predETS = forecast::forecast(fitETS, h=n)
ETS_summary = model_summary(test, predETS$mean, model = "ETS")
performance = rbind(performance, ETS_summary)

#' SETAR
fit_SETAR = setar(train, m = 4)
fc_SETAR = predict(fit_SETAR, n.ahead = n)
SETAR_summary = model_summary(test, fc_SETAR, model = "SETAR")
performance = rbind(performance, SETAR_summary)

#' TBATS 
fit_tbats = tbats(train)
predTBATS=forecast::forecast(fit_tbats, h= n)
TBATS_summary = model_summary(test, predTBATS$mean, model = "TBATS")
performance = rbind(performance, TBATS_summary)

#' Theta 
fit_theta=thetaf(train, h= n)
Theta_summary = model_summary(test, fit_theta$mean, model = "Theta")
performance = rbind(performance, Theta_summary)

#' MLP/ANN 
fit_ANN = mlp(train)
predANN = forecast::forecast(fit_ANN, h= n)
ANN_summary = model_summary(test, predANN$mean, model = "ANN")
performance = rbind(performance, ANN_summary)

#' ARNN 
fit_ARNN = nnetar(train)
predARNN=forecast::forecast(fit_ARNN, h= n)
ARNN_summary = model_summary(test, predARNN$mean, model = "ARNN")
performance = rbind(performance, ARNN_summary)

#' Wavelet ARIMA (WARIMA)
fit_wa <- WaveletFittingarma(train, Waveletlevels = floor(log(length(train))), boundary = 'periodic', FastFlag = TRUE, MaxARParam = 5,
                             MaxMAParam = 5, NForecast = n)
WA_summary = model_summary(test, fit_wa$Finalforecast, model = "Wavelet ARIMA")
performance = rbind(performance, WA_summary)

#' BSTS 
ss <- AddLocalLinearTrend(list(), train)
fit_bsts=bsts(train,state.specification = ss, niter = 1000)
predBSTS <- predict(fit_bsts, horizon = n)
burn <- SuggestBurn(0.1, fit_bsts)
fitted_bsts=as.numeric(-colMeans(fit_bsts$one.step.prediction.errors[-(1:burn),])+train)
BSTS_summary = model_summary(test, predBSTS$mean, model = "BSTS")
performance = rbind(performance, BSTS_summary)

#' Hybrid ARIMA-ANN 
fit_res_ANN=mlp(fitARIMA$residuals)
pred_res_ANN = forecast::forecast(fit_res_ANN, h= n)
pred_arima_ann = predARIMA$mean+pred_res_ANN$mean
ARIMA_ANN_summary = model_summary(test, pred_arima_ann, model = "ARIMA ANN")
performance = rbind(performance, ARIMA_ANN_summary)

#' Hybrid ARIMA-ARNN
fit_res_ARNN=nnetar(fitARIMA$residuals)
pred_res_ARNN = forecast::forecast(fit_res_ARNN, h= n)
pred_arima_arnn=predARIMA$mean+pred_res_ARNN$mean
ARIMA_ARNN_summary = model_summary(test, pred_arima_arnn, model = "ARIMA ARNN")
performance = rbind(performance, ARIMA_ARNN_summary)

#' Hybrid ARIMA-WARIMA
fit_res_wbf=WaveletFittingarma(fitARIMA$residuals, Waveletlevels = floor(log(length(train))), boundary = 'periodic', FastFlag = TRUE, MaxARParam = 5,
                               MaxMAParam = 5, NForecast = n)
pred_arima_wbf=predARIMA$mean+fit_res_wbf$Finalforecast
ARIMA_WBF_summary = model_summary(test, pred_arima_wbf, model = "ARIMA WBF")
performance = rbind(performance, ARIMA_WBF_summary)
