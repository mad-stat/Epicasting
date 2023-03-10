#' @title Implementation of deep learning forecasting models
#' @param \code{train} is a time series object for training the model
#' @param n an \code{integer} value indicating the desired forecast horizons
#' @library required - sktime, sklearn, darts, pdb, pywt 

#' SVR
from sklearn.svm import SVR
from sktime.forecasting.compose import make_reduction
regressor = SVR(C=1.0, epsilon=0.2)
forecaster = make_reduction(regressor, window_length=15, strategy="recursive")
forecaster.fit(train)
pred = forecaster.predict(n)

#' Advanced Deep Learners
from darts.models import TransformerModel, NBEATSModel, TCNModel, RNNModel

#' LSTM
model = RNNModel(input_chunk_length = 10, output_chunk_length=3, model='LSTM', n_epochs=100, hidden_dim=25)
model.fit(train)
pred = model.predict(n)

#' NBeats 
model_nbeats = NBEATSModel(input_chunk_length=15, output_chunk_length=7, random_state=42)
model_nbeats.fit([train], epochs=100, verbose=True)
pred_nbeats = model_nbeats.predict(series=train, n)

#' Transformers
my_tf_model = TransformerModel(
    input_chunk_length=15,
    output_chunk_length=7 ,
    batch_size=32,
    n_epochs=100,
    model_name="transformer",
    nr_epochs_val_period=10,
    d_model=16,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    activation="relu",
    random_state=42,
    save_checkpoints=True,
    force_reset=True,
)
my_tf_model.fit(series=train,  verbose=True)
pred_tf = my_tf_model.predict(n)

#' TCN 
deep_tcn = TCNModel(
    input_chunk_length=15,
    output_chunk_length=7,
    kernel_size=2,
    num_filters=4,
    dilation_base=2,
    dropout=0,
    random_state=0,
    likelihood=GaussianLikelihood(),
)
deep_tcn.fit(train, verbose=True)
pred_tcn = deep_tcn.predict(n)

#' DeepAR 
my_dar_model = RNNModel(
    model="LSTM",
    hidden_dim=20,
    dropout=0,
    batch_size=16,
    n_epochs=300,
    optimizer_kwargs={"lr": 1e-3},
    model_name="RNN",
    log_tensorboard=True,
    random_state=42,
    training_length=20,
    input_chunk_length=14,
    force_reset=True,
    save_checkpoints=True,
)
my_dar_model.fit(train, verbose=True)
pred_dar = my_dar_model.predict(n)

#' Wavelet Based Models
import numpy as np
import pdb
import pywt

def upArrow_op(li, j):
    if j == 0:
        return [1]
    N = len(li)
    li_n = np.zeros(2 ** (j - 1) * (N - 1) + 1)
    for i in range(N):
        li_n[2 ** (j - 1) * i] = li[i]
    return li_n


def period_list(li, N):
    n = len(li)
    # append [0 0 ...]
    n_app = N - np.mod(n, N)
    li = list(li)
    li = li + [0] * n_app
    if len(li) < 2 * N:
        return np.array(li)
    else:
        li = np.array(li)
        li = np.reshape(li, [-1, N])
        li = np.sum(li, axis=0)
        return li


def circular_convolve_mra( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
        Modification of 
            https://stackoverflow.com/questions/35474078/python-1d-array-circular-convolution
    '''
    return np.flip(np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(np.flip(ker))))).astype(np.int).tolist()


def circular_convolve_d(h_t, v_j_1, j):
    '''
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j)
    '''
    N = len(v_j_1)
    L = len(h_t)
    w_j = np.zeros(N)
    l = np.arange(L)
    for t in range(N):
        index = np.mod(t - 2 ** (j - 1) * l, N)
        v_p = np.array([v_j_1[ind] for ind in index])
        w_j[t] = (np.array(h_t) * v_p).sum()
    return w_j


def circular_convolve_s(h_t, g_t, w_j, v_j, j):
    '''
    (j-1)th level synthesis from w_j, w_j
    see function circular_convolve_d
    '''
    N = len(v_j)
    L = len(h_t)
    v_j_1 = np.zeros(N)
    l = np.arange(L)
    for t in range(N):
        index = np.mod(t + 2 ** (j - 1) * l, N)
        w_p = np.array([w_j[ind] for ind in index])
        v_p = np.array([v_j[ind] for ind in index])
        v_j_1[t] = (np.array(h_t) * w_p).sum()
        v_j_1[t] = v_j_1[t] + (np.array(g_t) * v_p).sum()
    return v_j_1


def modwt(x, filters, level):
    '''
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return np.vstack(wavecoeff)


def imodwt(w, filters):
    ''' inverse modwt '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    level = len(w) - 1
    v_j = w[-1]
    for jp in range(level):
        j = level - jp - 1
        v_j = circular_convolve_s(h_t, g_t, w[j], v_j, j + 1)
    return v_j


def modwtmra(w, filters):
    ''' Multiresolution analysis based on MODWT'''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    # D
    level, N = w.shape
    level = level - 1
    D = []
    g_j_part = [1]
    for j in range(level):
        # g_j_part
        g_j_up = upArrow_op(g, j)
        g_j_part = np.convolve(g_j_part, g_j_up)
        # h_j_o
        h_j_up = upArrow_op(h, j + 1)
        h_j = np.convolve(g_j_part, h_j_up)
        h_j_t = h_j / (2 ** ((j + 1) / 2.))
        if j == 0: h_j_t = h / np.sqrt(2)
        h_j_t_o = period_list(h_j_t, N)
        D.append(circular_convolve_mra(h_j_t_o, w[j]))
    # S
    j = level - 1
    g_j_up = upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2 ** ((j + 1) / 2.))
    g_j_t_o = period_list(g_j_t, N)
    S = circular_convolve_mra(g_j_t_o, w[-1])
    D.append(S)
    return np.vstack(D)


if __name__ == '__main__':
    s1 = np.arange(10)
    ws = modwt(s1, 'db2', 3)
    s1p = imodwt(ws, 'db2')
    mra = modwtmra(ws, 'db2')
    
     
# W-Transformers
wt = modwt(train, 'haar', int(np.log(len(train))))
wtmra = modwtmra(wt, 'haar')

# Wavelet tranformation
series = []
train = []
val = []
#number of point to forecast
nforecast = n ############################ Change
nb_time = len(train)
#Put true if you want to scale your data
scaling = False
for i in range(len(wt)):
    wt_df = TimeSeries.from_dataframe(pd.DataFrame(wt[i]))
    series.append(wt_df)
    wt_df_train = wt_df[:nb_time-nforecast]
    train.append(wt_df_train)
    wt_df_val = wt_df[nb_time-nforecast:]
    val.append(wt_df_val)
    
    
models_transformers = []
for i in range(len(train)):
    transformers = TransformerModel(
    input_chunk_length=5,
    output_chunk_length=1,
    batch_size=32,
    n_epochs=150,
    model_name="transformer"+str(i),
    nr_epochs_val_period=10,
    d_model=16,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    activation="relu",
    random_state=42,
    save_checkpoints=True,
    force_reset=True)
    models_transformers.append(transformers.fit(series = train[i], val_series=val[i], verbose=True))
    
    
prediction = []
for i in range(len(train)):
    model_loaded = TransformerModel.load_from_checkpoint("transformer"+str(i), best=True)
    pred_series = model_loaded.historical_forecasts(series[i],
                                                    start = nb_time-nforecast,
                                                    retrain=False,
                                                    verbose=False,
                                                    )
    prediction.append(pred_series)
    
    
prediction_tmp = prediction[0].pd_dataframe()
for i in range(1,len(prediction)):
    prediction_tmp[i] = prediction[i].pd_dataframe()

pred = prediction_tmp.sum(axis = 1)  



#' W-NBeats
series = []
train = []
val = []
nforecast = n ############################ Change
nb_time = len(train)
scaling = False
for i in range(len(wt)):
    wt_df = TimeSeries.from_dataframe(pd.DataFrame(wt[i]))
    if scaling == True:
        scaler = Scaler()
        wt_df = scaler.fit_transform(wt_df)
    
    wt_df_train = wt_df[:nb_time-nforecast]
    train.append(wt_df_train)
    wt_df_val = wt_df[nb_time-nforecast:]
    val.append(wt_df_val)
    series.append(wt_df)
    
model = []
for i in range(len(train)):
    model_nbeats = NBEATSModel(
    force_reset=True,
    input_chunk_length=5,
    output_chunk_length=3,
    generic_architecture=True,
    num_stacks=10,
    num_blocks=1,
    num_layers=4,
    layer_widths=256,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=800,
    model_name="nbeats_run_modwt"+str(i),
    save_checkpoints=True
    )
    model.append(model_nbeats.fit(series = train[i],val_series=val[i], verbose=True)) 
    
    
prediction = []
for i in range(len(train)):
    model_loaded = NBEATSModel.load_from_checkpoint("nbeats_run_modwt"+str(i), best=True)
    pred_series = model_loaded.historical_forecasts(series[i],
                                                    start = nb_time-nforecast,
                                                    retrain=False,
                                                    verbose=False,
                                                    )
    prediction.append(pred_series)
    
prediction_tmp = prediction[0].pd_dataframe()
for i in range(1,len(prediction)):
    prediction_tmp[i] = prediction[i].pd_dataframe()
pred = prediction_tmp.sum(axis = 1)     
