#' @title Implementation of deep learning forecasting models
#' @param \code{train} is a time series object for training the model
#' @param n an \code{integer} value indicating the desired forecast horizons
#' @library required - darts 

from darts.models import TransformerModel, NBEATSModel, TCNModel, RNNModel

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
