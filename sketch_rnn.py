import tensorflow as tf
import numpy as np
from HyperParameters import HP

def get_encoderRNN(model_name=HP.model_name, get_weights=True):
    encoder_input = tf.keras.layers.Input(shape =
                                          (HP.max_seq_length, HP.input_dimension),
                                          name = "encoder_input")

    encoderLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                    HP.enc_hidden_size, return_sequences=False,
                    name = "LSTM_encoder"), merge_mode='concat',
                    name = "BI_LSTM_encoder")(encoder_input)

    hidden_state_mean = tf.keras.layers.Dense(HP.latent_dim,
                        activation='linear', name = "mean_MLP")(encoderLSTM)

    hidden_state_variance = tf.keras.layers.Dense(HP.latent_dim,
                    activation='linear', name = "variance_MLP")(encoderLSTM)

    epsilon = tf.keras.backend.random_normal(shape=(1, HP.latent_dim))

    batch_z = hidden_state_mean + tf.exp(0.5*hidden_state_variance)*epsilon

    initial_state = tf.keras.layers.Dense(units=(2*HP.dec_hidden_size),
                    activation='tanh', name = "decoder_init_stat")(batch_z)

    h_0, c_0 = tf.split(initial_state, 2, axis=1)

    model = tf.keras.Model(encoder_input, [h_0, c_0, batch_z])
    
    if get_weights:
        model.load_weights("model/"+model_name+".h5", by_name = True)

    return model

def get_decoderRNN(critic, model_name=HP.model_name, get_weights=True):
    if critic:
        decoder_input = tf.keras.Input(shape=(1, 2*HP.input_dimension + HP.latent_dim))
    else:
        decoder_input = tf.keras.Input(shape=(1, HP.input_dimension + HP.latent_dim))
    initial_h_input = tf.keras.Input(shape=(HP.dec_hidden_size,))
    initial_c_input = tf.keras.Input(shape=(HP.dec_hidden_size,))

    decoderLSTM = tf.keras.layers.LSTM(HP.dec_hidden_size, return_sequences=True,
                                       return_state=True, name = "LSTM_decoder")

    decoder_output, h_new, c_new = decoderLSTM(decoder_input,
                            initial_state = [initial_h_input, initial_c_input])

    output_dimension = (3 + HP.M * 6)
    distribution_output = tf.keras.layers.Dense(output_dimension,
                                    name = "output_layer")(decoder_output)
    if critic:
        d1 = tf.keras.layers.Dense(64)(distribution_output)
        q_out = tf.keras.layers.Dense(1)(d1)
        model = tf.keras.Model([decoder_input, initial_h_input,
                                     initial_c_input], 
                                outputs=q_out)
        return model

    model = tf.keras.models.Model([decoder_input, initial_h_input,
                                     initial_c_input], 
                                outputs =[distribution_output , h_new, c_new])
    
    if get_weights:
        model.load_weights("model/"+model_name+".h5", by_name = True)

    return model
