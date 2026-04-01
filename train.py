from tensorflow.keras import backend as K


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
