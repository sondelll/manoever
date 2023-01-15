import tensorflow as tf

from .m_series import M31, M32
from .p_series import P6, P7

def get_mk_1() -> tuple[tf.keras.Model, tf.keras.Model]:
    """Loads and/or instances predefined models for self driving.

    Returns:
        tuple[tf.keras.Model, tf.keras.Model]: (Agent model, Predictor model)
    """
    try:
        predictor = tf.keras.models.load_model("./data/saved_models/P6") #PP6()
    except:
        predictor = P6()

    try:
        agent = tf.keras.models.load_model("./data/saved_models/M31") # M17()
    except:
        agent = M31(predictor, visual_dims=(256,32))
        
    return agent, predictor

def get_mk_2() -> tuple[tf.keras.Model, tf.keras.Model]:
    """Loads and/or instances predefined models for self driving.

    Returns:
        tuple[tf.keras.Model, tf.keras.Model]: (Agent model, Predictor model)
    """
    try:
        predictor = tf.keras.models.load_model("./data/saved_models/P7")
    except:
        predictor = P7()

    try:
        agent = tf.keras.models.load_model("./data/saved_models/M32")
    except:
        agent = M32(predictor, visual_dims=(256,32))
        
    return agent, predictor
