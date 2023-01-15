import tensorflow as tf
import random

class Pang(tf.optimizers.SGD):
    """Penalty Amplified Noise in Gradient optimization algorithm.
    """
    def __init__(self, amp:float = 1e-1, **kwargs):
        super(Pang, self).__init__(**kwargs)
        self._set_hyper('amp', amp)
    
    def _compute_gradients(
        self,
        loss,
        var_list,
        grad_loss = None,
        tape:tf.GradientTape = None
    ):
        """Compute gradients of loss for the variables in var_list.

        Args:
            penalty (float): Penalty value, e.g. generated in episode run.
            var_list (list[tf.Variable]): Variables (e.g. weights) to calculate grads for.

        Returns:
            list of (gradient, variable): _description_
        """
        
        if not callable(loss) and tape is None:
            raise ValueError(
                "`tape` is required when a `Tensor` loss is passed. "
                f"Received: loss={loss}, tape={tape}."
            )
        tape = tape if tape is not None else tf.GradientTape()

        if callable(loss):
            with tape:
                if not callable(var_list):
                    tape.watch(var_list)
                loss = loss()
                if callable(var_list):
                    var_list = var_list()

        with tape:
            loss = self._transform_loss(loss)
        
        _out = []
        amp = self._get_hyper('amp')
        var_list = tf.nest.flatten(var_list)
        for n, item in enumerate(var_list):
            _pseudograd = tf.random.normal(
                shape=tf.shape(item),
                mean=0.5,
                stddev=amp*loss
            )
            
            pseudograd = tf.multiply(_pseudograd, item)
            
            _out.append(pseudograd)
        if grad_loss:
            _out = tf.convert_to_tensor(_out)
            _out = tf.multiply(grad_loss, _out)
        return zip(_out, var_list)
