# Copyright (C) 2022 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
TensorFlow NN model and a semi-random collection of required parts.
"""
import warnings

import numpy

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf
    from tensorflow_addons.callbacks import TQDMProgressBar


class RARIFLayer(tf.keras.layers.Layer):
    r"""
    A trainable RAR IF layer in the logarithmic space, i.e. with
    :math:`\log g_{\rm bar} as input and \log g_{\rm obs}` as output, given in
    Eq. 4 of [1]. Note that this layer only supports 1-dimensional input.
    The trainable parameter `loga0` is initially set to 0.

    References
    ----------
    [1] The Radial Acceleration Relation in Rotationally Supported Galaxies;
     2016; Stacy McGaugh, Federico Lelli, Jim Schombert
    """
    def __init__(self):
        super(RARIFLayer, self).__init__()
        self.log_a0 = tf.Variable(0., name="log_g0", trainable=True)

    def call(self, inputs):
        if inputs.shape[1] > 1:
            raise ValueError("`RARIFLayer` must have 1-dimensional input.")

        out = -tf.math.pow(10., 0.5 * (inputs - self.log_a0))
        out = - tf.math.exp(out)
        out = - tf.math.log1p(out) / tf.math.log(10.)
        out += inputs
        return out


class PLModel(tf.keras.Model):
    """
    A profile likelihood model with a custom training, test step and a loss
    function, propagating uncerainty in the first independent variable and the
    (single) dependent variable.
    """
    @staticmethod
    def _unpack_data(data):
        """Unpack `X`, `y`, `varx` and `vary` from `data`."""
        X, yfull = data
        y = yfull[:, 0]
        varx = yfull[:, 1]
        vary = yfull[:, 2]
        return X, y, varx, vary

    @staticmethod
    def make_weights(ypred_grad, varx, vary):
        """
        Make profile likelihood weights from the variance of both the
        independent and dependent variables.

        Parameters
        ----------
        vary : 1-dimensional array
            Variance of the dependent variable.
        varx : 1-dimensional array
            Variance of the independent variable.
        ypred_grad : 1-dimensional array
            Derivative of the dependent variable with respect to the
            independent variable.

        Returns
        -------
        weights : 1-dimensional array
        """
        return 1 / (vary + tf.square(ypred_grad) * varx)

    def train_step(self, data):
        """
        A custom training step that calculates the gradient-dependent sample
        weight.
        """
        # Note this unpacking depends on what is passed to `fit()`
        X, y, varx, vary = self._unpack_data(data)

        trainable_vars = self.trainable_variables
        # We need two gradient tapes -- one for the loss and one ypred
        with tf.GradientTape() as model_tape:
            with tf.GradientTape() as loss_tape:
                loss_tape.watch(X)
                ypred = self(X, training=True)
            # Gradient with respect to the *first* feature
            ypred_grad = loss_tape.gradient(ypred, X)[:, 0]
            # Calculate the loss
            sample_weight = self.make_weights(ypred_grad, varx, vary)
            loss = self.compiled_loss(y, ypred, sample_weight=sample_weight)
            # Calculate its gradient
        loss_grad = model_tape.gradient(loss, trainable_vars)

        # Update network weights
        self.optimizer.apply_gradients(zip(loss_grad, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, ypred,
                                           sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        A custom test step that calculates the gradient-dependent sample
        weight.
        """
        # This shoult follow `self.train_step`
        X, y, varx, vary = self._unpack_data(data)

        with tf.GradientTape() as tape:
            tape.watch(X)
            ypred = self(X, training=False)
        # Gradient with respect to the *first* feature
        ypred_grad = tape.gradient(ypred, X)[:, 0]
        sample_weight = self.make_weights(ypred_grad, varx, vary)

        # Updates the metrics tracking the loss
        self.compiled_loss(y, ypred, sample_weight=sample_weight)
        # Update the metrics.
        self.compiled_metrics.update_state(y, ypred,
                                           sample_weight=sample_weight)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def train(self, X, y, epochs, patience=100, batch_fraction=0.25,
              val_fraction=0.75, verbose=True):
        """
        Train the model. Sets the validation to be the entire training set.

        Parameters
        ----------
        X : 2-dimensional array of shape `(n_samples, n_features)`.
            Feature array.
        y : 1-dimensional array of shape `(n_samples, 3)`.
            Target array. The first column is the target value, the second is
            the variance of the first feature and the third is the variance of
            the dependent predicted variable.
        epochs : int
            Number of epochs to train for.
        patience : int, optional
            Number of epochs to wait for validation improvement before
            stopping.
        batch_fraction : float, optional
            Fraction of the training data to use in each batch.
        val_fraction : float, optional
            Fraction of the training data to use in each validation batch.
        verbose : bool, optional
            Whether to show a progress bar. By default `True`.

        Returns
        -------
        train : `keras.callbacks.History`
        """
        cbs = [tf.keras.callbacks.EarlyStopping(patience=patience,
                                                restore_best_weights=True)]
        if verbose:
            cbs.append(TQDMProgressBar(show_epoch_progress=False))

        return self.fit(x=X, y=y, validation_data=(X, y), epochs=epochs,
                        batch_size=int(batch_fraction * y.size),
                        validation_batch_size=int(val_fraction * y.size),
                        verbose=0, callbacks=cbs, )

    def predict(self, X, return_grad=False):
        """
        Predict `y` and optionally its gradient with respect to `X`.

        Parameters
        ----------
        X : 2-dimensional array
            Feature matrix.
        return_grad : bool, optional
            Whether to return the gradient of the prediction with respect to
            features. By default `False`.

        Returns
        -------
        y : 1-dimensional array
            Predicted value.
        ypred_grad : n-dimensional array, optional
            Gradient of the predicted value with respect to `X`.
        """
        if isinstance(X, numpy.ndarray):  # Enforce we have a tensor
            X = tf.convert_to_tensor(X, dtype="float32")

        with tf.GradientTape() as tape:
            tape.watch(X)
            ypred = self(X, training=False)

        if return_grad:
            ypred_grad = tape.gradient(ypred, X)
            return ypred, ypred_grad

        return ypred

    def mcdropout_predict(self, X, ytrue=None, nsamples=100):
        """
        Predict `y` and its gradient with MC dropout, optionally also score the
        model if `ytrue` is provided


        Parameters
        ----------
        X : 2-dimensional array
            Feature matrix.
        ytrue : 2-dimensional array, optional
            True value of `y`. If provided, the score is also returned.
        n_samples : int, optional
            Number of samples to average over. By default 100.

        Returns
        -------
        ypred_mean : 1-dimensional array
            MC dropout mean of the predicted value.
        ypred_std : 1-dimensional array
            MC dropout standard deviation of the predicted value.
        ypred_grad_mean : n-dimensional array
            MC dropout mean of the gradient of the predicted value with respect
            to `X`.
        ypred_grad_std : n-dimensional array
            MC dropout standard deviation of the gradient of the predicted
            value with respect to `X`.
        scores_mean : float, optional
            MC dropout mean of the score. Returned only if `ytrue` is provided.
        scores_std : float, optional
            MC dropout standard deviation of the score. Returned only if
            `ytrue` is provided.
        """
        # Enforce we have a tensor
        if isinstance(X, numpy.ndarray):
            X = tf.convert_to_tensor(X, dtype="float32")
        if ytrue is not None:
            ytrue, varx, vary = ytrue[:, 0], ytrue[:, 1], ytrue[:, 2]
        else:
            ytrue, varx, vary = None, None, None

        ypred = [None] * nsamples
        ypred_grad = [None] * nsamples
        scores = [None] * nsamples

        for i in range(nsamples):
            with tf.GradientTape() as tape:
                tape.watch(X)
                ypred[i] = self(X, training=True)

            ypred_grad[i] = tape.gradient(ypred[i], X)

            if ytrue is not None:
                scores[i] = self._score(ytrue, ypred[i], ypred_grad[i],
                                        varx, vary)

        # Calculate the mean and std of the predictions
        ypred = tf.stack(ypred)
        ypred_mean = tf.math.reduce_mean(ypred, axis=0)
        ypred_std = tf.math.reduce_std(ypred, axis=0)
        out = (ypred_mean, ypred_std,)

        # Calculate the mean and std of the gradients
        ypred_grad = tf.stack(ypred_grad)
        ypred_grad_mean = tf.math.reduce_mean(ypred_grad, axis=0)
        ypred_grad_std = tf.math.reduce_std(ypred_grad, axis=0)
        out += (ypred_grad_mean, ypred_grad_std,)

        if ytrue is not None:
            scores_mean = tf.math.reduce_mean(scores).numpy()
            scores_std = tf.math.reduce_std(scores).numpy()
            out += (scores_mean, scores_std,)

        return out

    def _score(self, ytrue, ypred, ypred_grad, varx, vary):
        """Shortcut to evaluate the PL loss."""
        # Make sure these are 1D arrays
        ypred = ypred[:, 0] if ypred.ndim > 1 else ypred
        ypred_grad = ypred_grad[:, 0] if ypred_grad.ndim > 1 else ypred_grad

        weight = self.make_weights(ypred_grad, varx, vary)
        loss = weight * tf.square(ytrue - ypred)
        return 0.5 * tf.math.reduce_sum(loss).numpy()

    def score(self, X, y):
        """
        Calculate the PL loss.

        Parameters
        ----------
        X : 2-dimensional array of shape `(n_samples, n_features)`.
            Feature array.
        y : 1-dimensional array of shape `(n_samples, 3)`.
            Target array. The first column is the target value, the second is
            the variance of the first feature and the third is the variance of
            the dependent predicted variable.

        Returns
        -------
        score : float
        """
        ytrue, varx, vary = y[:, 0], y[:, 1], y[:, 2]
        ypred, ypred_grad = self.predict(X, return_grad=True)
        return self._score(ytrue, ypred, ypred_grad, varx, vary)

    @staticmethod
    def make_network(Xtrain, layers, dropout_rate=None, add_rarif=False):
        r"""
        Shortcut to create a network.

        Parameters
        ----------
        Xtrain : 2-dimensional array of shape `(n_samples, n_features)`.
            Feature array, required to initialise the normalisation layer.
        layers : int or list of int
            The shape of the deep network.
        dropout_rate : float, optional
            The dropout rate to be appended after each layer. By default no
            dropout.
        add_rarif : bool, optional
            Whether to replace the linear connection with the `RARIF` layer.

        Returns
        -------
        input : `KerasTensor`
            The input layer of the network.
        output : `KerasTensor`
            The output layer of the network.
        """
        norm = tf.keras.layers.Normalization(axis=-1)
        norm.adapt(Xtrain)

        inp = tf.keras.layers.Input(shape=(Xtrain.shape[1], ))
        deep = norm(inp)  # Normalise the inputs
        # Add the deep layers with optional dropout
        for i in range(len(layers)):
            deep = tf.keras.layers.Dense(
                layers[i], activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                kernel_initializer=tf.keras.initializers.HeNormal(),
                kernel_regularizer="l2", bias_regularizer="l2",
                activity_regularizer="l2")(deep)

            if dropout_rate is not None:
                deep = tf.keras.layers.Dropout(rate=dropout_rate)(deep)
        deep = tf.keras.layers.Dense(1)(deep)

        if add_rarif:
            dirconnect = RARIFLayer()(inp)
            pass
        else:
            dirconnect = tf.keras.layers.Dense(1)(inp)

        out = tf.keras.layers.Add()([dirconnect, deep])
        return inp, out

    @classmethod
    def from_hyperparams(cls, Xtrain, layers, opt, dropout_rate,
                         add_rarif=False):
        """
        Shortcut to get a compiled model directly from hyperparameters.

        Parameters
        ----------
        Xtrain : 2-dimensional array of shape `(n_samples, n_features)`.
            Feature array, required to initialise the normalisation layer.
        layers : int or list of int
            The shape of the deep network.
        opt : optimizer
            NN weights optimiser.
        dropout_rate : float, optional
            The dropout rate to be appended after each layer.  If `None` then
            no dropout is used.
        add_rarif : bool, optional
            Whether to replace the linear connection with the `RARIF` layer.

        Returns
        -------
        model : :py:class:`PLModel`
            The compiled NN model.
        """
        inp, out = cls.make_network(Xtrain, layers, dropout_rate,
                                    add_rarif=add_rarif)
        model = cls(inputs=inp, outputs=out)
        model.compile(opt, loss=tf.keras.losses.MeanSquaredError(),
                      jit_compile=True)
        return model
