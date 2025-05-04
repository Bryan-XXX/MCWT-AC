import tensorlayer as tl
import tensorflow as tf
import numpy as np
import wandb

OUTPUT_GRAPH = False
MAX_EPISODE = 5000
DISPLAY_REWARD_THRESHOLD = 100
MAX_EP_STEPS = 1000
RENDER = False
LAMBDA = 0.9
LR_A = 0.001
LR_C = 0.001


class Critic(object):

    def __init__(self, n_features, name, lr):
        def get_model(inputs_shape, name):
            ni = tl.layers.Input(inputs_shape, name='state')
            nn = tl.layers.Dense(n_units=64, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01),
                                 name='hidden1')(ni)
            nn = tl.layers.Dense(n_units=32, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01),
                                 name='hidden2')(nn)
            nn = tl.layers.Dense(n_units=10, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01),
                                 name='hidden3')(nn)
            nn = tl.layers.Dense(n_units=1, act=None, name='value')(nn)
            return tl.models.Model(inputs=ni, outputs=nn, name=name)

        self.model = get_model([1, n_features], name)
        self.model.train()

        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, s, r, s_):
        v_ = self.model(np.array([s_]))
        with tf.GradientTape() as tape:
            arr = np.array([s])
            arr = arr.astype(np.float32)
            v = self.model(arr)
            td_error = r + LAMBDA * v_ - v
            loss = tf.square(td_error)
        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        return td_error

    def save_ckpt(self, name):
        tl.files.save_npz(self.model.trainable_weights, name=name)

    def load_ckpt(self, name):
        tl.files.load_and_assign_npz(name=name, network=self.model)

    def get_weights(self):
        return self.model.trainable_weights
