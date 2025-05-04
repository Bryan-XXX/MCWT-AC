import numpy as np
import tensorflow as tf
import tensorlayer as tl

OUTPUT_GRAPH = False
MAX_EPISODE = 5000
DISPLAY_REWARD_THRESHOLD = 100
MAX_EP_STEPS = 1000
RENDER = False
LAMBDA = 0.9


class Actor(object):

    def __init__(self, n_features, n_actions, name, lr):

        def get_model(inputs_shape, name):
            ni = tl.layers.Input(inputs_shape, name='state')
            nn = tl.layers.Dense(n_units=64, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden1')(ni)
            nn = tl.layers.Dense(n_units=32, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden2')(nn)
            nn = tl.layers.Dense(n_units=10, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden3')(nn)
            nn = tl.layers.Dense(n_units=n_actions, name='actions')(nn)
            return tl.models.Model(inputs=ni, outputs=nn, name=name)

        self.model = get_model([None, n_features], name)
        self.model.train()
        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, s, a, td):
        with tf.GradientTape() as tape:
            arr = np.array([s])
            arr = arr.astype(np.float32)
            _logits = self.model(arr)
            _exp_v = tl.rein.cross_entropy_reward_loss(logits=_logits, actions=[a], rewards=td[0])
        grad = tape.gradient(_exp_v, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        return _exp_v

    def choose_action(self, s, curr, localTask):
        arr = np.array([s])
        arr = arr.astype(np.float32)
        _logits = self.model(arr)
        t = tf.nn.softmax(_logits).numpy()
        _probs = t.copy()
        for i in range(len(_probs)):
            _probs[i] = 0
        if curr in localTask:
            _probs[0][0] = 1
        else:
            _probs = t
        flag = 0
        for p in _probs[0]:
            if p != 0:
                flag = 1
                break
        if flag == 0:
            _probs[0][0] = 1
        else:
            s = sum(_probs[0])
            for i in range(len(_probs[0])):
                if _probs[0][i] != 0:
                    _probs[0][i] = _probs[0][i] / s
            s = sum(_probs[0])
            if s != 1:
                i = np.argmax(_probs[0])
                _probs[0][i] = _probs[0][i] + (1 - s)
        return tl.rein.choice_action_by_probs(_probs.ravel())

    def choose_action_greedy(self, s, curr, localTask):
        arr = np.array([s])
        arr = arr.astype(np.float32)
        _logits = self.model(arr)
        t = tf.nn.softmax(_logits).numpy()
        _probs = t.copy()
        for i in range(len(_probs)):
            _probs[i] = 0
        if curr in localTask:
            _probs[0][0] = 1
        else:
            _probs = t
        flag = 0
        for p in _probs[0]:
            if p != 0:
                flag = 1
                break
        if flag == 0:
            _probs[0][0] = 1
        else:
            s = sum(_probs[0])
            for i in range(len(_probs[0])):
                if _probs[0][i] != 0:
                    _probs[0][i] = _probs[0][i] / s
            s = sum(_probs[0])
            if s != 1:
                i = np.argmax(_probs[0])
                _probs[0][i] = _probs[0][i] + (1 - s)
        return np.argmax(_probs.ravel())

    def save_ckpt(self, name):
        tl.files.save_npz(self.model.trainable_weights, name=name)

    def load_ckpt(self, name):
        tl.files.load_and_assign_npz(name=name, network=self.model)

    def get_weights(self):
        return self.model.trainable_weights
