# Alec Radford, Indico, Kyle Kastner
# License: MIT
"""
Convolutional VAE in a single file.
Bringing in code from IndicoDataSolutions and Alec Radford (NewMu)
Additionally converted to use default conv2d interface instead of explicit cuDNN
"""
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.compile.nanguardmode import NanGuardMode
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet import conv2d, softmax, relu, softplus
import tarfile
import tempfile
import gzip
import cPickle
import fnmatch
from time import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave, imread
import os
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import svd
#from skimage.transform import resize

from agents.pathsampler import Recorder
from environments.visualisers.pg2dvis import PyGame2D


def softmax(x):
    return T.nnet.softmax(x)


def rectify(x):
    return (x + abs(x)) / 2.0


def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def linear(x):
    return x


def t_rectify(x):
    return x * (x > 1)


def t_linear(x):
    return x * (abs(x) > 1)


def maxout(x):
    return T.maximum(x[:, 0::2], x[:, 1::2])


def clipped_maxout(x):
    return T.clip(T.maximum(x[:, 0::2], x[:, 1::2]), -1., 1.)


def clipped_rectify(x):
    return T.clip((x + abs(x)) / 2.0, 0., 1.)


def hard_tanh(x):
    return T.clip(x, -1., 1.)


def steeper_sigmoid(x):
    return 1./(1. + T.exp(-3.75 * x))


def hard_sigmoid(x):
    return T.clip(x + 0.5, 0., 1.)


def shuffle(*data):
    idxs = np.random.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]


def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    if len(data[0]) % size != 0:
        batches += 1
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])

def iter_data_tv(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = data[0].shape[0].eval() / size
    if data[0].shape[0].eval() % size != 0:
        batches += 1
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


def intX(X):
    return np.asarray(X, dtype=np.int32)


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def uniform(shape, scale=0.05):
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))


def normal(shape, scale=0.05):
    return sharedX(np.random.randn(*shape) * scale)


def orthogonal(shape, scale=1.1):
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])


def color_grid_vis(X, show=True, save=False, transform=False):
    ngrid = int(np.ceil(np.sqrt(len(X))))
    npxs = np.sqrt(X[0].size/3)
    img = np.zeros((npxs * ngrid + ngrid - 1,
                    npxs * ngrid + ngrid - 1, 3))
    for i, x in enumerate(X):
        j = i % ngrid
        i = i / ngrid
        if transform:
            x = transform(x)
        img[i*npxs+i:(i*npxs)+npxs+i, j*npxs+j:(j*npxs)+npxs+j] = x
    if show:
        plt.imshow(img, interpolation='nearest')
        plt.show()
    if save:
        imsave(save, img)
    return img


def bw_grid_vis(X, show=True, save=False, transform=False):
    ngrid = int(np.ceil(np.sqrt(len(X))))
    npxs = np.sqrt(X[0].size)
    img = np.zeros((npxs * ngrid + ngrid - 1,
                    npxs * ngrid + ngrid - 1))
    for i, x in enumerate(X):
        j = i % ngrid
        i = i / ngrid
        if transform:
            x = transform(x)
        img[i*npxs+i:(i*npxs)+npxs+i, j*npxs+j:(j*npxs)+npxs+j] = x
    if show:
        plt.imshow(img, interpolation='nearest')
        plt.show()
    if save:
        imsave(save, img)
    return img


def center_crop(img, n_pixels):
    img = img[n_pixels:img.shape[0] - n_pixels,
              n_pixels:img.shape[1] - n_pixels]
    return img


def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def make_paths(n_code, n_paths, n_steps=480):
    """
    create a random path through code space by interpolating between points
    """
    paths = []
    p_starts = np.random.randn(n_paths, n_code)
    for i in range(n_steps/48):
        p_ends = np.random.randn(n_paths, n_code)
        for weight in np.linspace(0., 1., 48):
            paths.append(p_starts*(1-weight) + p_ends*weight)
        p_starts = np.copy(p_ends)

    paths = np.asarray(paths)
    return paths


def Adam(params, cost, lr=0.0001, b1=0.1, b2=0.001, e=1e-8):
    """
    no bias init correction
    """
    updates = []
    grads = T.grad(cost, params)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    return updates

class PickleMixin(object):
    def __getstate__(self):
        if not hasattr(self, '_pickle_skip_list'):
            self._pickle_skip_list = []
            for k, v in self.__dict__.items():
                try:
                    f = tempfile.TemporaryFile()
                    cPickle.dump(v, f)
                except:
                    self._pickle_skip_list.append(k)
        state = OrderedDict()
        for k, v in self.__dict__.items():
            if k not in self._pickle_skip_list:
                state[k] = v
        return state

    def __setstate__(self, state):
        self.__dict__ = state

def log_prior(mu, log_sigma):
    """
    yaost kl divergence penalty
    """
    return 0.5 * T.sum(1 + 2 * log_sigma - mu ** 2 - T.exp(2 * log_sigma))


def conv(X, w, b, activation):
    # z = dnn_conv(X, w, border_mode=int(np.floor(w.get_value().shape[-1]/2.)))
    s = int(np.floor(w.get_value().shape[-1]/2.))
    z = conv2d(X, w, border_mode='full')[:, :, s:-s, s:-s]
    if b is not None:
        z += b.dimshuffle('x', 0, 'x', 'x')
    return activation(z)

def conv_and_pool(X, w, b=None, activation=rectify):
    return max_pool_2d(conv(X, w, b, activation=activation), (2, 2))

def deconv(X, w, b=None):
    # z = dnn_conv(X, w, direction_hint="*not* 'forward!",
    #              border_mode=int(np.floor(w.get_value().shape[-1]/2.)))
    s = int(np.floor(w.get_value().shape[-1]/2.))
    z = conv2d(X, w, border_mode='full')[:, :, s:-s, s:-s]
    if b is not None:
        z += b.dimshuffle('x', 0, 'x', 'x')
    return z


def depool(X, factor=2):
    """
    luke perforated upsample
    http://www.brml.org/uploads/tx_sibibtex/281.pdf
    """
    output_shape = [
        X.shape[1],
        X.shape[2]*factor,
        X.shape[3]*factor
    ]
    stride = X.shape[2]
    offset = X.shape[3]
    in_dim = stride * offset
    out_dim = in_dim * factor * factor

    upsamp_matrix = T.zeros((in_dim, out_dim))
    rows = T.arange(in_dim)
    cols = rows*factor + (rows/stride * factor * offset)
    upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)

    flat = T.reshape(X, (X.shape[0], output_shape[0], X.shape[2] * X.shape[3]))

    up_flat = T.dot(flat, upsamp_matrix)
    upsamp = T.reshape(up_flat, (X.shape[0], output_shape[0],
                                 output_shape[1], output_shape[2]))

    return upsamp


def deconv_and_depool(X, w, b=None, activation=rectify):
    return activation(deconv(depool(X), w, b))

class ConvVAE(PickleMixin):
    def __init__(self, image_save_root=None, snapshot_file="snapshot_11_1_17.pkl"):
        self.srng = RandomStreams()
        self.n_code = 512
        self.n_hidden = 2048
        self.n_batch = 128
        self.costs_ = []
        self.epoch_ = 0
        self.snapshot_file = snapshot_file
        self.image_save_root = image_save_root
        if os.path.exists(self.snapshot_file):
            print("Loading from saved snapshot " + self.snapshot_file)
            f = open(self.snapshot_file, 'rb')
            classifier = cPickle.load(f)
            self.__setstate__(classifier.__dict__)
            f.close()

    def _setup_functions(self, trX):
        l1_e = (64, trX.shape[1], 5, 5)
        print("l1_e", l1_e)
        l1_d = (l1_e[1], l1_e[0], l1_e[2], l1_e[3])
        print("l1_d", l1_d)
        l2_e = (128, l1_e[0], 5, 5)
        print("l2_e", l2_e)
        l2_d = (l2_e[1], l2_e[0], l2_e[2], l2_e[3])
        print("l2_d", l2_d)
        # 2 layers means downsample by 2 ** 2 -> 4, with input size 28x28 -> 7x7
        # assume square
        self.downpool_sz = trX.shape[-1] // 4
        l3_e = (l2_e[0] * self.downpool_sz * self.downpool_sz,
                self.n_hidden)
        print("l3_e", l3_e)
        l3_d = (l3_e[1], l3_e[0])
        print("l4_d", l3_d)

        if not hasattr(self, "params"):
            print('generating weights')
            we = uniform(l1_e)
            w2e = uniform(l2_e)
            w3e = uniform(l3_e)
            b3e = shared0s(self.n_hidden)
            wmu = uniform((self.n_hidden, self.n_code))
            bmu = shared0s(self.n_code)
            wsigma = uniform((self.n_hidden, self.n_code))
            bsigma = shared0s(self.n_code)

            wd = uniform((self.n_code, self.n_hidden))
            bd = shared0s((self.n_hidden))
            w2d = uniform(l3_d)
            b2d = shared0s((l3_d[1]))
            w3d = uniform(l2_d)
            wo = uniform(l1_d)
            self.enc_params = [we, w2e, w3e, b3e, wmu, bmu, wsigma, bsigma]
            self.dec_params = [wd, bd, w2d, b2d, w3d, wo]
            self.params = self.enc_params + self.dec_params

        print('theano code')

        X = T.tensor4()
        e = T.matrix()
        Z_in = T.matrix()

        code_mu, code_log_sigma, Z, y = self._model(X, e)

        y_out = self._deconv_dec(Z_in, *self.dec_params)

        #rec_cost = T.sum(T.abs_(X - y))
        rec_cost = T.sum(T.sqr(X - y)) # / T.cast(X.shape[0], 'float32')
        prior_cost = log_prior(code_mu, code_log_sigma)

        cost = rec_cost - prior_cost

        print('getting updates')

        updates = Adam(self.params, cost)

        print('compiling')
        self._fit_function = theano.function([X, e], cost, updates=updates)
        self._reconstruct = theano.function([X, e], y)
        self._x_given_z = theano.function([Z_in], y_out)
        self._z_given_x = theano.function([X], (code_mu, code_log_sigma))

    def _conv_gaussian_enc(self, X, w, w2, w3, b3, wmu, bmu, wsigma, bsigma):
        h = conv_and_pool(X, w)
        h2 = conv_and_pool(h, w2)
        h2 = h2.reshape((h2.shape[0], -1))
        h3 = T.tanh(T.dot(h2, w3) + b3)
        mu = T.dot(h3, wmu) + bmu
        log_sigma = 0.5 * (T.dot(h3, wsigma) + bsigma)
        return mu, log_sigma

    def _deconv_dec(self, X, w, b, w2, b2, w3, wo):
        h = rectify(T.dot(X, w) + b)
        h2 = rectify(T.dot(h, w2) + b2)
        #h2 = h2.reshape((h2.shape[0], 256, 8, 8))
        # Referencing things outside function scope... will have to be class
        # variable
        h2 = h2.reshape((h2.shape[0], w3.shape[1], self.downpool_sz,
                        self.downpool_sz))
        h3 = deconv_and_depool(h2, w3)
        y = deconv_and_depool(h3, wo, activation=hard_tanh)
        return y

    def _model(self, X, e):
        code_mu, code_log_sigma = self._conv_gaussian_enc(X, *self.enc_params)
        Z = code_mu + T.exp(code_log_sigma) * e
        y = self._deconv_dec(Z, *self.dec_params)
        return code_mu, code_log_sigma, Z, y

    def fit(self, trX):
        if not hasattr(self, "_fit_function"):
            self._setup_functions(trX)

        xs = floatX(np.random.randn(100, self.n_code))
        print('TRAINING')
        x_rec = floatX(shuffle(trX)[:100])
        t = time()
        n = 0.
        epochs = 1000
        for e in range(epochs):
            iter_num = 0;
            #print( size(trX) );
            for xmb in iter_data(trX, size=self.n_batch):
                iter_num += 1;
                print ("In batch: " + format(iter_num) + " of " );
                xmb = floatX(xmb)
                cost = self._fit_function(xmb, floatX(
                    np.random.randn(xmb.shape[0], self.n_code)))
                self.costs_.append(cost)
                n += xmb.shape[0]
            print("Train iter", e)
            print("Total iters run", self.epoch_)
            print("Cost", cost)
            print("Mean cost", np.mean(self.costs_))
            print("Time", n / (time() - t))
            self.epoch_ += 1

            if e % 5 == 0:
                print("Saving model snapshot")
                f = open(self.snapshot_file, 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()

            def tf(x):
                return ((x + 1.) / 2.).transpose(1, 2, 0)

            if e == epochs or e % 100 == 0:
                if self.image_save_root is None:
                    image_save_root = os.path.split(__file__)[0]
                else:
                    image_save_root = self.image_save_root
                samples_path = os.path.join(
                    image_save_root, "sample_images_epoch_%d" % self.epoch_)
                if not os.path.exists(samples_path):
                    os.makedirs(samples_path)

                samples = self._x_given_z(xs)
                recs = self._reconstruct(x_rec, floatX(
                    np.ones((x_rec.shape[0], self.n_code))))
                if trX.shape[1] == 3:
                    img1 = color_grid_vis(x_rec,
                                        transform=tf, show=False)
                    img2 = color_grid_vis(recs,
                                        transform=tf, show=False)
                    img3 = color_grid_vis(samples,
                                        transform=tf, show=False)
                elif trX.shape[1] == 1:
                    img1 = bw_grid_vis(x_rec, show=False)
                    img2 = bw_grid_vis(recs, show=False)
                    img3 = bw_grid_vis(samples, show=False)

                imsave(os.path.join(samples_path, 'source.png'), img1)
                imsave(os.path.join(samples_path, 'recs.png'), img2)
                imsave(os.path.join(samples_path, 'samples.png'), img3)

                paths = make_paths(self.n_code, 3)
                for i in range(paths.shape[1]):
                    path_samples = self._x_given_z(floatX(paths[:, i, :]))
                    for j, sample in enumerate(path_samples):
                        if trX.shape[1] == 3:
                            imsave(os.path.join(
                                samples_path, 'paths_%d_%d.png' % (i, j)),
                                tf(sample))
                        else:
                            imsave(os.path.join(samples_path,
                                                'paths_%d_%d.png' % (i, j)),
                                sample.squeeze())

    def transform(self, x_rec):
                recs = self._reconstruct(x_rec, floatX(
                    np.ones((x_rec.shape[0], self.n_code))))
                return recs

    def encode(self, X, e=None):
        if e is None:
            e = np.ones((X.shape[0], self.n_code))
        return self._z_given_x(X, e)

    def decode(self, Z):
        return self._z_given_x(Z)

class MPEDQN(PickleMixin):
    def __init__(self, image_save_root=None, snapshot_file="dqn_snapshot.pkl", n_state=4, n_actions=6, n_hidden=20, gamma=0.9, sample_temp=0.5, target_sync=30):

        self.srng = RandomStreams()
        self.n_state = n_state
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.n_batch = 100
        self.target_sync = target_sync
        self.gamma = gamma
        self.costs_ = []
        self.epoch_ = 0
        self.snapshot_file = snapshot_file
        self.image_save_root = image_save_root
        self.sample_temp = sample_temp


        if os.path.exists(self.snapshot_file):
            print("Loading from saved snapshot " + self.snapshot_file)
            f = open(self.snapshot_file, 'rb')
            classifier = cPickle.load(f)
            self.__setstate__(classifier.__dict__)
            f.close()

    def _setup_functions(self):

        # Actual parameter lengths.
        #sh_w_n = (self.n_state + self.n_actions + 1, self.n_state + 1, self.n_state)
        #print("sh_w_n", sh_w_n)
        sh_w_a = (self.n_state, self.n_hidden)
        print("sh_w_a", sh_w_a)
        sh_w_a_bias = (self.n_hidden,)
        print("sh_w_a_bias", sh_w_a_bias)

        sh_w_b = (self.n_hidden, self.n_actions)
        print("sh_w_b", sh_w_b)



        if not hasattr(self, "params"):
            print('generating weights')

            # (A+1)x(S+1)xS
            wa = uniform(sh_w_a, scale=0.2)
            # (P+1)x(S+1)xR
            wb = uniform(sh_w_b, scale=0.2)
            # (P+1)x(S+1)xR
            wa_bias = shared0s(sh_w_a_bias)

            # (A+1)x(S+1)xS
            w2a = uniform(sh_w_a, scale=0.2)
            # (P+1)x(S+1)xR
            w2b = uniform(sh_w_b, scale=0.2)
            # (P+1)x(S+1)xR
            w2a_bias = shared0s(sh_w_a_bias)

            self.q_params = [wa, wb, wa_bias]
            self.target_params = [w2a, w2b, w2a_bias]
        else:
            wa, wb, wa_bias = self.q_params
            w2a, w2b, w2a_bias = self.target_params


        # Target network variables.

        #NxS
        S = sharedX( np.zeros((1,1)), name="S")
        #NxA
        A = sharedX( np.zeros((1,1)), name="A")
        #N
        R = sharedX( np.zeros((1,)), name="R")
        #NxS
        S_prime = sharedX( np.zeros((1,1)), name="S_")

        # Control variable
        # 1
        k = sharedX( 0, name="k")
        K = sharedX( 10, name="K")

        #Sv = S[k*]
        self.inputs = {"S":S,"A":A,"R":R,"S_":S_prime}
        # Q network variables
        # Do we need more complexity here?
        # NxH
        T_H = relu( T.tensordot( S_prime, w2a, axes=[1,0] ) + w2a_bias )
        # NxA
        T_Q = T.tensordot( T_H, w2b, axes=[1,0] )
        # N
        T_ = T.max(T_Q, axis=1) * self.gamma + R

        # NxH
        H_ = relu(T.tensordot(S, wa, axes=[1, 0]) + wa_bias)
        # NxA
        Q_ = T.tensordot(H_, wb, axes=[1,0])

        # NxA
        A_sample = self.srng.multinomial( pvals=softmax( Q_/self.sample_temp ) )

        # Mask the unused actions out.
        cost = T.sum( T.sqr( T.sum(Q_ * A, axis=1) - T_)  )

        # Adaptive Gradient technique 'Adam'
        updates = Adam( self.q_params, cost )



        self._fit_function = theano.function([], cost, updates=updates)
        self._sample = theano.function([], A_sample)
        self._q = theano.function([], Q_)
        # Output just the cost to check with a test set.
        self._cost = theano.function([], cost)

    def sample(self, trS):
        if not hasattr(self, "_sample"):
            self._setup_functions()
        self.inputs["S"].set_value(np.asarray(trS, dtype=np.float32))
        return self._sample()

    def q(self, trS):
        if not hasattr(self, "_q"):
            self._setup_functions()
        self.inputs["S"].set_value(np.asarray(trS, dtype=np.float32))
        return self._q()

    def fit(self, trS, trA, trR, trS_, epochs=100):
        if not hasattr(self, "_fit_function"):
            self._setup_functions()

        self.inputs["S"].set_value(np.asarray(trS, dtype=np.float32))
        self.inputs["A"].set_value(np.asarray(trA, dtype=np.float32))
        self.inputs["R"].set_value(np.asarray(trR, dtype=np.float32))
        self.inputs["S_"].set_value(np.asarray(trS_, dtype=np.float32))
        print('TRAINING DQN')
        t = time()
        n = 0.
        for e in range(epochs):
            iter_num = 0;
            tot_cost = 0;
            #for xmS, xmA, xmR, xmS_ in iter_data_tv(trS, trA, trR, trS_):
            iter_num += 1
            #    #xmb = np.asarray(xmb)
            #    #xmA = floatX(xmA)
            #    #xmS = floatX(xmS)
            #    #xmR = floatX(xmR)
            #    #xmS_ = floatX(xmS_)
            cost = self._fit_function()
            tot_cost += cost

            #    n += xmA.shape[0]

            self.costs_.append(tot_cost)

            print(format(iter_num), "batches complete. DQN")

            print("Train iter", e)
            print("Total iters run", self.epoch_)
            print("Cost", tot_cost)
            print("Mean cost", np.mean(self.costs_))
            print("Time", n / (time() - t))

            print("Debugtrace: ")

            self.epoch_ += 1

            # Synchronise the target network parameters
            # with the Q network parameters every target_sync epochs
            if e % self.target_sync == 0:
                print("Synchronising")
                for tparam, qparam in zip(self.target_params, self.q_params):
                    tparam.set_value( qparam.get_value() )

            if e % 5 == 0:
                print("Saving model snapshot")
                f = open(self.snapshot_file, 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()

            def tf(x):
                return ((x + 1.) / 2.).transpose(1, 2, 0)

            if e == epochs or e % 100 == 0:
                if self.image_save_root is None:
                    image_save_root = os.path.split(__file__)[0]
                else:
                    image_save_root = self.image_save_root

                samples_path = os.path.join(
                    image_save_root, "sample_images_epoch_%d" % self.epoch_)

                if not os.path.exists(samples_path):
                    os.makedirs(samples_path)

def cropLast( trA ):
    return trA[:,:-1]

class ModelPredictivePDQN(PickleMixin):

    def __init__(self, envbuilder, num_episodes=100, num_actions=7, pnn_epochs=100, dqn_epochs=100, reward_multiplier=0.02, buffer_samples=500):
        self.acpnn = ACPNN()
        self.dqn = MPEDQN()
        self.pnn_epochs = pnn_epochs
        self.dqn_epochs = dqn_epochs
        self.buffer_samples = buffer_samples
        self.envs = []
        self.num_episodes = num_episodes
        self.num_actions = num_actions
        self.reward_multiplier = reward_multiplier
        self.builder = envbuilder

    def run(self, episode_length=10):
        for k in range(0,10):
            self.envs = [];
            for i in range(0, self.num_episodes):
                env = Recorder(self.builder.build())
                env.reset()

                # INITIALISE
                env.action(6)

                self.envs.append(env)
            self.iter(episode_length=episode_length)

    def iter(self, episode_length=10):

        # TxN_xA
        trA = np.zeros((1, self.num_episodes, self.num_actions));

        # TxN_x(A-1)
        dbgQ = np.zeros((1, self.num_episodes, self.num_actions - 1));
        # One Hot Action: Initialise.
        trA[:,:,6] = 1
        for epno in range(0, episode_length):
            # TxN_xS
            trS = self.acpnn.next_state( trA )
            # One hot action samples.
            # N_x(A-1)
            nA = self.dqn.sample( trS[-1,:,:] )
            # N_x(A-1)
            nQ = self.dqn.q( trS[-1,:,:] )

            # N_xA
            nA = np.concatenate([nA, np.zeros((nA.shape[0],1))], axis=1)

            for env, ac in zip(self.envs, nA):
                env.action( ac.argmax() )

            trA = np.concatenate( [trA,nA.reshape( (1,) + nA.shape)], axis=0 )
            dbgQ = np.concatenate( [dbgQ, nQ.reshape((1,) + nQ.shape)], axis=0 )

        outputss = []
        for env in self.envs:
            env.flush()
            # WARN: Check get_trace() index.
            outputss.append(env.get_trace()[-1])


        # N_xTxWxC
        obs = np.asarray([[output[0][0] for output in outputs] for outputs in outputss])
        # N_xTxA
        acs = np.asarray([[to_one_hot(output[1], 7) for output in outputs] for outputs in outputss])
        # N_xTx2
        poss = np.asarray([[output[2] for output in outputs] for outputs in outputss])
        # N_xTx2
        dirs = np.asarray([[output[3] for output in outputs] for outputs in outputss])
        # N_xTxW
        uvs = np.asarray([[np.linspace(-1, 1, obs.shape[2]) for output in outputs] for outputs in outputss])

        states = np.concatenate([poss, dirs], axis=2)

        N_ = obs.shape[0]
        W = obs.shape[2]

        # 80% for train and 20% for test.
        N_split = N_
        train_uvs, train_obs, train_acs, train_states = transform(uvs, obs, acs,
                                                                  states)
        #test_uvs, test_obs, test_acs, test_states = transform(uvs[N_split:], obs[N_split:], acs[N_split:],
        #                                                      states[N_split:])

        # Action Values.



        # Train the ACPNN on these traces.
        self.acpnn.fit( train_acs, train_uvs, train_obs, epochs=self.pnn_epochs, qvals=dbgQ.transpose([1,0,2]) )

        # TxNxC
        preds = self.acpnn.predict(train_acs, train_uvs)
        # TxNxC
        actuals = train_obs

        # TxN_xWxC
        preds = preds.reshape([preds.shape[0], N_, W, preds.shape[2]])
        actuals = actuals.reshape([actuals.shape[0], N_, W, actuals.shape[2]])

        # TxN_xR
        trR = np.tanh(self.reward_multiplier * np.sum(np.sum(np.square(preds - actuals), axis=3), axis=2))
        trR = trR.reshape((trR.shape + (1,)))

        rewards = trR

        # We have TxN_xS state vector now
        trS = self.acpnn.next_state( trA )
        attns = self.acpnn.predict_attention( train_acs, train_uvs )

        attns = attns.reshape([attns.shape[0], N_, W, attns.shape[2]])

        # Leave the first one to make S prime.
        # T-1xN_xS
        trS_ = trS[1:,:,:]
        # Leave the last one
        # T-1xN_xS
        trS = trS[:-1,:,:]

        # Get a random shuffle index.
        shuf = np.arange(trS.shape[0] * trS.shape[1])
        np.random.shuffle(shuf)

        # Flatten and Shuffle all tensors.
        # TODO: Check that the index for trA and trR are correct.
        # (N*)xA
        trA = trA[1:,:,:].reshape(((trA.shape[0]-1)*trA.shape[1], trA.shape[2]))[shuf,:]
        # (N*)x(A-1)
        trA = cropLast( trA )
        # (N*)x(S-1)
        trS = trS.reshape((trS.shape[0] * trS.shape[1], trS.shape[2]))[shuf,:]
        # (N*)x(S-1)
        trS_ = trS_.reshape((trS_.shape[0] * trS_.shape[1], trS_.shape[2]))[shuf,:]
        # (N*)x(S-1)
        trR = trR[1:,:,:].reshape(((trR.shape[0]-1) * trR.shape[1], trR.shape[2]))[shuf,:]

        # Train DQN
        self.dqn.fit( trS[:self.buffer_samples], trA[:self.buffer_samples], trR[:self.buffer_samples,0], trS_[:self.buffer_samples], epochs=self.dqn_epochs)

        plotter = PyGame2D()
        plotter.draw_path_cmp(self.envs[0].env, outputss[0], preds.transpose([1,0,2,3])[0], attns.transpose([1,0,2,3])[0], rewards.transpose([1,0,2])[0] )

# Action-Conditional Projection Neural Network with Model Predictive Bonus
class ACPNN(PickleMixin):
    def __init__(self, image_save_root=None, snapshot_file="snapshot.pkl", n_state=4, n_actions=7, n_tex=1, n_ray=3, n_scene=4, n_key=3, n_interaction=20):

        self.srng = RandomStreams()
        self.n_state = n_state
        self.n_actions = n_actions
        self.n_tex = n_tex
        self.n_ray = n_ray
        self.n_scene = n_scene
        self.n_interaction = n_interaction
        self.n_key = n_key
        self.n_batch = 100

        self.costs_ = []
        self.epoch_ = 0
        self.snapshot_file = snapshot_file
        self.image_save_root = image_save_root
        if os.path.exists(self.snapshot_file):
            print("Loading from saved snapshot " + self.snapshot_file)
            f = open(self.snapshot_file, 'rb')
            classifier = cPickle.load(f)
            self.__setstate__(classifier.__dict__)
            f.close()

    def _setup_functions(self):

        # Actual parameter lengths.
        #sh_w_n = (self.n_state + self.n_actions + 1, self.n_state + 1, self.n_state)
        #print("sh_w_n", sh_w_n)
        sh_w_n = (self.n_actions + 1, self.n_state + 1, self.n_state)
        print("sh_w_n", sh_w_n)
        sh_w_t = (self.n_tex + 1, self.n_state + 1, self.n_ray)
        print("sh_w_t", sh_w_t)
        sh_l1 = (self.n_ray + self.n_key, self.n_interaction)
        print("sh_l1", sh_l1)
        sh_l2 = (self.n_interaction, 1)
        print("sh_l2", sh_l2)

        # Memory cells.
        sh_mk = (self.n_scene, self.n_key)
        sh_mc = (self.n_scene, 4)
        print("sh_mk", sh_mk)
        print("sh_mc", sh_mc)


        if not hasattr(self, "params"):
            print('generating weights')

            # (A+1)x(S+1)xS
            wn = uniform(sh_w_n, scale=0.2)
            # (P+1)x(S+1)xR
            wt = uniform(sh_w_t, scale=0.2)
            # (R+K)xH
            wl1 = uniform(sh_l1, scale=0.2)
            # H
            wb1 = shared0s((self.n_interaction,))
            # Hx1
            wl2 = uniform(sh_l2, scale=0.2)
            # MxK
            wmk = uniform(sh_mk, scale=0.2)
            # MxC
            wmc = uniform(sh_mc, scale=0.2)

            self.params = [wn, wt, wl1, wb1, wl2, wmk, wmc]
        else:
            wn,wt,wl1,wb1,wl2,wmk,wmc = self.params

        #TxNxA
        A = sharedX( np.zeros((2,2,2)),name="A")
        #TxNxP
        P = sharedX( np.zeros((2,2,2)),name="P")
        #TxNxC
        y = sharedX( np.zeros((2,2,2)),name="y")

        self.inputs = {"A":A,"P":P,"y":y}
        # Inputs: NxS, NxA
        def state_transform( a_, s_ ):
            # Nx(S+1)xS
            temp_ = T.tensordot(T.concatenate([a_, T.ones( (s_.shape[0], 1) )], axis=1), wn, axes=[1,0])
            # NxS
            return T.sum(temp_ * T.concatenate([s_, T.ones( (s_.shape[0], 1) )], axis=1 ).dimshuffle([0,1,'x']), axis=1)
            #return s_

        # TxNxS
        S, _ = theano.scan( fn=state_transform,
                         outputs_info=[T.zeros([A.shape[1],self.n_state])],
                         sequences=[A]
                        )

        # TxNx(S+1)xR
        temp_ = T.tensordot(T.concatenate( [P, T.ones( [S.shape[0], S.shape[1], 1] )], axis=2 ), wt, axes=[2,0])

        # TxNxR Ray Elements.
        R = T.sum(temp_ * T.concatenate([S, T.ones( (S.shape[0], S.shape[1], 1) )], axis=2).dimshuffle([0, 1, 2, 'x']), axis=2)

        # TxNxMx(R+K) Transformation input.
        R_2 = T.concatenate([ T.tile(R.dimshuffle([0,1,'x',2]),[1,1,self.n_scene,1]), T.tile( wmk.dimshuffle(['x','x',0,1]), [R.shape[0], R.shape[1], 1, 1]) ], axis=3)

        # TxNxMxH
        L1 = sigmoid( T.tensordot(R_2, wl1, axes=[3,0]) + wb1.dimshuffle(['x','x','x',0]) )
        # TxNxM Soft attention weights.
        Att_temp = T.exp( T.tensordot(L1, wl2, axes=[3,0]).sum( axis=3 ) )
        Att = Att_temp / (T.sum( Att_temp, axis=2, keepdims=True) + 0.01)
        #Att = sigmoid( T.tensordot(L1, wl2, axes=[3,0]).sum( axis=3 ) )

        # TxNxC final colors.
        Col = T.tensordot( Att, wmc, axes=[2,0] )

        rec_cost = T.sum(T.sqr(Col - y)) # / T.cast(X.shape[0], 'float32')
        cost = rec_cost

        print('getting updates')
        #updates = Adam([wt,wn,wmk,wl1,wb1,wl2,wmc], cost)
        updates = Adam(self.params, cost)

        print('compiling')
        self._fit_function = theano.function([], cost, updates=updates)

        theano.printing.debugprint(self._fit_function)

        #self._predict = theano.function([A, P], Col, allow_input_downcast=True)
        self._predict = theano.function([], Col, allow_input_downcast=True)
        #self._next_state = theano.function([A], S, allow_input_downcast=True)
        self._next_state = theano.function([], S, allow_input_downcast=True)
        #self._predict_attn = theano.function([A, P], Att, allow_input_downcast=True)
        self._predict_attn = theano.function([], Att, allow_input_downcast=True)
        # Output just the cost to check with a test set.
        #self._cost = theano.function([A,P,y], cost, allow_input_downcast=True)
        self._cost = theano.function([], cost, allow_input_downcast=True)

    def next_state(self, trA):
        if not hasattr(self, "_next_state"):
            self._setup_functions()
        self.inputs["A"].set_value(np.asarray(trA, dtype=np.float32))
        return self._next_state()

    def predict(self, trA, trP):
        self.inputs["A"].set_value(np.asarray(trA, dtype=np.float32))
        self.inputs["P"].set_value(np.asarray(trP, dtype=np.float32))
        return self._predict()

    def verify(self, trA, trP, trY):
        self.inputs["A"].set_value(np.asarray(trA, dtype=np.float32))
        self.inputs["P"].set_value(np.asarray(trP, dtype=np.float32))
        self.inputs["y"].set_value(np.asarray(trY, dtype=np.float32))
        return self._cost()

    def predict_attention(self, trA, trP):
        self.inputs["A"].set_value(np.asarray(trA, dtype=np.float32))
        self.inputs["P"].set_value(np.asarray(trP, dtype=np.float32))
        return self._predict_attn()

    # Inputs: TxNxA, TxNxP, TxNxC
    def fit(self, trA, trP, trY, epochs=100, qvals=None):
        if not hasattr(self, "_fit_function"):
            self._setup_functions()

        #xs = floatX(np.random.randn(100, self.n_code))
        print('TRAINING')
        #x_rec = floatX(shuffle(trX)[:100])
        t = time()
        n = 0.
        #epochs = 10
        self.inputs["A"].set_value(np.asarray(trA, dtype=np.float32))
        self.inputs["P"].set_value(np.asarray(trP, dtype=np.float32))
        self.inputs["y"].set_value(np.asarray(trY, dtype=np.float32))

        for e in range(epochs):
            iter_num = 0
            tot_cost = 0
            #for xmA, xmP, xmY in iter_data_tv(trA.transpose([1,0,2]), trP.transpose([1,0,2]), trY.transpose([1,0,2]), size=self.n_batch):
            iter_num += 1
                #xmb = np.asarray(xmb)
            cost = self._fit_function()
            tot_cost += cost

            n += trA.shape[0]

            self.costs_.append(tot_cost)

            print(format(iter_num), "batches complete.")

            print("Train iter", e)
            print("Total iters run", self.epoch_)
            print("Cost", tot_cost)
            print("Mean cost", np.mean(self.costs_))
            print("Time", n / (time() - t))

            print("Debugtrace: ")
            print(self.params[5].eval())
            print(self.params[6].eval())
            #print("wn")
            #print(self.params[0].eval())
            #print("wb1")
            #print(self.params[3].eval())
            #print("wl1")
            #print(self.params[2].eval())
            #print("wl2")
            #print(self.params[4].eval())
            #print("wt")
            #print(self.params[1].eval())

            print("S_final")
            #self.inputs["A"].set_value(np.asarray(trA, dtype=np.float32))
            S = self._next_state( )
            print(trA[:,-1,:])
            print(qvals[-1,:,:])
            print(S[:,-1,:])

            self.epoch_ += 1

            if e % 5 == 0:
                print("Saving model snapshot")
                f = open(self.snapshot_file, 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()

            def tf(x):
                return ((x + 1.) / 2.).transpose(1, 2, 0)

            if e == epochs or e % 100 == 0:
                if self.image_save_root is None:
                    image_save_root = os.path.split(__file__)[0]
                else:
                    image_save_root = self.image_save_root

                samples_path = os.path.join(
                    image_save_root, "sample_images_epoch_%d" % self.epoch_)

                if not os.path.exists(samples_path):
                    os.makedirs(samples_path)

def to_one_hot(num, tot):
    return [0] * (num) + [1] + [0] * (tot - (num + 1))

def transform( uvs, obs, acs, states ):
    # Extend N_xTxW to N_xTxWxP
    uvs = uvs.reshape(uvs.shape + (1,))
    # N_xTxWxP to TxNxP
    uvs = uvs.transpose([1, 0, 2, 3]).reshape([uvs.shape[1], uvs.shape[0] * uvs.shape[2], uvs.shape[3]])
    W = obs.shape[2];

    # Convert the rest
    # TxNxC
    obs = obs.transpose([1, 0, 2, 3]).reshape([obs.shape[1], obs.shape[0] * obs.shape[2], obs.shape[3]])

    # TxNxA
    acs = np.tile(acs.transpose([1, 0, 2]).reshape([acs.shape[1], acs.shape[0], 1, acs.shape[2]]), [1, 1, W, 1]) \
        .reshape([acs.shape[1], acs.shape[0] * W, acs.shape[2]])

    # TxNxS
    states = np.tile(states.transpose([1, 0, 2]).reshape([states.shape[1], states.shape[0], 1, states.shape[2]]), [1, 1, W, 1]) \
        .reshape([states.shape[1], states.shape[0] * W, states.shape[2]])
    return uvs, obs, acs, states

if __name__ == "__main__":
    # lfw is (9164, 3, 64, 64)
    pass
