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
from theano.tensor.nnet import conv2d, softmax
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
from skimage.transform import resize


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

# Action-Conditional Projection Neural Network
class StatefulACPNN(PickleMixin):
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
            self.params, self.epoch_, self.costs_ = cPickle.load(f)
            #self.__setstate__(classifier.__dict__)
            #self.params = params;
            f.close()

    def _setup_functions(self):

        # Actual parameter lengths.
        #sh_w_n = (self.n_state + self.n_actions + 1, self.n_state + 1, self.n_state)
        #print("sh_w_n", sh_w_n)
        #sh_w_n = (self.n_actions + 1, self.n_state + 1, self.n_state)
        #print("sh_w_n", sh_w_n)
        sh_w_t = (self.n_tex + 1, self.n_state + 1, self.n_ray)
        print("sh_w_t", sh_w_t)
        sh_l1 = (self.n_ray + self.n_key, self.n_interaction)
        print("sh_l1", sh_l1)
        # 1 for I-ACT and 1 for C-ACT
        sh_l2 = (self.n_interaction, 2)
        print("sh_l2", sh_l2)

        # Memory cells.
        sh_mk = (self.n_scene, self.n_key)
        sh_mc = (self.n_scene, 4)
        print("sh_mk", sh_mk)
        print("sh_mc", sh_mc)


        if not hasattr(self, "params"):
            print('generating weights')

            # (A+1)x(S+1)xS
            #wn = uniform(sh_w_n, scale=0.2)
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

            self.params = [wt, wl1, wb1, wl2, wmk, wmc]
        else:
            wt,wl1,wb1,wl2,wmk,wmc = self.params

        #TxNxS
        S = T.tensor3()
        #TxNxP
        P = T.tensor3()
        #TxNxC
        y = T.tensor3()

        # TxNx(S+1)xR
        temp_ = T.tensordot(T.concatenate( [P, T.ones( [S.shape[0], S.shape[1], 1] )], axis=2 ), wt, axes=[2,0])

        # TxNxR Ray Elements.
        R = T.sum(temp_ * T.concatenate([S, T.ones( (S.shape[0], S.shape[1], 1) )], axis=2).dimshuffle([0, 1, 2, 'x']), axis=2)

        # TxNxMx(R+K) Transformation input.
        R_2 = T.concatenate([ T.tile(R.dimshuffle([0,1,'x',2]),[1,1,self.n_scene,1]), T.tile( wmk.dimshuffle(['x','x',0,1]), [R.shape[0], R.shape[1], 1, 1]) ], axis=3)

        # TxNxMxH
        L1 = sigmoid( T.tensordot(R_2, wl1, axes=[3,0]) + wb1.dimshuffle(['x','x','x',0]) )
        # TxNxM Soft Competitive attention weights.
        Att_temp = T.exp( T.tensordot(L1, wl2, axes=[3,0])[:,:,:,0] )
        cact = Att_temp / (T.sum( Att_temp, axis=2, keepdims=True) + 0.01)

        # TxNxM Soft Individual Activation weights.
        iact = sigmoid( T.tensordot(L1, wl2, axes=[3,0])[:,:,:,1] )

        # Final Attention is the product of Individual and Competitive attention.
        Att = iact * cact;

        # TxNxC final colors.
        Col = T.tensordot( Att, wmc, axes=[2,0] )

        rec_cost = T.sum(T.sqr(Col - y)) # / T.cast(X.shape[0], 'float32')
        cost = rec_cost

        print('getting updates')
        #updates = Adam([wt,wn,wmk,wl1,wb1,wl2,wmc], cost)
        updates = Adam(self.params, cost)

        print('compiling')
        self._fit_model = theano.function([S, P, y], cost, updates=updates)
        self._predict_obs = theano.function([S, P], Col)
        #self._next_state = theano.function([A], S)

        # Output just the cost to check with a test set.
        self._cost = theano.function([S,P,y], cost)

    def predict(self, trS, trP):
        trS = floatX(trS)
        trP = floatX(trP)
        return self._predict_obs(trS, trP)

    def verify(self, trS, trP, trY):
        trS = floatX(trS)
        trP = floatX(trP)
        trY = floatX(trY)
        return self._cost(trS, trP, trY)

    # Inputs: TxNxS, TxNxP, TxNxC
    def fit(self, trS, trP, trY, epochs=100):
        if not hasattr(self, "_fit_function"):
            self._setup_functions()

        #xs = floatX(np.random.randn(100, self.n_code))
        print('TRAINING')
        #x_rec = floatX(shuffle(trX)[:100])
        t = time()
        n = 0.
        #epochs = 10
        for e in range(epochs):
            iter_num = 0;
            tot_cost = 0;
            for xmS, xmP, xmY in iter_data(trS.transpose([1,0,2]), trP.transpose([1,0,2]), trY.transpose([1,0,2]), size=self.n_batch):
                iter_num += 1
                #xmb = np.asarray(xmb)

                xmS = floatX(xmS.transpose([1,0,2]))
                xmP = floatX(xmP.transpose([1,0,2]))
                xmY = floatX(xmY.transpose([1,0,2]))
                cost = self._fit_model( xmS, xmP, xmY )
                tot_cost += cost

                n += xmS.shape[0]

            self.costs_.append(tot_cost)

            print(format(iter_num), "batches complete.")

            print("Train iter", e)
            print("Total iters run", self.epoch_)
            print("Cost", tot_cost)
            print("Mean cost", np.mean(self.costs_))
            print("Time", n / (time() - t))

            print("Debugtrace: ")
#           print(self.params[5].eval())
#           print(self.params[6].eval())
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

            #print("S_final")
            #S = self._next_state( trA )
            #print(S[:,-1,:])

            self.epoch_ += 1

            if e % 5 == 0:
                print("Saving model snapshot")
                f = open(self.snapshot_file, 'wb')
                cPickle.dump((self.params, self.epoch_, self.costs_), f, protocol=2)
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

if __name__ == "__main__":
    # lfw is (9164, 3, 64, 64)
    pass
