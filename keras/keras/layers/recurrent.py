# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
import os
import scipy.io
import math

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros, alloc_zeros_matrix
from ..layers.core import Layer
from .. import regularizers

from six.moves import range

#context_path = os.path.join('./data/', 'example.mat')     #3000*4096
#context_temp = scipy.io.loadmat(context_path)['feats'].transpose()
#context = context_temp[0:100]
#print context.shape[0],context.shape[1]

class BLSTM(Layer):
    def __init__(self, input_dim, output_dim,init='glorot_uniform', inner_init='orthogonal',
        activation='tanh', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False,
        is_entity=False, regularize=False):

        self.is_entity = is_entity
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_if = self.init((self.input_dim, self.output_dim))    # input
        self.W_ib = self.init((self.input_dim, self.output_dim))
        self.U_if = self.inner_init((self.output_dim, self.output_dim))
        self.U_ib = self.inner_init((self.output_dim, self.output_dim))
        #self.Z_if = self.inner_init((self.output_dim, self.output_dim))       #z
        #self.Z_ib = self.inner_init((self.output_dim, self.output_dim))
        self.b_if = shared_zeros((self.output_dim))
        self.b_ib = shared_zeros((self.output_dim))

        self.W_ff = self.init((self.input_dim, self.output_dim))    #forget
        self.W_fb = self.init((self.input_dim, self.output_dim))
        self.U_ff = self.inner_init((self.output_dim, self.output_dim))
        self.U_fb = self.inner_init((self.output_dim, self.output_dim))
        #self.Z_ff = self.inner_init((self.output_dim, self.output_dim))
        #self.Z_fb = self.inner_init((self.output_dim, self.output_dim))
        self.b_ff = shared_zeros((self.output_dim))
        self.b_fb = shared_zeros((self.output_dim))

        self.W_cf = self.init((self.input_dim, self.output_dim))    #memory
        self.W_cb = self.init((self.input_dim, self.output_dim))
        self.U_cf = self.inner_init((self.output_dim, self.output_dim))
        self.U_cb = self.inner_init((self.output_dim, self.output_dim))
        #self.Z_cf = self.inner_init((self.output_dim, self.output_dim))
        #self.Z_cb = self.inner_init((self.output_dim, self.output_dim))
        self.b_cf = shared_zeros((self.output_dim))
        self.b_cb = shared_zeros((self.output_dim))

        self.W_of = self.init((self.input_dim, self.output_dim))    #output
        self.W_ob = self.init((self.input_dim, self.output_dim))
        self.U_of = self.inner_init((self.output_dim, self.output_dim))
        self.U_ob = self.inner_init((self.output_dim, self.output_dim))
        #self.Z_of = self.inner_init((self.output_dim, self.output_dim))
        #self.Z_ob = self.inner_init((self.output_dim, self.output_dim))
        self.b_of = shared_zeros((self.output_dim))
        self.b_ob = shared_zeros((self.output_dim))

        self.W_yf = self.init((self.output_dim, self.output_dim))
        self.W_yb = self.init((self.output_dim, self.output_dim))
        #self.W_y = self.init((self.output_dim, self.output_dim))
        self.b_y = shared_zeros((self.output_dim))

        self.params = [
            self.W_if, self.U_if, self.b_if,
            self.W_ib, self.U_ib, self.b_ib,

            self.W_cf, self.U_cf, self.b_cf,
            self.W_cb, self.U_cb, self.b_cb,

            self.W_ff, self.U_ff, self.b_ff,
            self.W_fb, self.U_fb, self.b_fb,

            self.W_of, self.U_of, self.b_of,
            self.W_ob, self.U_ob, self.b_ob,

            self.W_yf, self.W_yb, self.b_y
            #self.W_y, self.b_y
        ]


        if regularize:
            self.regularizers = []
            for i in self.params:
                self.regularizers.append(regularizers.my_l2)

        if weights is not None:
            self.set_weights(weights)

    # some utilities
    def ortho_weight(ndim):
        """
        Random orthogonal weights

        Used by norm_weights(below), in which case, we
        are ensuring that the rows are orthogonal
        (i.e W = U \Sigma V, U has the same
        # of rows, V has the same # of cols)
        """
        W = np.random.randn(ndim, ndim)
        u, _, _ = np.linalg.svd(W)
        return u.astype('float32')

    def norm_weight(nin,nout=None, scale=0.01, ortho=True):
        """
        Random weights drawn from a Gaussian
        """
        #print type(nin), nin, type(nout), nout
        if nout is None:
            nout = nin
        if nout == nin and ortho:
            W = self.ortho_weight(nin)
        else:

            W = scale * np.random.randn(nin, nout)
        return W.astype('float32')

    def tanh(x):
        return T.tanh(x)

    def _step(self,
        xi_t, xf_t, xo_t, xc_t,
        h_tm1, c_tm1, 
        u_i, u_f, u_o, u_c):

        #assert context, 'Context must be provided'
        #print context.shape

        # attention: context -> hidden
        #Wc_att = self.norm_weight(4096, ortho=False)   #dimctx
        #index = h_tm1.shape[1]
        #context = context_temp[0:index]

        #Wc_att = 0.01 * np.random.randn(4096,4096)

        # attention: LSTM -> hidden
        #Wd_att = self.norm_weight(300,4096)  #dim,dimctx
        #Wd_att = 0.01 * np.random.randn(100,4096)
        #Wd_att = 0.01 * np.random.randn(300,4096)

        # attention:
        #U_att = self.norm_weight(dimctx,1)
        #U_att = 0.01 * np.random.randn(4096,1)

        #W_att = 0.01 * np.random.randn(100,4096)
        #W_att = 0.01 * np.random.randn(300,4096)

        #ctx = T.dot(context, Wc_att)
        #print "ctx.shape",ctx.shape
        #state = T.dot(h_tm1, Wd_att)
        #print "state.shape",state.shape
        #ctx = ctx + state[:,None,:]
        '''
        ctx = ctx + state
        ctx_list = []
        ctx_list.append(ctx)
        ctx = T.tanh(ctx)
        alpha = T.dot(ctx,U_att)

        alpha_pre = alpha
        alpha_shp = alpha.shape
        alpha = T.nnet.softmax(alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
        alpha = T.dot(alpha.transpose(),W_att)
        z_t = ( context * alpha).sum(1)
        '''

        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))      #tm1 => t - 1
        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)

        alpha = T.nnet.softmax(h_t)
        #c_t = (c_t * alpha).sum(1)
        c_t = c_t * alpha

        return h_t, c_t

    def output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))


        if self.is_entity:
            Entity = X[-1:].dimshuffle(1,0,2)
            X = X[:-1]

        b_y = self.b_y
        b_yn = T.repeat(T.repeat(b_y.reshape((1,self.output_dim)),X.shape[0],axis=0).reshape((1,X.shape[0],self.output_dim)), X.shape[1], axis=0)

        xif = T.dot(X, self.W_if) + self.b_if
        xib = T.dot(X, self.W_ib) + self.b_ib

        xff = T.dot(X, self.W_ff) + self.b_ff
        xfb = T.dot(X, self.W_fb) + self.b_fb

        xcf = T.dot(X, self.W_cf) + self.b_cf
        xcb = T.dot(X, self.W_cb) + self.b_cb

        xof = T.dot(X, self.W_of) + self.b_of
        xob = T.dot(X, self.W_ob) + self.b_ob

        [outputs_f, memories_f], updates_f = theano.scan(        #forword  
            self._step,
            sequences=[xif, xff, xof, xcf],
            outputs_info=[
                alloc_zeros_matrix(X.shape[1], self.output_dim),
                alloc_zeros_matrix(X.shape[1], self.output_dim)
            ],
            non_sequences=[self.U_if, self.U_ff, self.U_of, self.U_cf],
            truncate_gradient=self.truncate_gradient
        )
        [outputs_b, memories_b], updates_b = theano.scan(        #backword
            self._step,
            sequences=[xib, xfb, xob, xcb],
            outputs_info=[
                alloc_zeros_matrix(X.shape[1], self.output_dim),
                alloc_zeros_matrix(X.shape[1], self.output_dim)
            ],
            non_sequences=[self.U_ib, self.U_fb, self.U_ob, self.U_cb],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            y = T.add(T.add(
                    T.tensordot(outputs_f.dimshuffle((1,0,2)), self.W_yf, [[2],[0]]),
                    T.tensordot(outputs_b[::-1].dimshuffle((1,0,2)), self.W_yb, [[2],[0]])),
                b_yn)
            # y = T.add(T.tensordot(
            #     T.add(outputs_f.dimshuffle((1, 0, 2)),
            #           outputs_b[::-1].dimshuffle((1,0,2))),
            #     self.W_y,[[2],[0]]),b_yn)
            if self.is_entity:
                return T.concatenate([y, Entity], axis=1)
            else:
                return y
        return T.concatenate((outputs_f[-1], outputs_b[0]))

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}


class BRNN(Layer):
    '''
        Fully connected Bi-directional RNN where:
            Output at time=t is fed back to input for time=t+1 in a forward pass
            Output at time=t is fed back to input for time=t-1 in a backward pass
    '''
    def __init__(self, input_dim, output_dim,
        init='uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
        truncate_gradient=-1,  return_sequences=False, is_entity=False, regularize=False):
        #whyjay
        self.is_entity = is_entity

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.input = T.tensor3()
        self.W_o  =  self.init((self.input_dim, self.output_dim))
        self.W_if = self.init((self.input_dim, self.output_dim))    # Input -> Forward
        self.W_ib = self.init((self.input_dim, self.output_dim))    # Input -> Backward
        self.W_ff = self.init((self.output_dim, self.output_dim))   # Forward tm1 -> Forward t
        self.W_bb = self.init((self.output_dim, self.output_dim))   # Backward t -> Backward tm1
        self.b_if = shared_zeros((self.output_dim))
        self.b_ib = shared_zeros((self.output_dim))
        self.b_f = shared_zeros((self.output_dim))
        self.b_b = shared_zeros((self.output_dim))
        self.b_o =  shared_zeros((self.output_dim))
        self.params = [self.W_o,self.W_if,self.W_ib, self.W_ff, self.W_bb,self.b_if,self.b_ib, self.b_f, self.b_b, self.b_o]

        if regularize:
            self.regularizers = []
            for i in self.params:
                self.regularizers.append(regularizers.my_l2)

        if weights is not None:
            self.set_weights(weights)

    def _step(self, x_t, h_tm1, u,b):
        return self.activation(x_t + T.dot(h_tm1, u)+b)

    def output(self, train):
        X = self.get_input(train) # shape: (nb_samples, time (padded with zeros at the end), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        X = X.dimshuffle((1, 0, 2))

        if self.is_entity:
            lenX=X.shape[0]
            Entity=X[lenX-1:].dimshuffle(1,0,2)
            X=X[:lenX-1]

        xf = self.activation(T.dot(X, self.W_if) + self.b_if)
        xb = self.activation(T.dot(X, self.W_ib) + self.b_ib)
        b_o=self.b_o
        b_on= T.repeat(T.repeat(b_o.reshape((1,self.output_dim)),X.shape[0],axis=0).reshape((1,X.shape[0],self.output_dim)),X.shape[1],axis=0)

        # Iterate forward over the first dimension of the x array (=time).
        outputs_f, updates_f = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=xf,  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=alloc_zeros_matrix(X.shape[1], self.output_dim),
            non_sequences=[self.W_ff,self.b_f],  # static inputs to _step
            truncate_gradient=self.truncate_gradient
        )
        # Iterate backward over the first dimension of the x array (=time).
        outputs_b, updates_b = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=xb,  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=alloc_zeros_matrix(X.shape[1], self.output_dim),
            non_sequences=[self.W_bb,self.b_b],  # static inputs to _step
            truncate_gradient=self.truncate_gradient,
            go_backwards=True  # Iterate backwards through time
        )
        #return outputs_f.dimshuffle((1, 0, 2))
        if self.return_sequences:
            if self.is_entity:
                return T.concatenate([T.add(T.tensordot(T.add(outputs_f.dimshuffle((1, 0, 2)), outputs_b[::-1].dimshuffle((1,0,2))),self.W_o,[[2],[0]]),b_on),Entity],axis=1)
            else:
                return T.add(T.tensordot(T.add(outputs_f.dimshuffle((1, 0, 2)), outputs_b[::-1].dimshuffle((1,0,2))),self.W_o,[[2],[0]]),b_on)

        return T.concatenate((outputs_f[-1], outputs_b[0]))

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}

