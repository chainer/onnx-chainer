import collections
import os

import mxnet
import numpy as np


def check_compatibility(model, x, fn):
    chainer.config.train = False
    chainer_out = model(x).array

    onnx_chainer.export(model, x, fn)

    sym, arg, aux = mxnet.contrib.onnx.import_model(fn)

    data_names = [graph_input for graph_input in sym.list_inputs()
                  if graph_input not in arg and graph_input not in aux]
    mod = mxnet.mod.Module(
        symbol=sym, data_names=data_names, context=mxnet.cpu(),
        label_names=None)
    mod.bind(
        for_training=False, data_shapes=[(data_names[0], x.shape)],
        label_shapes=None)
    mod.set_params(
        arg_params=arg, aux_params=aux, allow_missing=True,
        allow_extra=True)

    Batch = collections.namedtuple('Batch', ['data'])
    mod.forward(Batch([mxnet.nd.array(x)]))
    mxnet_outs = mod.get_outputs()
    mxnet_out = mxnet_outs[0].asnumpy()

    np.testing.assert_almost_equal(
        chainer_out, mxnet_out, decimal=5)

    os.remove(fn)
