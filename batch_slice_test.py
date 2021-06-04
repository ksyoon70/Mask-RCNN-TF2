
import numpy as np
import tensorflow as tf

def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        #inputs_slice = [x[i] for x in inputs]  #inputs가 1차원이면 이렇게 수정 필요
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result

#scores = tf.constant([1, 2, 98, 1, 1, 99, 3, 1, 3, 96, 4, 1], dtype = tf.float64)
scores = tf.constant([[1, 2,3,4],[5,6,7,8]])
ix = tf.math.top_k(scores,k=3,sorted=True,name="top_anchors").indices
result = batch_slice([scores,ix],lambda x, y: tf.gather(x, y),2)

print(result[0].eval)
print(result[1].eval)
