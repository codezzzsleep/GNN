import torch

# def _loss(self):
#     # Weight decay loss
#     for var in self.layers0.vars.values():
#         self.loss1 += FLAGS.weight_decay * tf.nn.l2_loss(var)
#     for var in self.layers1.vars.values():
#         self.loss2 += FLAGS.weight_decay * tf.nn.l2_loss(var)
#
#     # Graph Learning loss
#     D = tf.matrix_diag(tf.ones(self.placeholders['num_nodes'])) * -1
#     D = tf.sparse_add(D, self.S) * -1
#     D = tf.matmul(tf.transpose(self.x), D)
#     self.loss1 += tf.trace(tf.matmul(D, self.x)) * FLAGS.losslr1
#     self.loss1 -= tf.trace(
#         tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.S), tf.sparse_tensor_to_dense(self.S))) * FLAGS.losslr2
#
#     # Cross entropy error
#     self.loss2 += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
#                                                self.placeholders['labels_mask'])
#
#     self.loss = self.loss1 + self.loss2
def gl_loss(x, adj, losslr1, losslr2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    adj = adj.to(device)

    D = torch.diag(torch.ones(x.shape[0])) * -1  # 计算度矩阵
    D = D.to(device)

    D = (D + adj) * -1
    D = torch.mm(x.t(), D)

    loss1 = torch.trace(torch.mm(D, x)) * losslr1
    loss2 = torch.trace(torch.mm(adj.t(), adj)) * losslr2

    return loss1 - loss2
