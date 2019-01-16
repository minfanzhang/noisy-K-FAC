from core.base_train import BaseTrain
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pickle
import torch
import os


class Trainer(BaseTrain):
    def __init__(self, sess, model, train_loader, test_loader, config, logger):
        super(Trainer, self).__init__(sess, model, config, logger)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):
        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))
            self.train_epoch()
            #self.test_epoch()

    def train_epoch(self):
        loss_list = []
        acc_list = []
        for itr, (x, y) in enumerate(tqdm(self.train_loader)):
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.n_particles: self.config.train_particles
            }

            feed_dict.update({self.model.is_training: True})
            self.sess.run([self.model.train_op], feed_dict=feed_dict)

            feed_dict.update({self.model.is_training: False})  # note: that's important
            loss, acc = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)

            cur_iter = self.model.global_step_tensor.eval(self.sess)
            if cur_iter % self.config.TCov == 0:
                self.sess.run([self.model.cov_update_op], feed_dict=feed_dict)

            if cur_iter % self.config.TInv == 0:
                self.sess.run([self.model.inv_update_op, self.model.var_update_op], feed_dict=feed_dict)

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        self.logger.info("train | loss: %5.4f | accuracy: %5.4f"%(float(avg_loss), float(avg_acc)))

        # summarize
        summaries_dict = dict()
        summaries_dict['train_loss'] = avg_loss
        summaries_dict['train_acc'] = avg_acc

        # summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

        self.model.save(self.sess)

    def test_epoch(self):
        loss_list = []
        acc_list = []
        for (x, y) in self.test_loader:
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.is_training: False,
                self.model.n_particles: self.config.test_particles
            }
            loss, acc = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        self.logger.info("test | loss: %5.4f | accuracy: %5.4f\n"%(float(avg_loss), float(avg_acc)))

        # summarize
        summaries_dict = dict()
        summaries_dict['test_loss'] = avg_loss
        summaries_dict['test_acc'] = avg_acc

        # summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

    def grad_check(self, sess, batch_size, precon):
        self.model.load(sess)

        x, y = torch.load("cifar10_x")[:batch_size], \
             torch.load("cifar10_y")[:batch_size]

        

        num_samples = 150
        num_trials = 1

        opt = self.model.optim

        trainable_vars = tf.trainable_variables()
        if precon:
            gradient_step = opt.compute_precon_gradients(
                self.model.total_loss, opt.variables)
        else:
            gradient_step = tf.gradients(
                self.model.total_loss, opt.variables)

        feed_dict = {self.model.is_training: True,
                     self.model.inputs: x, self.model.targets: y,
                     self.model.n_particles: self.config.train_particles}

        W4_shape = [3, 3, 64, 64]
        W9_shape = [3, 3, 256, 256]
        W13_shape = [3, 3, 256, 256]
        W_FC_shape = [256, 10]

        W4_grad_var = np.zeros([num_trials])
        W9_grad_var = np.zeros([num_trials])
        W13_grad_var = np.zeros([num_trials])
        W_FC_grad_var = np.zeros([num_trials])

        for i in range(num_trials) :
            #print('Iter {}/{}'.format(i, num_trials))
            W4_grad_lst = np.zeros([num_samples,W4_shape[0],W4_shape[1],W4_shape[2],W4_shape[3]])
            W9_grad_lst = np.zeros([num_samples,W9_shape[0],W9_shape[1],W9_shape[2],W9_shape[3]])
            W13_grad_lst = np.zeros([num_samples,W13_shape[0],W13_shape[1],W13_shape[2],W13_shape[3]])
            W_FC_grad_lst = np.zeros([num_samples,W_FC_shape[0],W_FC_shape[1]])

            for j in range(num_samples) :
                grad_W = sess.run(gradient_step, feed_dict=feed_dict)
                W4_grad_lst[j,:,:,:,:] = grad_W[6][0]
                W9_grad_lst[j,:,:,:,:] = grad_W[16][0]
                W13_grad_lst[j,:,:,:,:] = grad_W[24][0]
                W_FC_grad_lst[j,:,:] = grad_W[26][0]

            W4_grad_var[i] = np.mean(np.var(W4_grad_lst, axis=0))
            W9_grad_var[i] = np.mean(np.var(W9_grad_lst, axis=0))
            W13_grad_var[i] = np.mean(np.var(W13_grad_lst, axis=0))
            W_FC_grad_var[i] = np.mean(np.var(W_FC_grad_lst, axis=0))

        print("Batch size: ",str(batch_size),\
              " With flip: ",str(self.config.use_flip),", \
              W4 gradients has variance: \n",W4_grad_var)
        print("Batch size: ",str(batch_size)," \
              With flip: ",str(self.config.use_flip),", \
              W9 gradients has variance: \n",W9_grad_var)
        print("Batch size: ",str(batch_size)," \
              With flip: ",str(self.config.use_flip),", \
              W13 gradients has variance: \n",W13_grad_var)
        print("Batch size: ",str(batch_size)," \
              With flip: ",str(self.config.use_flip),", \
              W_FC gradients has variance: \n",W_FC_grad_var)

        grad_save_path = 'experiments/grad_check/81train_acc_pre{}'.format(precon)
        if not os.path.exists(grad_save_path):
            os.makedirs(grad_save_path)
        with open('{}/{}.pkl'.format(grad_save_path, batch_size), 'wb') as f1:
            pickle.dump(
                [W4_grad_var, W9_grad_var, W13_grad_var, W_FC_grad_var], f1)
            print("=================save_model_batch_size_" + \
                  "{}=================".format(batch_size))
