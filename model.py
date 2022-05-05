from utils.common import exists
import tensorflow as tf
import neuralnet as nn
import numpy as np

# -----------------------------------------------------------
# VDSR
# -----------------------------------------------------------

class VDSR:
    def __init__(self): 
        self.model = nn.VDSR_model()
        self.optimizer = None
        self.loss =  None
        self.metric = None
        self.model_path = None
        self.ckpt = None
        self.ckpt_dir = None
        self.ckpt_man = None
    
    def setup(self, optimizer, loss, metric, model_path):
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        # @the best model weights
        self.model_path = model_path
    
    def load_checkpoint(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir
        self.ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), 
                                        optimizer=self.optimizer,
                                        net=self.model)
        self.ckpt_man = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=1)
        self.ckpt.restore(self.ckpt_man.latest_checkpoint)
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def predict(self, lr):
        sr = self.model(lr)
        return sr
    
    def evaluate(self, dataset, batch_size=64):
        losses, metrics = [], []
        isEnd = False
        while isEnd == False:
            lr, hr, isEnd = dataset.get_batch(batch_size, shuffle_each_epoch=False)
            sr = self.predict(lr)
            losses.append(self.loss(hr, sr))
            metrics.append(self.metric(hr, sr))

        metric = tf.reduce_mean(metrics).numpy()
        loss = tf.reduce_mean(losses).numpy()
        return loss, metric

    def train(self, train_set, valid_set, batch_size, 
              epochs, save_best_only=False):

        cur_epoch = self.ckpt.epoch.numpy()
        max_epoch = epochs + self.ckpt.epoch.numpy()

        prev_loss = np.inf
        if save_best_only and exists(self.model_path):
            self.load_weights(self.model_path)
            prev_loss, _ = self.evaluate(valid_set)
            self.load_checkpoint(self.ckpt_dir)

        while cur_epoch < max_epoch:
            cur_epoch += 1
            self.ckpt.epoch.assign_add(1)
            loss_array = []
            metric_array = []
            isEnd = False
            while isEnd == False:
                lr, hr, isEnd = train_set.get_batch(batch_size)
                loss, metric = self.train_step(lr, hr)
                loss_array.append(loss)
                metric_array.append(metric)

            val_loss, val_metric = self.evaluate(valid_set)
            print(f"Epoch {cur_epoch}/{max_epoch}",
                  f"- loss: {tf.reduce_mean(loss_array).numpy():.7f}",
                  f"- {self.metric.__name__}: {tf.reduce_mean(metric_array).numpy():.3f}",
                  f"- val_loss: {val_loss:.7f}",
                  f"- val_{self.metric.__name__}: {val_metric:.3f}")

            self.ckpt_man.save(checkpoint_number=0)

            if save_best_only and val_loss > prev_loss:
                continue
            prev_loss = val_loss
            self.model.save_weights(self.model_path)
            print(f"Save model to {self.model_path}\n")

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            sr = self.model(lr, training=True)
            loss = self.loss(hr, sr)
            metric = self.metric(hr, sr)
        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        return loss, metric

