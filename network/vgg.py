import tensorflow as tf
import numpy as np

from misc.layers import dense, conv2d, test_conv2d_flip, conv2d_fully_indep
from network.registry import register_model


def VGG(inputs, sampler, is_training, batch_norm, layer_collection, particles, num_blocks, config):
    def VGGBlock(inputs, layers, out_channel, layer_idx, use_flip, use_fully_indep, batch_size):
        l2_loss = 0.
        for l in range(layers):
            in_channel = inputs.shape.as_list()[-1]
            sampler.register_block(layer_idx+l, (3, 3, in_channel, out_channel))
            if use_fully_indep :
                print("use fully independent conv2d")
                weights = sampler.sample(layer_idx+l)
                l2_loss += 0.5 * tf.reduce_sum(weights ** 2)
                pre, act = conv2d_fully_indep(inputs, weights, (3, 3, in_channel, out_channel),
                                  batch_norm, is_training, batch_size, particles, padding="SAME")  
            elif use_flip :
                print("Use flip layer for conv2d")
                weights = sampler.sample(layer_idx+l)
                weights_mean, u_c, v_c = sampler.sample_mean_and_var(layer_idx+l, nobias_flag=True)
                l2_loss += 0.5 * tf.reduce_sum(weights_mean ** 2)
                pre, act = test_conv2d_flip(inputs, weights, (3, 3, in_channel, out_channel), batch_norm, is_training, weights_mean, u_c, v_c, config.batch_size, particles, padding="SAME")

            else :
                print("Do not Use flip layer for conv2d")
                weights = sampler.sample(layer_idx+l)
                l2_loss += 0.5 * tf.reduce_sum(weights ** 2)
                pre, act = conv2d(inputs, weights, (3, 3, in_channel, out_channel),
                                  batch_norm, is_training, particles, padding="SAME")
            layer_collection.register_conv2d(sampler.get_params(layer_idx+l), (1, 1, 1, 1), "SAME", inputs, pre)
            inputs = act
            
        outputs = tf.layers.max_pooling2d(inputs, 2, 2, "SAME")
        return outputs, l2_loss

    inputs = tf.tile(inputs, [particles, 1, 1, 1])
    layer_idx = 0
    # block 1
    layer1, l2_loss1 = VGGBlock(inputs, num_blocks[0], 32, layer_idx, config.use_flip, 
        config.use_fully_indep, config.batch_size)
    layer_idx += num_blocks[0]
    # block 2
    layer2, l2_loss2 = VGGBlock(layer1, num_blocks[1], 64, layer_idx, config.use_flip, 
        config.use_fully_indep, config.batch_size)
    layer_idx += num_blocks[1]
    # block 3
    layer3, l2_loss3 = VGGBlock(layer2, num_blocks[2], 128, layer_idx, config.use_flip, 
        config.use_fully_indep, config.batch_size)
    layer_idx += num_blocks[2]
    # block 4
    layer4, l2_loss4 = VGGBlock(layer3, num_blocks[3], 256, layer_idx, config.use_flip, 
        config.use_fully_indep, config.batch_size)
    layer_idx += num_blocks[3]
    # block 5
    layer5, l2_loss5 = VGGBlock(layer4, num_blocks[4], 256, layer_idx, config.use_flip, 
        config.use_fully_indep, config.batch_size)
    layer_idx += num_blocks[4]

    # l2_loss
    l2_loss = l2_loss1 + l2_loss2 + l2_loss3 + l2_loss4 + l2_loss5

    flat = tf.reshape(layer5, shape=[-1, int(np.prod(layer5.shape[1:]))])
    sampler.register_block(layer_idx, (256, 10))
    weights = sampler.sample(layer_idx)
    l2_loss += 0.5 * tf.reduce_sum(weights ** 2)
    logits, _ = dense(flat, weights, batch_norm, is_training, particles)
    layer_collection.register_fully_connected(sampler.get_params(layer_idx), flat, logits)
    layer_collection.register_categorical_predictive_distribution(logits, name="logits")

    return logits, l2_loss


@register_model("vgg11")
def VGG11(inputs, sampler, is_training, batch_norm, layer_collection, particles, config):
    return VGG(inputs, sampler, is_training, batch_norm, layer_collection, particles, [1, 1, 2, 2, 2], config)


@register_model("vgg13")
def VGG13(inputs, sampler, is_training, batch_norm, layer_collection, particles, config):
    return VGG(inputs, sampler, is_training, batch_norm, layer_collection, particles, [2, 2, 2, 2, 2], config)


@register_model("vgg16")
def VGG16(inputs, sampler, is_training, batch_norm, layer_collection, particles, config):
    return VGG(inputs, sampler, is_training, batch_norm, layer_collection, particles, [2, 2, 3, 3, 3], config)


@register_model("vgg19")
def VGG19(inputs, sampler, is_training, batch_norm, layer_collection, particles, config):
    return VGG(inputs, sampler, is_training, batch_norm, layer_collection, particles, [2, 2, 4, 4, 4], config)
