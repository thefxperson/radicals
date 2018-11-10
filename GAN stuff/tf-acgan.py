import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
tf.logging.set_verbosity(tf.logging.FATAL)

#code based on keras mnist_acgan example MIT licence. 

#tf.reset_default_graph()       #clear previous tensorboard stuff

#hyperparams
latent_size = 100       #noise size
epochs = 3
batch_size = 100
num_classes = 10
learning_rate = 0.0002
beta_1 = 0.5            #adam parameter
img_size = 28           #image assumed to be img_size x img_size. if img_size = 28 (ie MNIST), then the image should be 28x28

def generator(noise, label, reuse=False):
    with tf.variable_scope("Generator", reuse=reuse):
        #input preparation
        embedding = tf.Variable(tf.random_normal([num_classes, latent_size]))   #learned embedding of each class
        label_embed = tf.nn.embedding_lookup(embedding, label)                  #convert class to its embedding
        label_flat = tf.contrib.layers.flatten(label_embed)                     #flatten embedded class
        seed = tf.multiply(noise, label_flat)                                   #element-wise multiplication to create seed for GAN

        #network
        fc1 = tf.layers.dense(seed, (3*3*384))                                                                                                          #fully connected layer
        fc_reshape = tf.reshape(fc1, [-1, 3, 3, 384])
        deconv1 = tf.layers.conv2d_transpose(fc_reshape, 192, 5, strides=1, padding="valid", activation="relu", kernel_initializer="glorot_normal")     #upsample to (7x7) with deconv
        deconv1_bn = tf.layers.batch_normalization(deconv1)
        deconv2 = tf.layers.conv2d_transpose(deconv1_bn, 96, 5, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal")       #upsample to (14x14) with deconv
        deconv2_bn = tf.layers.batch_normalization(deconv2)
        deconv3 = tf.layers.conv2d_transpose(deconv2_bn, 1, 5, strides=2, padding="same", activation="tanh", kernel_initializer="glorot_normal")        #upsample to (28x28) with deconv
        deconv3_bn = tf.layers.batch_normalization(deconv3)

    return deconv3_bn

def discriminator(image, reuse=False, training=True):       #training = are you training model (if yes, apply dropout) and are you updating the weights for the disc (True) or for the generator (False)
    with tf.variable_scope("Discriminator", reuse=reuse):
        #network
        conv1 = tf.layers.conv2d(image, 32, 3, padding="same", strides=2, trainable=training)       #conv layer
        conv1_ac = tf.nn.leaky_relu(conv1)                                      #leaky relu activation
        conv1_do = tf.layers.dropout(conv1_ac, rate=0.3, training=training)                        #dropout
        conv2 = tf.layers.conv2d(conv1_do, 64, 3, padding="same", strides=1, trainable=training)    #repeat x3
        conv2_ac = tf.nn.leaky_relu(conv2)
        conv2_do = tf.layers.dropout(conv2_ac, rate=0.3, training=training)
        conv3 = tf.layers.conv2d(conv2_do, 128, 3, padding="same", strides=2, trainable=training)
        conv3_ac = tf.nn.leaky_relu(conv3)
        conv3_do = tf.layers.dropout(conv3_ac, rate=0.3, training=training)
        conv4 = tf.layers.conv2d(conv3_do, 256, 3, padding="same", strides=1, trainable=training)
        conv4_ac = tf.nn.leaky_relu(conv4)
        conv4_do = tf.layers.dropout(conv4_ac, rate=0.3, training=training)
        disc_dense = tf.contrib.layers.flatten(conv4_do)                        #dense representation of given image

        #classification
        fake = tf.layers.dense(disc_dense, 1, name="primary", trainable=training)             #is the image real or fake?
        aux = tf.layers.dense(disc_dense, num_classes, name="auxiliary", trainable=training)  #what class is the image?

    return fake, aux

def plot(samples):      #create matplotlib figure of 10 characters
    fig = plt.figure(figsize=(3, 4))
    gs = gridspec.GridSpec(3, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


#input variables
input_image = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 1))
input_noise = tf.placeholder(tf.float32, shape=(None, latent_size))
input_label = tf.placeholder(tf.int32, shape=(None, 1))
fake_label = tf.placeholder(tf.float32, shape=(None, 1))
class_label = tf.placeholder(tf.int32, shape=(None, 1))
sample_weights = tf.placeholder(tf.float32, shape=(None, 1))

#forward prop
g_img = generator(input_noise, input_label)
d_logits, d_aux = discriminator(input_image)
g_logits, g_aux = discriminator(g_img, reuse=True, training=False)  #we only want to update generator; disc is only predicting now

#losses
dl_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=fake_label))
dl_aux = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=d_aux, labels=class_label, weights=sample_weights))
disc_loss = tf.reduce_sum(tf.stack([dl_fake, dl_aux]))              #combine both losses so optimizer can properly optimize

gl_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=g_logits, labels=fake_label))
gl_aux = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=g_aux, labels=class_label))
gen_loss = tf.reduce_sum(tf.stack([gl_fake, gl_aux]))         #combine both losses so optimizer can properly optimize

#optimizers
d_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1).minimize(disc_loss)
g_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1).minimize(gen_loss)

#tensorboard
dl_summary = tf.summary.scalar("disc_loss", disc_loss)
gl_summary = tf.summary.scalar("gen_loss", gen_loss)

#load data, normalize data to be between [-1, 1]
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)
num_train = x_train.shape[0]

#training
sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("./logs/loss2", sess.graph)

#find number of batches / epoch based on batch size
num_batches = int(np.ceil(num_train/float(batch_size)))

for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch+1, epochs))
    for batch in range(num_batches):
        #real images / labels
        image_batch = x_train[batch * batch_size:(batch+1) * batch_size]
        real_labels = y_train[batch * batch_size:(batch+1) * batch_size]

        #noise / random labels
        noise = np.random.uniform(-1,1, (2*batch_size, latent_size))
        gen_label = np.random.randint(0, num_classes, 2*batch_size)

        #fake images
        gen_images = sess.run(g_img, feed_dict={input_noise: noise, input_label: gen_label.reshape((-1,1))})  #reshape labels to [batch_size, 1] so it can be fed to embedding layer as a len=1 sequence
        split_images, _ = np.split(gen_images, 2, axis=0)       #gen images generates double the images we need b/c of placeholder sizes. So split in two and only use first half
        split_labels, _ = np.split(gen_label, 2, axis=0)

        #combine real/fake
        x = np.concatenate((image_batch, split_images), axis=0)
        soft_zero, soft_one = 0, 0.95                                                                   #helps train GAN using one-sided soft real/fake labels
        y = np.array([soft_one] * batch_size + [soft_zero] * batch_size).reshape(2*batch_size, 1)       #array of Ts, then Fs. T=.95 instead of 1, F = 0
        aux_y = np.concatenate((real_labels, split_labels), axis=0).reshape(2*batch_size, 1)

        #train disc

        #when training disc, we don't want it to mess with aux classifier accuracy with generated images
        #therefore only train disc on real images
        #to keep sample weight sum, real images = 2, fake images = 0
        disc_sample_weight = np.concatenate((np.ones(batch_size)*2,np.zeros(batch_size))).reshape(2*batch_size, 1)

        dl, _ = sess.run([dl_summary, d_train_op], feed_dict={input_image: x, fake_label: y, class_label: aux_y, sample_weights: disc_sample_weight})

        #train gen

        #new noise for generator's training. len=2*batch_size so it trains on same num as disc
        noise = np.random.uniform(-1,1, (2*batch_size, latent_size)).reshape(2*batch_size, latent_size)
        gen_label = np.random.randint(0, num_classes, 2*batch_size).reshape(2*batch_size, 1)

        #we want gen to trick the disc. -- therefore we want all y labels to say not fake
        trick = np.ones(2*batch_size) * soft_one
        trick = np.reshape(trick, (200, 1))

        gl, _ = sess.run([gl_summary, g_train_op], feed_dict={input_noise: noise, input_label: gen_label, fake_label: trick, class_label: gen_label})
        writer.add_summary(gl, batch+epoch*num_batches)
        writer.add_summary(dl, batch+epoch*num_batches)

    #save images after each epoch
    noise = np.random.uniform(-1,1, (num_classes, latent_size))
    labels = np.asarray([i for i in range(num_classes)])
    samples = sess.run(g_img, feed_dict={input_noise: noise, input_label: labels.reshape((-1,1))})  #reshape labels to [batch_size, 1] so it can be fed to embedding layer as a len=1 sequence
    fig = plot(samples)
    plt.savefig('out/{}.png'.format(str(epoch+1).zfill(3)), bbox_inches='tight')
    plt.close(fig)

sess.close()