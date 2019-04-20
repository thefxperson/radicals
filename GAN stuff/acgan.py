import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow import keras
from dataset import make_dataset

#quiet tensorflow down a bit
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
tf.logging.set_verbosity(tf.logging.FATAL)

#code based on keras mnist_acgan example MIT licence. 

#global const
num_classes = 10
latent_size = 100       #noise size
epochs = 25
batch_size = 32
learning_rate = 0.0005   
beta_1 = 0.5            #adam parameter

def make_generator():
    #input prep
    image_class = keras.layers.Flatten()(keras.layers.Embedding(num_classes, latent_size, embeddings_initializer="glorot_normal")(label))
    h = keras.layers.multiply([noise, image_class])     #hadamard product between noise and class conditional embedding
    #network
    gen = keras.Sequential([
        keras.layers.Dense(4*4*256, input_dim=latent_size, activation="relu"),
        keras.layers.Reshape((4,4,256)),
        keras.layers.Conv2DTranspose(128, 5, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal"),    #upsample to (8x8) with deconvolutional layer
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(128, 5, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal"),  #upsample to (16x16) with deconvolutional layer
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal"),  #upsample to (32x32) with deconvolutional layer
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal"),  #upsample to (64x64) with deconvolutional layer
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(1, 2, strides=2, padding="same", activation="tanh", kernel_initializer="glorot_normal")])  #upsample to (128x128) with deconvolutional layer
    #gen.summary()
    gen_img = gen(h)    #generated image
    return keras.Model([noise, label], gen_img)

def make_discriminator():
    #input
    image = keras.layers.Input(shape=(128, 128, 1))
    #network
    dis = keras.Sequential([
        keras.layers.Conv2D(128, 5, padding="same", strides=2, input_shape=(128,128,1)),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, 5, padding="same", strides=1),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, 3, padding="same", strides=2),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, 3, padding="same", strides=1),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten()])
    #discrimination
    features = dis(image)
    fake = keras.layers.Dense(1, activation="sigmoid", name="generation")(features)             #does disc think image is real or generated?
    aux = keras.layers.Dense(num_classes, activation="softmax", name="auxiliary")(features)     #what class does disc think image is from?
    return keras.Model(image, [fake, aux])

#plot example images (only for num_classes<=12)
def plot(samples):
    fig = plt.figure(figsize=(3, 4))
    gs = gridspec.GridSpec(3, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(128, 128), cmap='Greys_r')
    return fig

#normalize data to be between [-1, 1] (applied using dataset.map to parallelize)
def normalize_data(image, label):
    tf.divide(tf.subtract(image, 127.5), 127.5)
    return tf.expand_dims(image, axis=-1), label


#build discriminator
print("Driscriminator Model:")
discriminator = make_discriminator()
discriminator.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1), loss=["binary_crossentropy", "sparse_categorical_crossentropy"])
discriminator.summary()

#build generator
noise = keras.layers.Input(shape=(latent_size,))
label = keras.layers.Input(shape=(1,), dtype="int32")
generator = make_generator()
gen_img = generator([noise, label])
discriminator.trainable = False     #only want to train generator in combined model
fake, aux = discriminator(gen_img)
combined = keras.Model([noise, label], [fake, aux])
print("Combined Model:")
combined.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1), loss=["binary_crossentropy", "sparse_categorical_crossentropy"])
combined.summary()

#load data
x_train, y_train = make_dataset(num_classes)
num_train = x_train.shape[0]

#create placeholders and dataset
x_placeholder = tf.placeholder(x_train.dtype, x_train.shape)
y_placeholder = tf.placeholder(y_train.dtype, y_train.shape)

#create dataset from placeholders. Repeat the images pulled 30 times (math below). Shuffle to the size of the dataset so order is random. normalize data between [-1, 1]. split into batches, and prefetch 1 batch
#MNIST 60000 images 10 classes. thus 6k ex/class. ETL9G has 200ex/class. 30x that so they have the same number of images per epoch and epochs can remain the same
dataset = tf.data.Dataset.from_tensor_slices((x_placeholder, y_placeholder)).repeat(30).shuffle(30*200*num_classes).map(normalize_data).batch(batch_size=batch_size).prefetch(1)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


#training
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(iterator.initializer, feed_dict={x_placeholder: x_train, y_placeholder: y_train})

#find number of batches / epoch based on batch size
num_batches = int(np.ceil(num_train/float(batch_size)))
for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch+1, epochs))

    gen_loss = []
    disc_loss = []
    for batch in range(num_batches):
        #real images / labels
        image_batch, real_labels = sess.run(next_element)

        #print reference image
        if epoch == 0 and batch == 0:
            fig = plot(image_batch[0:10])
            plt.savefig('out/base.png'.format(str(epoch+1).zfill(3)), bbox_inches='tight')
            plt.close(fig)

        #noise / random labels
        noise = np.random.normal(size=(batch_size, latent_size))
        fake_labels = np.random.randint(0, num_classes, batch_size)

        #fake images
        gen_images = generator.predict([noise, fake_labels.reshape((-1,1))], verbose=0)     #reshape labels to [batch_size, 1] so it can be fed to embedding layer as a len=1 sequence

        #combine real/fake
        x = np.concatenate((image_batch, gen_images))
        soft_zero, soft_one = 0.05, 0.95       #helps train GAN using one-sided soft real/fake labels
        y = np.array([soft_zero] * batch_size + [soft_one] * batch_size)                    #array of Ts, then Fs. T=.05 instead of 0, F = .95
        aux_y = np.concatenate((real_labels, fake_labels), axis=0)

        #train disc
        #when training disc, we don't want it to mess with aux classifier accuracy with generated images
        #therefore only train disc on real images
        #to keep sample weight sum, real images = 2, fake images = 0
        disc_sample_weight = [np.ones(2*batch_size), np.concatenate((np.ones(batch_size)*2,np.zeros(batch_size)))]
        dl = discriminator.train_on_batch(x, [y, aux_y], sample_weight=disc_sample_weight)

        #new noise for generator's training. len=2*batch_size so it trains on same num as disc
        noise = np.random.uniform(-1,1, (2*batch_size, latent_size))
        fake_labels = np.random.randint(0, num_classes, 2*batch_size)

        #train gen
        #we want gen to trick the disc. -- therefore we want all y labels to say not fake
        trick = np.ones(2*batch_size) * soft_zero
        gl = combined.train_on_batch([noise, fake_labels.reshape((-1,1))], [trick, fake_labels])

        #update losses
        disc_loss.append(dl)
        gen_loss.append(gl)

    #save images after each epoch
    noise = np.random.uniform(-1,1, (num_classes, latent_size))
    labels = np.asarray([i for i in range(num_classes)])
    samples = generator.predict([noise, labels.reshape((-1,1))])
    fig = plot(samples)
    plt.savefig('out/{}.png'.format(str(epoch+1).zfill(3)), bbox_inches='tight')
    plt.close(fig)

    #print losses
    print("Disc Loss: {} Gen Loss: {}".format(np.mean(gen_loss), np.mean(disc_loss)))

    #fetch next epoch's data
    sess.run(iterator.initializer, feed_dict={x_placeholder: x_train, y_placeholder: y_train})
sess.close() 