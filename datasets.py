import tensorflow as tf
import tensorflow_datasets as tfds


def pre_train_preprocessing(x, y):
    # x = tf.image.resize(x, (224, 224))
    # x = tf.image.random_hue(x, 0.25)
    # x = tf.image.random_brightness(x, max_delta=63)
    # x = tf.image.random_contrast(x, lower=0.2, upper=1.8)
    # x = tf.cast(tf.transpose(x, [0,3,1,2]), tf.float32) / 255.
    x = tf.reshape(-1, 784) / 255.

    return x, y


def load_train_ds(name, batch_size, preprocess_fn, 
    seed=42, num_prefetch=5, slice_point=75):
    train_ds = tfds.load(name=name, split=f"train[:{slice_point}%]")
    train_ds = train_ds.shuffle(int(len(train_ds)/seed), seed=seed
        ).batch(batch_size).prefetch(num_prefetch)
    # train_set = train_set.shuffle(len(train_set), seed=seed, reshuffle_each_iteration=True).batch(batch_size).map(preprocess_fn).prefetch(num_prefetch)

    return train_ds


def load_valid_ds(name, batch_size, preprocess_fn, 
    seed=42, num_prefetch=5, slice_point=75):
    valid_ds = tfds.load(name=name, split=f"train[{slice_point}%:]")
    valid_ds = valid_ds.shuffle(int(len(valid_ds)/seed), seed=seed
        ).batch(batch_size).prefetch(num_prefetch)

    return valid_ds


def load_test_ds(name, batch_size, preprocess_fn, num_prefetch):
    test_ds = tfds.load(name=name, split="test")
    test_ds = test_ds.batch(batch_size).prefetch(num_prefetch)

    return test_ds


def load_tfds(name, batch_size, preprocess_fn, seed=42, num_prefetch=5, slice_point=75):
    train_ds = load_train_ds(name, batch_size, preprocess_fn, 
        seed=seed, num_prefetch=num_prefetch, slice_point=slice_point)
    valid_ds = load_valid_ds(name, batch_size, preprocess_fn, 
        seed=seed, num_prefetch=num_prefetch, slice_point=slice_point)
    test_ds = load_test_ds(name, batch_size, preprocess_fn, num_prefetch)

    return train_ds, valid_ds, test_ds