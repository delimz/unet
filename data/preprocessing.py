import functools
import tensorflow as tf
import tensorflow_addons as tfa


def sortkey(x):
    sp = x.split('_')
    return int(sp[2])*100000+int(sp[3])


def filter_fn(x, y):
    return tf.reduce_sum(y) > 0  # or tf.random.uniform([],0,1) < 0.05


def process_pathnames(fname, label_path):
    ''' fetch the image data

    fname : image filenames
    label_path : list of image labels
    '''
    # We map this function onto each pathname pair
    img_str = tf.io.read_file(fname)
    img = tf.truediv(tf.cast(tf.image.decode_jpeg(
        img_str, channels=3), dtype=tf.float32), 255.0)
    label_img_str = tf.map_fn(tf.io.read_file, label_path)
    decoder = functools.partial(tf.image.decode_jpeg)
    label_img = tf.map_fn(decoder, label_img_str, dtype=tf.uint8)
    label_img = tf.image.convert_image_dtype(
        label_img[:, :, :, 0], dtype=tf.float32)
    label_img = tf.transpose(label_img, perm=[1, 2, 0])

    return img, label_img


def get_dataset(filenames,
                labels,
                num_x,
                preproc_fn=lambda x: x,
                threads=16,
                batch_size=5,
                shuffle=True,
                buff_size=20,
                filter_empty=False,
                deterministic=True):
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if shuffle:
        dataset = dataset.shuffle(num_x)

    dataset = dataset.map(
        process_pathnames, num_parallel_calls=threads, deterministic=deterministic)
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    if filter_empty:
        dataset = dataset.filter(filter_fn)

    dataset = dataset.map(
        preproc_fn, num_parallel_calls=threads, deterministic=deterministic)

    return dataset
    # It's necessary to repeat our data for all epochs
    #dataset = dataset.batch(batch_size)

    # return dataset.prefetch(buff_size)


def crop_strech_img(img, label_img, deformation_range, img_shape, orig_shape=1024):
    x_deform = img_shape[0]*deformation_range
    y_deform = img_shape[1]*deformation_range
    if deformation_range != 0:
        cx = tf.random.uniform([], x_deform/2+img_shape[0]/2,
                               orig_shape-img_shape[0]/2-x_deform/2)
        cy = tf.random.uniform([], y_deform/2+img_shape[1]/2,
                               orig_shape-img_shape[1]/2-y_deform/2)
    else:
        cx = orig_shape/2
        cy = orig_shape/2
    y1 = cy-img_shape[0]/2
    y2 = cy+img_shape[0]/2
    x1 = cx-img_shape[1]/2
    x2 = cx+img_shape[1]/2
    dx = 0  # tf.random.uniform([],0.0,x_deform/2)
    dy = 0  # tf.random.uniform([],0.0,y_deform/2)
    bbox = tf.add(tf.stack([y1, x1, y2, x2]), [-dy, -dx, dy, dx])/orig_shape

    bbox = tf.reshape(tf.cast(bbox, tf.float32), [1, 4])
    box_ind = [0]

    crop_size = [img_shape[0], img_shape[1]]
    img = tf.image.crop_and_resize([img], bbox, box_ind, crop_size)[0]
    label_img = tf.image.crop_and_resize(
        [label_img], bbox, box_ind, crop_size)[0]
    return img, label_img


def rotate_img(img, label_img, deg):
    angle = tf.random.uniform([], -deg, deg)
    img = tfa.image.rotate(img, angle*(3.14159265358979323846/180))
    label_img = tfa.image.rotate(label_img, angle*(3.14159265358979323846/180))
    return img, label_img


def augment(img,
            label_img,
            resize=None,  # Resize the image to some size e.g. [256, 256]
            hue_delta=0,  # Adjust the hue of an RGB image by random factor
            saturation_delta=0,
            brightness_delta=0,
            contrast_delta=0,
            rotate=0,
            deformation_range=0,
            nn_input_shape=None,
            nn_output_shape=None,
            background_class=False,
            orig_shape=1024):

    if rotate != 0:
        img, label_img = rotate_img(img, label_img, rotate)
    img, label_img = crop_strech_img(
        img, label_img, deformation_range, nn_input_shape, orig_shape)

    label_img = tf.image.central_crop(
        label_img, nn_output_shape[0]/nn_input_shape[0])
    if hue_delta:
        img = tf.image.random_hue(img, hue_delta)
    if saturation_delta:
        img = tf.image.random_saturation(
            img, 1 - saturation_delta, 1+saturation_delta)
    if brightness_delta:
        img = tf.image.random_brightness(img, brightness_delta)
    if contrast_delta:
        img = tf.image.random_contrast(
            img, 1 - contrast_delta, 1 + contrast_delta)

    if background_class:
        neg = tf.maximum(tf.ones_like(
            label_img[:, :, 0:1])-tf.reduce_sum(label_img, axis=2, keep_dims=True), 0)
        label_img = tf.math.softmax(tf.concat([label_img, neg], axis=-1))

    return img, label_img
