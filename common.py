import json
import math
import os
import random

import numpy as np
import tensorflow as tf

height = 0
width = 0
# 方块的行、列个数
cell_count = 7
# 每个方块的像素
cell_size = 32
# 方块内部，远离边缘的像素
in_cell_margin = 3
# 方块内，正方形遮盖区域的长度
block_size = 14


def read_val_data(val_path):
    assert os.path.exists(val_path), "dataset path: {} does not exist.".format(val_path)
    # 遍历每个文件夹（类别）
    classes = [cls for cls in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, cls))]
    # 排序
    classes.sort()
    # 建立字典
    cls_dict = dict((k, v) for k, v in enumerate(classes))
    json_str = json.dumps(dict((k, v) for k, v in cls_dict.items()), indent=3)
    with open("class_indices.json", 'w') as js_file:
        js_file.write(json_str)

    val_paths = []  # 存储验证集的所有图片路径
    val_labels = []  # 存储验证集图片对应索引信息
    supported = [".jpg", ".JPG", ".jpeg", ".JPEG"]  # 支持的文件后缀类型
    # 遍历每个类别下的图片
    for cls in classes:
        val_img_path = os.path.join(val_path, cls)
        # 所有图片
        val_imgs = [os.path.join(val_img_path, img_name) for img_name in os.listdir(val_img_path)
                    if os.path.splitext(img_name)[-1] in supported]
        # 获取该类别对应的索引
        img_cls = get_key(cls_dict, cls)

        for img_path in val_imgs:
            # 否则存入训练集
            val_paths.append(img_path)
            val_labels.append(img_cls)
    return val_paths, val_labels


def read_predict_data(prd_path):
    assert os.path.exists(prd_path), "dataset path: {} does not exist.".format(prd_path)
    # 遍历每个文件夹（类别）
    classes = [cls for cls in os.listdir(prd_path) if os.path.isdir(os.path.join(prd_path, cls))]
    # 排序
    classes.sort()
    # 建立字典
    cls_dict = dict((k, v) for k, v in enumerate(classes))

    prd_paths = []  # 存储验证集的所有图片路径
    prd_labels = []  # 存储验证集图片对应索引信息
    supported = [".jpg", ".JPG", ".jpeg", ".JPEG"]  # 支持的文件后缀类型
    # 遍历每个类别下的图片
    for cls in classes:
        prd_img_path = os.path.join(prd_path, cls)
        # 所有图片
        prd_imgs = [os.path.join(prd_img_path, img_name) for img_name in os.listdir(prd_img_path)
                    if os.path.splitext(img_name)[-1] in supported]
        # 获取该类别对应的索引
        img_cls = get_key(cls_dict, cls)

        for img_path in prd_imgs:
            # 否则存入训练集
            prd_paths.append(img_path)
            prd_labels.append(img_cls)
    return prd_paths, prd_labels


def get_val_ds(val_path, im_height, im_width, batch_size):
    global height, width
    height = im_height
    width = im_width
    # 读取所有路径
    val_paths, val_labels = read_val_data(val_path)
    # 使用值tf.data.experimental.AUTOTUNE，则根据可用的CPU动态设置并行调用的数量。
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # Deal with val ds ----- Begin -----
    val_ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(val_paths), tf.constant(val_labels)
    ))
    # 使用Dataset.map为模型准备数据
    val_ds = val_ds.map(handle_val_img, num_parallel_calls=AUTOTUNE)
    val_ds = configure_for_performance(val_ds, batch_size=batch_size, buffer_size=AUTOTUNE)
    # Deal with val ds -----  End  -----
    return val_ds


def get_predict_ds(val_path, im_height, im_width, batch_size):
    global height, width
    height = im_height
    width = im_width
    # 读取所有路径
    prd_paths, prd_labels = read_predict_data(val_path)
    # 使用值tf.data.experimental.AUTOTUNE，则根据可用的CPU动态设置并行调用的数量。
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # Deal with val ds ----- Begin -----
    prd_ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(prd_paths), tf.constant(prd_labels)
    ))
    # 使用Dataset.map为模型准备数据
    prd_ds = prd_ds.map(handle_prd_img, num_parallel_calls=AUTOTUNE)
    prd_ds = configure_for_performance(prd_ds, batch_size=batch_size, buffer_size=AUTOTUNE)
    # Deal with val ds -----  End  -----
    return prd_ds


def read_train_data(train_path):
    """
    读取路径中包含的文件（按文件夹分类），
    :return: 测试集和标签
    """
    assert os.path.exists(train_path), "dataset path: {} does not exist.".format(train_path)
    # 遍历每个文件夹（类别）
    classes = [cls for cls in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, cls))]
    # 排序
    classes.sort()
    # 建立字典
    cls_dict = dict((k, v) for k, v in enumerate(classes))

    train_paths = []  # 存储训练集的所有图片路径
    train_labels = []  # 存储训练集图片对应索引信息
    supported = [".jpg", ".JPG", ".jpeg", ".JPEG"]  # 支持的文件后缀类型
    # 遍历每个类别下的图片
    for cls in classes:
        train_dir = os.path.join(train_path, cls)
        for cls_name in ['L', 'R']:
            train_img_path = os.path.join(train_dir, cls_name)
            # 所有图片
            train_imgs = [os.path.join(train_img_path, img_name) for img_name in os.listdir(train_img_path)
                          if os.path.splitext(img_name)[-1] in supported]
            # 获取该类别对应的索引
            img_cls = get_key(cls_dict, cls)

            for img_path in train_imgs:
                # 存入训练集
                train_paths.append(img_path)
                train_labels.append(img_cls)
    return train_paths, train_labels


def get_train_ds(train_path, im_height, im_width, batch_size):
    global height, width
    height = im_height
    width = im_width
    # 读取所有路径
    train_paths, train_labels = read_train_data(train_path)
    # 使用值tf.data.experimental.AUTOTUNE，则根据可用的CPU动态设置并行调用的数量。
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # Deal with train ds ----- Begin -----
    train_ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(train_paths), tf.constant(train_labels)
    ))
    train_cnt = len(train_paths)
    # 使用Dataset.map为模型准备数据
    train_ds = train_ds.map(handle_train_img, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds, train_cnt, shuffle=True,
                                         batch_size=batch_size, buffer_size=AUTOTUNE)
    # Deal with train ds -----  End  -----
    return train_ds


def read_train_mix_data(train_path):
    """
    读取路径中包含的文件（按文件夹分类），
    :return: 随机抽取后的测试集、混合文件路径和标签
    """
    assert os.path.exists(train_path), "dataset path: {} does not exist.".format(train_path)
    # 遍历每个文件夹（类别）
    classes = [cls for cls in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, cls))]
    # 排序
    classes.sort()
    # 建立字典
    cls_dict = dict((k, v) for k, v in enumerate(classes))

    train_paths = []  # 存储训练集的所有图片路径
    mix_paths = []  # 存储训练集对应混合图片路径
    train_labels = []  # 存储训练集图片对应索引信息
    supported = [".jpg", ".JPG", ".jpeg", ".JPEG"]  # 支持的文件后缀类型
    # 遍历每个类别下的图片
    for cls in classes:
        train_dir = os.path.join(train_path, cls)
        for cls_name in ['L', 'R']:
            train_img_path = os.path.join(train_dir, cls_name)
            # 所有图片
            train_imgs = [os.path.join(train_img_path, img_name) for img_name in os.listdir(train_img_path)
                          if os.path.splitext(img_name)[-1] in supported]
            # 混合图片
            mix_imgs = train_imgs.copy()
            # 打乱混合图片的顺序
            random.shuffle(mix_imgs)
            # 获取该类别对应的索引
            img_cls = get_key(cls_dict, cls)

            for img_path in train_imgs:
                # 存入训练集
                train_paths.append(img_path)
                train_labels.append(img_cls)
            for mix_path in mix_imgs:
                # 存入混合集
                mix_paths.append(mix_path)

    return train_paths, mix_paths, train_labels


def get_mix_train_ds(train_path, im_height, im_width, mask_size, batch_size):
    global height, width, block_size
    height = im_height
    width = im_width
    # 读取所有路径
    train_paths, mix_paths, train_labels = read_train_mix_data(train_path)
    # 使用值tf.data.experimental.AUTOTUNE，则根据可用的CPU动态设置并行调用的数量。
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # Deal with train ds ----- Begin -----
    train_ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(train_paths), tf.constant(mix_paths), tf.constant(train_labels)
    ))
    train_cnt = len(train_paths)
    block_size = mask_size
    # 使用Dataset.map为模型准备数据
    train_ds = train_ds.map(handle_train_mix_img, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds, train_cnt, shuffle=True,
                                         batch_size=batch_size, buffer_size=AUTOTUNE)
    # Deal with train ds -----  End  -----
    return train_ds


def log_data(log_name, log_info: []):
    with open(log_name, 'a') as log_file:
        for item in log_info:
            log_file.write(str(item) + ",")
        log_file.write("\n")


def cosine_rate(now_step, total_step, end_lr_rate):
    rate = ((1 + math.cos(now_step * math.pi / total_step)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
    return rate


def cosine_scheduler(initial_lr, epochs, steps, warmup_epochs=1, end_lr_rate=1e-6, train_writer=None):
    """custom learning rate scheduler"""
    assert warmup_epochs < epochs
    warmup = np.linspace(start=1e-8, stop=initial_lr, num=warmup_epochs * steps)
    remainder_steps = (epochs - warmup_epochs) * steps
    cosine = initial_lr * np.array([cosine_rate(i, remainder_steps, end_lr_rate) for i in range(remainder_steps)])
    lr_list = np.concatenate([warmup, cosine])

    for i in range(len(lr_list)):
        new_lr = lr_list[i]
        if train_writer is not None:
            # writing lr into tensorboard
            with train_writer.as_default():
                tf.summary.scalar('learning rate', data=new_lr, step=i)
        yield new_lr


# region Tools
def get_keys(d, value):
    return [k for k, v in d.items() if v == value]


def get_key(d, value):
    return [k for k, v in d.items() if v == value][0]


def draw_block(g, delta, a, b):
    """
    基于 tensorflow的
    抖动的矩形掩码
    掩码格式 0-1
    224大小，每个block为32，一共7*7=49个block
    """
    mask_block = np.zeros([g, g])
    # 随机位置
    rd_h = 0
    rd_w = 0
    if (g - 2 * delta - a) > 0:
        rd_h = np.random.randint(g - 2 * delta - a)
        rd_w = np.random.randint(g - 2 * delta - b)
    for i in range(g):
        for j in range(g):
            if (delta + rd_h) < i < (delta + rd_h + a) and (delta + rd_w) < j < (delta + rd_w + b):
                # 挖掉
                mask_block[i][j] = 0
            else:
                # 保留
                mask_block[i][j] = 1
    return mask_block


def stack_block(cell_count, cell_size, in_cell_margin, block_size):
    line_block = []
    for i in range(cell_count):
        mask_stack = []
        for j in range(cell_count):
            temp_cell = draw_block(cell_size, in_cell_margin, block_size, block_size)
            if j == 0:
                mask_stack = temp_cell
            else:
                mask_stack = np.hstack((mask_stack, temp_cell))
        if i == 0:
            line_block = mask_stack
        else:
            line_block = np.vstack((line_block, mask_stack))
    return line_block


def handle_train_masked_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [height, width], method=tf.image.ResizeMethod.AREA)
    img = tf.image.random_flip_left_right(img)

    mask_2d = stack_block(cell_count, cell_size, in_cell_margin, block_size)
    mask_3d = np.stack([mask_2d, mask_2d, mask_2d], axis=2)
    mask_3d_tensor = tf.constant(mask_3d, tf.float32)
    masked_img = img * mask_3d_tensor
    # img = (img - 0.5) * 2
    return masked_img, label


def handle_train_mix_img(path, mix_path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [height, width], method=tf.image.ResizeMethod.AREA)
    # padding+裁剪  另一种：放大+裁剪
    img = tf.pad(img, [[20, 20], [20, 20], [0, 0]])
    img = tf.image.random_crop(img, size=(height, width, 3))

    mix_img = tf.io.read_file(mix_path)
    mix_img = tf.image.decode_jpeg(mix_img, channels=3)
    mix_img = tf.image.resize(mix_img, [height, width], method=tf.image.ResizeMethod.AREA)

    mask_2d = stack_block(cell_count, cell_size, in_cell_margin, block_size)
    mask_3d = np.stack([mask_2d, mask_2d, mask_2d], axis=2)
    mask_3d_tensor = tf.constant(mask_3d, tf.float32)
    reversed_mask_3d = tf.constant(1 - mask_3d, tf.float32)

    masked_img = img * mask_3d_tensor
    reversed_mask_img = mix_img * reversed_mask_3d
    mixed_img = masked_img + reversed_mask_img
    mixed_img = tf.image.random_flip_left_right(mixed_img)
    return mixed_img, label


def handle_train_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [height, width], method=tf.image.ResizeMethod.AREA)
    img = tf.image.random_flip_left_right(img)
    # img = (img - 0.5) * 2
    return img, label


def handle_val_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [height, width], method=tf.image.ResizeMethod.AREA)
    # img = (img - 0.5) * 2
    return img, label


def handle_prd_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [height, width], method=tf.image.ResizeMethod.AREA)
    # img = (img - 0.5) * 2
    return img, label, path


# Configure dataset for performance
def configure_for_performance(ds,
                              shuffle_size=0,
                              shuffle: bool = False,
                              batch_size=-1,
                              buffer_size=-1):
    ds = ds.cache()  # 读取数据后缓存至内存
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_size)  # 打乱数据顺序
    ds = ds.batch(batch_size)  # 指定batch size
    ds = ds.prefetch(buffer_size=buffer_size)  # 在训练的同时提前准备下一个step的数据
    return ds
# endregion
