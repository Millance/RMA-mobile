import os.path
from common import get_train_ds, get_val_ds, cosine_scheduler, log_data
import time
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from RMA_mobile import RMA_mobile

# 参数
im_height = 224
im_width = 224
batch_size = 128
epochs = 150
num_classes = 4
random_seed = 1000
drop_out = 0.5
initial_lr = 0.005


def main():
    # 路径
    train_path = r"\train_dataset"
    val_path = r"\val_dataset"
    now_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")  # 格式化时间字符串
    # 归类文件
    floder_name = "RMA_mobile_train"
    if not os.path.exists("./{}".format(floder_name)):
        os.makedirs("./{}".format(floder_name))

    log_name = "./{}/{}_RMA_origin.csv".format(floder_name, now_time)
    weight_dir = "{}/{}_weights_RMA_origin".format(floder_name, now_time)
    log_data(log_name,
             ["Epoch", "Wall Time", "Train Time", "Train Loss", "Train Accuracy", "Validate Loss",
              "Validate Accuracy", "Best Validate Accuracy"])

    log_dir = "./{}/boards/{}_RMA_origin".format(floder_name, now_time)
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    # 权重文件
    if not os.path.exists("./{}".format(weight_dir)):
        os.makedirs("./{}".format(weight_dir))
    # 获取数据集
    train_ds = get_train_ds(train_path, im_height, im_width, batch_size)
    val_ds = get_val_ds(val_path, im_height, im_width, batch_size)

    # custom learning rate curve
    scheduler = cosine_scheduler(initial_lr, epochs, len(train_ds), train_writer=train_writer)

    model = RMA_mobile(input_shape=(im_height, im_width, 3), num_classes=num_classes)
    # model.summary()

    # 为训练选择优化器与损失函数：
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    # 选择衡量指标来度量模型的损失值（loss）和准确率（accuracy）。这些指标在 epoch 上累积值，然后打印出整体结果。
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    # 使用 tf.GradientTape 来训练模型：
    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            loss = loss_object(train_labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(train_labels, output)

    # 测试模型
    @tf.function
    def val_step(val_images, val_labels):
        output = model(val_images, training=False)
        loss = loss_object(val_labels, output)

        val_loss(loss)
        val_accuracy(val_labels, output)

    best_val_acc = 0.
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        t1 = time.perf_counter()
        # train
        train_bar = tqdm(train_ds, colour="yellow")
        for images, labels in train_bar:
            # update learning rate
            optimizer.learning_rate = next(scheduler)
            train_step(images, labels)
            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}" \
                .format(epoch + 1, epochs, train_loss.result(), train_accuracy.result())

        # validate
        val_bar = tqdm(val_ds, colour="blue")
        for images, labels in val_bar:
            val_step(images, labels)
            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}, best:{:.3f}" \
                .format(epoch + 1, epochs, val_loss.result(), val_accuracy.result(), best_val_acc)

        train_time = time.perf_counter() - t1
        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            model.save_weights("./{}/RMA.ckpt"
                               .format(weight_dir), save_format="tf")

        # 写入日志
        log_data(log_name, [epoch + 1, time.time(), train_time,
                            train_loss.result().numpy(),
                            train_accuracy.result().numpy(),
                            val_loss.result().numpy(),
                            val_accuracy.result().numpy(),
                            best_val_acc.numpy()])
        # writing training loss and acc
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        # writing validation loss and acc
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)
    best_acc_log = "./{}/train_best.csv".format(floder_name)
    log_data(best_acc_log, [now_time, best_val_acc.numpy()])


if __name__ == "__main__":
    main()
