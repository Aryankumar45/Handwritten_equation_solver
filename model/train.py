import os
# import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras import Sequential
from keras.utils import image_dataset_from_directory
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input


os.system("cls")

train = image_dataset_from_directory(
    "D:\\tech\\programming language\\equation solver\\dataset\\data_aug",
    batch_size=128,
    seed=42,
    subset="training",
    validation_split=0.3,
    image_size=(224,224)
)

test = image_dataset_from_directory(
    "D:\\tech\\programming language\\equation solver\\dataset\\data_aug",
    batch_size=128,
    seed=42,
    subset="validation",
    validation_split=0.3,
    image_size=(224,224)
)

val = image_dataset_from_directory(
    "D:\\tech\\programming language\\equation solver\\dataset\\test",
    batch_size=128,
    seed=42,
    image_size=(224,224)
)

# for images, labels in train_data.take(1):
#     plt.figure(figsize=(10,10))
#     for i in range(56):
#         ax = plt.subplot(7,8, i + 1)
#         plt.imshow(images[i].numpy())
#         plt.title(labels[i].numpy())
#         plt.axis("off")
#     plt.tight_layout()
#     plt.show()




train_data = train.map(lambda x, y: (preprocess_input(x), y))
test_data = test.map(lambda x, y: (preprocess_input(x), y))
val_data = val.map(lambda x, y: (preprocess_input(x), y))

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(22, activation="softmax")
])

model.summary()



model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
history = model.fit(train_data,validation_data=test_data,epochs=10)

model.save("model.h5")

loss, accuracy = model.evaluate(train_data, verbose=0)
print(accuracy * 100)
loss, accuracy = model.evaluate(test_data, verbose=0)
print(accuracy * 100)
loss, accuracy = model.evaluate(val_data, verbose=0)
print(accuracy * 100)
