from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.utils import plot_model

model = VGG16(weights='imagenet', include_top=True)
model = VGG16(weights='imagenet', include_top=False)

model = InceptionV3(weights='imagenet', include_top=True)
model = InceptionV3(weights='imagenet', include_top=False)

model = ResNet50(weights='imagenet', include_top=True)
model = ResNet50(weights='imagenet', include_top=False)

plot_model(model, to_file='model_top.png', show_shapes=True)
plot_model(model, to_file='model_notop.png', show_shapes=True)