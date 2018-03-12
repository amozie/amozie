from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.utils import plot_model

model_name = 'vgg16'
model = VGG16(weights='imagenet', include_top=True)
model = VGG16(weights='imagenet', include_top=False)

model_name = 'googleV3'
model = InceptionV3(weights='imagenet', include_top=True)
model = InceptionV3(weights='imagenet', include_top=False)

model_name = 'resnet50'
model = ResNet50(weights='imagenet', include_top=True)
model = ResNet50(weights='imagenet', include_top=False)

plot_model(model, to_file=model_name + '_top.png', show_shapes=True)
plot_model(model, to_file=model_name + '_notop.png', show_shapes=True)
