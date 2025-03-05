from tensorflow.keras.applications import ResNet50,VGG16, MobileNetV2

class DlModel:
    def __init__(self):
        pass

    def create_dl_model(self, config: str, num_classes, input_shape):
        model = None

        if "resnet50" in config:
            model = ResNet50(weights=None, 
                             input_shape=input_shape,
                             classes=num_classes
                             )

        elif "vgg16" in config:
            model = VGG16(weights=None, 
                             input_shape=input_shape,
                             classes=num_classes
                             )

        elif "mobilenetv2" in config:
            model = MobileNetV2(weights=None, 
                             input_shape=input_shape,
                             classes=num_classes
                             )

            
        return model