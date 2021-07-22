from Model.Original_ResNet50 import mkResNet50
from ArticleModel.DilatedPyConv import DilatedPyConvModel
from Model.ResNetPyConv import PyResNet


def get_model(name: str = None, classes_num: int = 6):
    model_names = ["ResNet50", "SE-ResNet", "CBAM", "PyResNet", "DilatedPyConvResNet","SE-PyResNet","CBMA-PyResNet"]
    assert name in model_names , 'model name must be one of ResNet50,SE-ResNet, CBAM PyResNet, ' \
                                 'DilatedPyConvResNet,SE-PyResNet,CBMA-PyResNet '
    if name == "ResNet50":
        return mkResNet50(classes_num=classes_num)
    elif name == "SE-ResNet":
        return mkResNet50(classes_num=classes_num, senet=True)
    elif name == "CBAM":
        return mkResNet50(classes_num=classes_num, CBAM=True)
    elif name == "PyResNet":
        return PyResNet(classes_num=classes_num)
    elif name == "DilatedPyConvResNet":
        return DilatedPyConvModel(classes_num=classes_num)
    elif name == "SE-PyResNet":
        return PyResNet(classes_num=classes_num,senet=True)
    elif name == "CBMA-PyResNet":
        return PyResNet(classes_num=classes_num,cbam=True)
    else:
        raise Exception('No')

