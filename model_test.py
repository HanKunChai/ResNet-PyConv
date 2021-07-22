from Model.model import get_model
if __name__ == '__main__':
    model = get_model(name="SE-PyResNet")
    model.summary()
    import sys
    sys.exit()