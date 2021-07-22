from Model.model import get_model
if __name__ == '__main__':
    model = get_model(name="CBMA-PyResNet")
    model.summary()
    import sys
    sys.exit()