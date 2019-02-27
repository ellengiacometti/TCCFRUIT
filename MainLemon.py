import argparse
import TrataImagem as TI
import DataAugmentation as DA

if __name__ == '__main__':

    """PARÂMETROS"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--path", required=True, help="path to the input image")
    args = vars(ap.parse_args())
    imageDir_val= args["path"]
    # Atributos = []
    # Atributos.append(TI.TrataImagem(imageDir_val, visual=0, verbose=0))

    origem = '/home/ellengiacometti/PycharmProjects/TCCFRUIT/OR_DA'
    destino = '/home/ellengiacometti/PycharmProjects/TCCFRUIT/DES_DA'
    DA.DataAugmentation(srcDirOr=origem,srcDirEnd=destino,type=0,angle=90,size=(500,500),dB=0.1)