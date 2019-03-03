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
    TI.TrataImagem('/home/ellengiacometti/PycharmProjects/TCCFRUIT/TEST/RS/0007_RSG.jpg',1,0)
    #DA.DataAugmentation(srcDirOr=origem,srcDirEnd=destino,initialid= 2632,type=3,angle=0,size=(400,400),dB=0.01,direction='Vertical')