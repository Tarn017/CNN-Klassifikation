# Image Classification with CNNs

**Vorgehen:**  
1. Ladet zunächst die beiden Skripte [ConvNeuralNetwork.py](https://github.com/Tarn017/CNN-Klassifikation/blob/main/ConvNeuralNetwork.py) und [main_CNN.py](https://github.com/Tarn017/CNN-Klassifikation/blob/main/main_CNN.py) herunter.  
2. Kopiert die beiden Skripte in euer Python-Projekt.  

**Training:**  
Öffnet nun das Skript `main_CNN.py` und passt eure Parameter entsprechend an. Hier eine kurze Erklärung, wie man die Funktion nutzt:  
`CNN(train_path, epochs, lr, conv_filters, fully_layers, resize, model_name, train_split, droprate, augmentation, dec_lr)`  
1. *train_path* entspricht dem Ordner auf den ihr das Netz trainieren wollt
2. *epochs* entspricht der Anzahl an Epochen, die das Netz trainiert werden soll
3. *lr* entspricht der Lernrate
4. *conv_filters* hat die Form [x,y,z,...], wobei x der Anzahl an convolutional Filtern in der 1. Schicht entsprich, y der Anzahl in der 2., usw. Die GEsamtzahl an Schichten wird somit ebenfalls hier bestimmt.
5. *fully_layers* hat die Form [x,y,z,...], wobei x der Anzahl an Neuronen in der 1. voll verbundenen Schicht entspricht, usw.
6. *resize* hat die form (höhe,breite), wobei beide Angaben in Anzahl Pixel gemacht werden. Quadratische Angaben werden bevorzugt
7. *model_name* gibt eurem Modell einen Namen. Beachte: Existiert bereits eines mit demselben Namen, wird das alte überschrieben.
8. *train_split* bspw. 0.8 bedeutet, dass 80% der Bilder fürs Training und 20% für Validation genutzt werden. Bei 1 werden alle Daten für das Training genutzt.
9. *droprate* (optional) bspw 0.2 bedeutet, dass jedes Neuron mit einer Wahrscheinlichkeit von 20% deaktiviert wird.
10. *augmentation* (optional) hat die Form [flip, rotate, brightness, contrast, saturation]. Jeder dieser Werte muss zwischen 0 und 1 liegen. *flip* ist die Wahrscheinlichkeit, dass ein Bild gespiegelt wird, *rotate* gibt die Stärke einer zufälligen Rotation an (1 für stark), *brightness, contrast, saturation* geben an wie stark Helligkeit, Kontrast und Sättigung maximal verändert werden.
11. *dec_lr* (optional) wirg genutzt, falls die Lernrate während des Trainigs abnehmen soll. Der Wert der für diesen Parameter angegeben wird, entspricht der Lernrate der letzten Epoche.

```python
from ConvNeuralNetwork import CNN

if __name__ == "__main__":
    CNN(
        train_path="tiere400",
        epochs=10,
        lr=0.005,
        conv_filters=[16, 32, 64, 128],
        fully_layers=[256],
        resize=(128, 128),
        model_name='peter',
        train_split=0.9,
        droprate=0.5,
        augmentation=[0.1,0.1,0.1,0.1,0.1],
        dec_lr=10e-5
    )
