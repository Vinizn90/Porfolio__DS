import cv2
import numpy as np

def get_user_id():
    while True:
        try:
            user_id = int(input('Digite seu identificador (um número inteiro): '))
            return user_id
        except ValueError:
            print("Digite um número inteiro válido.")

id = get_user_id()
classifier = cv2.CascadeClassifier('C:/Users/viniz/Downloads/Python-para-DataScience-e-Machine-Learning-Apostias/Python-Data-Science-and-Machine-Learning-Bootcamp/Reconhecimento Facial com OpenCV/haarcascade_frontalface_default.xml')
eyeClassifier = cv2.CascadeClassifier('C:/Users/viniz/Downloads/Python-para-DataScience-e-Machine-Learning-Apostias/Python-Data-Science-and-Machine-Learning-Bootcamp/Reconhecimento Facial com OpenCV/haarcascade_eye.xml')
amostra = 1
numeroAmostras = 25
largura, altura = 220, 220

print('Capturando as faces...')

camera = cv2.VideoCapture(0)

while True:
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    detectedFaces = classifier.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100,100))

    for (x, y, l, a) in detectedFaces:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = imagem[x:x + a, y:y + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        detectedEyes = eyeClassifier.detectMultiScale(regiaoCinzaOlho)
        for (ox, oy, ol, oa) in detectedEyes:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(imagemCinza) > 110:
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                    cv2.imwrite(f'C:/Users/viniz/Downloads/Python-para-DataScience-e-Machine-Learning-Apostias/Python-Data-Science-and-Machine-Learning-Bootcamp/Reconhecimento Facial com OpenCV/fotos/pessoa.{str(id)}.{str(amostra)}.jpg', imagemFace)
                    print(f'[foto {str(amostra)} capturada com sucesso]')
                    amostra += 1


    cv2.imshow("Face", imagem)

    
    if amostra >= numeroAmostras + 1:
        break

print('Faces capturadas com sucesso')
camera.release()
cv2.destroyAllWindows()