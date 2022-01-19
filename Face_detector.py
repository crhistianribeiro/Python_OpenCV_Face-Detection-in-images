#----------------------------------------------------------------------------------------------------------------------------------------------------
#Detectar rostos nas imagens
#----------------------------------------------------------------------------------------------------------------------------------------------------
import cv2

#Importar face cascade detector
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

#Criar imagem
img = cv2.imread("./fla.jpg")

#Transformar imagem de BRG para grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Cascade multiscale detector
faces = face_cascade.detectMultiScale(gray, 1.08,5)

#Criar for para faces detectadas na imagem e desenhar retângulo
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),8)
    img_resize = cv2.resize(img,(1080,720))

#Criar janela para apresentação do resultado
cv2.imshow("Time", img_resize)

#Salvar a imagem com detecção
cv2.imwrite('Detected.jpg',img_resize)

#Manter a imagem aberta indefinidamente até apertar uma tecla
cv2.waitKey(0)

#Fechar a janela
cv2.destroyWindow('Time')