#===============================================================================
# Trabalho 2
#-------------------------------------------------------------------------------
# Autor: Eduarda Simonis Gavião
# UNICAMP
#===============================================================================
# Importando bibliotecas
import sys
import numpy as np
import cv2
import math
import statistics
import matplotlib.pyplot as plt
import seaborn as sns


BABOON_IMAGE =  'img1.pgm'
FIDUCIAL_IMAGE =  'img2.pgm'
MONARCH_IMAGE =  'img3.pgm'
PEPPERS_IMAGE =  'img4.pgm'
RETINA_IMAGE =  'img5.pgm'
SONNET_IMAGE =  'img6.pgm'
WEDGE_IMAGE= 'img7.pgm'
ARROZ_IMAGE= 'img8.png'

THRESHOLD_GLOBAL = 210


def global_limiar(img,threshold):
    img=np.where(img > threshold,255, img*0) # se a intensidade do pixel for maior que o parâmetro do Thresold
    #é denominado objeto, caso contrário é fundo
    return(img)

def otsu_limiar(img):
    pixels = img.shape[0] * img.shape[1] #verifica o número de pixels PRESENTES na imagem

    mean_weight = 1.0/pixels # verifica a média a ser aplicada
    his, bins = np.histogram(img, np.arange(0,257)) # obtem o histograma e o número de intervalos
##
    thresh = 1 ## passa um valor inicial de threshold
    final_value = 255 # passa um valor final 
    intensidade_array = np.arange(256) #vetor de intensidades
##
    for t in bins[1: 256]: # percorre de 0 a 256
    #somatorio das probabilidades
        q1 = np.sum(his[:t]) 
        q2 = np.sum(his[t:])
        Wb = q1 * mean_weight
        Wf = q2 * mean_weight
## Obtendo as médias das classes acima 
        mub = np.sum(intensidade_array[:t]*his[:t]) / q1
        muf = np.sum(intensidade_array[t:]*his[t:]) / q2
        
    ## valor da variancia
        value = Wb * Wf * (mub - muf) ** 2
        if value > final_value:
            thresh = t
            final_value = value
    final_img = img.copy()
    
    #verifica o limiar
    final_img[img > thresh] = 255 #verifica objeto
    final_img[img <= thresh] = 0 #verifica fundo
    return final_img  

def bernsen(img,n):
    (h, w) = img.shape[:2] ##tamanho da imagem   
    copia = img.copy()
    
    N=n//2
    # primeiro e segundo laço pra percorrer a imagem 
    for i in range(h):
        for j in range(w):
             #verifica se é a borda da imagem
            if (i - N) < 0 or (i + N + 1) >= h or (j - N) < 0 or (j + N + 1)  >= w:
                    copia[i][j] = img[i][j] #caso for mantem como a original
            else:
                #janela deslizante de altura n e largura n
                janela=img[i-N:i+N+1,j-N:j+N+1]

                minimum=np.min(janela)
                maximum=np.max(janela)
                thresh=(minimum+maximum)/2
                #verifica o limiar
                if(img[i,j]<=thresh): #verifica fundo 
                    copia[i,j]=0
                else:
                    copia[i,j]=255 #objeto
    return copia  

def niblack(img,n,k):
    (h, w) = img.shape[:2] ##tamanho da imagem   
    copia = img.copy()
    
    N=n//2
    
    # primeiro e segundo laço pra percorrer a imagem 
    for i in range(h):
        for j in range(w):
             #verifica se é a borda da imagem
            if (i - N) < 0 or (i + N + 1) >= h or (j - N) < 0 or (j + N + 1)  >= w:
                    copia[i][j] = img[i][j] #caso for mantem como a original
            else:
                #janela deslizante de altura n e largura n
                janela=img[i-N:i+N+1,j-N:j+N+1]
                media=np.average(janela)
                dp=np.std(janela)
                thresh=(media+k*dp)

                #verifica o limiar
                if(img[i,j]<=thresh):#fundo
                    copia[i,j]=0
                else:
                    copia[i,j]=255 #objeto
    return copia 

def sauvola(img,n,k,r):
    (h, w) = img.shape[:2] ##tamanho da imagem   
    copia = img.copy()
    
    N=n//2
    
    # primeiro e segundo laço pra percorrer a imagem 
    for i in range(h):
        for j in range(w):
             #verifica se é a borda da imagem
            if (i - N) < 0 or (i + N + 1) >= h or (j - N) < 0 or (j + N + 1)  >= w:
                    copia[i][j] = img[i][j] #caso for mantem como a original
            else:
                #janela deslizante de altura n e largura n
                janela=img[i-N:i+N+1,j-N:j+N+1]
                media=np.average(janela)
                dp=np.std(janela)
                thresh=media*(1+k*((dp/r)-1))

                #verifica o limiar
                if(img[i,j]<=thresh): #fundo
                    copia[i,j]=0
                else:
                    copia[i,j]=255 #objeto
    return copia  
def more(img,n,k,r,p,q):
    (h, w) = img.shape[:2] ##tamanho da imagem   
    copia = img.copy()
    
    N=n//2 # divide o tamanho da janela
    
    # primeiro e segundo laço pra percorrer a imagem 
    for i in range(h):
        for j in range(w):
             #verifica se é a borda da imagem
            if (i - N) < 0 or (i + N + 1) >= h or (j - N) < 0 or (j + N + 1)  >= w:
                    copia[i][j] = img[i][j] #caso for mantem como a original
            else:
                #janela deslizante de altura n e largura n
                janela=img[i-N:i+N+1,j-N:j+N+1]
                media=np.average(janela) #calcula média
                dp=np.std(janela) #calcula desvio padrão
                thresh=media*(1+(p*math.exp(-q*media))+k*((dp/r)-1)) #realiza o calculo do limiar
                
                #verifica o limiar
                
                if(img[i,j]<=thresh):
                    copia[i,j]=0 #fundo
                else:
                    copia[i,j]=255 #objeto
    return copia 
def contraste(img,n):
    (h, w) = img.shape[:2] ##tamanho da imagem   
    copia = img.copy()
    
    N=n//2

    # primeiro e segundo laço pra percorrer a imagem 
    for i in range(h):
        for j in range(w):
             #verifica se é a borda da imagem
            if (i - N) < 0 or (i + N + 1) >= h or (j - N) < 0 or (j + N + 1)  >= w:
                    copia[i][j] = img[i][j] #caso for mantem como a original
            else:
                #janela deslizante de altura n e largura n
                janela=img[i-N:i+N+1,j-N:j+N+1]

                #calcula limiar através da média
                thresh=np.average(janela)

                if(img[i,j]<=thresh): #valores abaixo da média são considerados perto do minimo, logo recebem valor 0, indicando fundo
                    copia[i,j]=0
                else:
                    copia[i,j]=255 #valores acima da média estão proximos dos máximos, sendo assim ganham valor 255, indicando objeto
    return copia 


def media(img,n,c):
    (h, w) = img.shape[:2] ##tamanho da imagem   
    copia = img.copy()
    
    N=n//2
    
    # primeiro e segundo laço pra percorrer a imagem 
    for i in range(h):
        for j in range(w):
             #verifica se é a borda da imagem
            if (i - N) < 0 or (i + N + 1) >= h or (j - N) < 0 or (j + N + 1)  >= w:
                    copia[i][j] = img[i][j] #caso for mantem como a original
            else:
                #janela deslizante de altura n e largura n
                janela=img[i-N:i+N+1,j-N:j+N+1]
                #calcula limiar através da média
                thresh=np.average(janela)-c

                if(img[i,j]<=thresh): #indica fundo
                    copia[i,j]=0
                else:
                    copia[i,j]=255 #indica objeto
    return copia 

def mediana(img,n):
    (h, w) = img.shape[:2] ##tamanho da imagem   
    copia = img.copy()
    
    N=n//2
    
    # primeiro e segundo laço pra percorrer a imagem 
    for i in range(h):
        for j in range(w):
             #verifica se é a borda da imagem
            if (i - N) < 0 or (i + N + 1) >= h or (j - N) < 0 or (j + N + 1)  >= w:
                    copia[i][j] = img[i][j] #caso for mantem como a original
            else:
                #janela deslizante de altura n e largura n
                janela=img[i-N:i+N+1,j-N:j+N+1]

                
                #calcula limiar através da mediana
                thresh=np.median(janela)

                if(img[i,j]<=thresh): # verifica o limiar
                    copia[i,j]=0 #verifica fundo
                else:
                    copia[i,j]=255 #verifica objeto
    return copia           
def main ():

    print('Escolha um dos métodos de limiarização:')
    print('Método Global - 1')
    print('Método de Otsu - 2')
    print('Método de Bernsen - 3')
    print('Método de Niblack - 4')
    print('Método de Sauvola e Pietaksinen - 5')
    print('Método de Phansalskar, More e Sabale - 6')
    print('Método do Contraste - 7')
    print('Método da Média - 8')
    print('Método da Mediana - 9')
    op = input("Indique a operação ")

    #tratamento de opções
    if op == "1":
        img = cv2.imread (ARROZ_IMAGE,cv2.IMREAD_GRAYSCALE)
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        
        img2=global_limiar(img,THRESHOLD_GLOBAL)
        cv2.imshow("Global",img2)
        cv2.imshow("Original",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        sns.histplot(img2, fill= True,cbar= False, legend=False) 
        plt.xlabel('Níveis de Cinza')
        plt.ylabel('Número de Pixels')
        plt.show()

        ##sns.distplot(img2) # distribuição normal
        ##plt.xlabel('Níveis de Cinza')
        ##plt.ylabel('Frequência')
        ##plt.show()
        

    elif op =="2":
        img = cv2.imread (BABOON_IMAGE,cv2.IMREAD_GRAYSCALE)
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        img2=otsu_limiar(img)
        cv2.imshow("OTSU",img2)
        cv2.imshow("Original",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.hist(img2) 

        #sns.distplot(img) # distribuição normal
        #plt.xlabel('Níveis de Cinza')
        #plt.ylabel('Frequência')
        #plt.show()
        
        plt.show()

    elif op == "3":
        img = cv2.imread (WEDGE_IMAGE,cv2.IMREAD_GRAYSCALE)
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()

        n = input("Digite o a dimensão da janela ")
        
        img2=bernsen(img,int(n))
        cv2.imshow("Bernsen",img2)
        cv2.imshow("Original",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.hist(img2) 
        plt.show()

    elif op =="4":
        img = cv2.imread (SONNET_IMAGE,cv2.IMREAD_GRAYSCALE)
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()

        n = input("Digite o a dimensão da janela ") 
        k = input("Digite o valor de k para ajustar a fração de borda ") 
        img2 = niblack(img,int(n),float(k))
        cv2.imshow("Niblack",img2)
        cv2.imshow("Original",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.hist(img2) 
        plt.show()

    elif op == "5":
        img = cv2.imread (SONNET_IMAGE,cv2.IMREAD_GRAYSCALE)
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        n = input("Digite o a dimensão da janela ") 
        k = input("Digite o valor de k para ajustar a fração de borda ") 
        r = input("Digite o valor de R ") 
        img2=sauvola(img,int(n),float(k),float(r))
        cv2.imshow("Sauvola",img2)
        cv2.imshow("Original",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.hist(img2) 
        plt.show()

    elif op =="6":
        img = cv2.imread (WEDGE_IMAGE,cv2.IMREAD_GRAYSCALE)
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        
        n = input("Digite o a dimensão da janela ") 
        k = input("Digite o valor de k para ajustar a fração de borda ") 
        r = input("Digite o valor de R ") 
        p = input("Digite o valor de p ") 
        q = input("Digite o valor de q ") 

        img2= more(img,int(n),float(k),float(r),float(p),float(q))
        cv2.imshow("Sauvola",img2)
        cv2.imshow("Original",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.hist(img2) 
        plt.show()

    elif op =="7":
        img = cv2.imread (BABOON_IMAGE,cv2.IMREAD_GRAYSCALE)
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        cv2.imshow("Contraste",contraste(img,int(n)))
        cv2.imshow("Original",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif op =="8":
        img = cv2.imread (PEPPERS_IMAGE,cv2.IMREAD_GRAYSCALE) 
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        n = input("Digite o a dimensão da janela ")
        c = input("Digite o valor da constante de ajuste ")
        img2= media(img,int(n), int(c))
        cv2.imshow("Media",img2)
        cv2.imshow("Original",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.hist(img2) 
        plt.show()

    elif op =="9":
        img = cv2.imread (MONARCH_IMAGE,cv2.IMREAD_GRAYSCALE)
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        n = input("Digite o a dimensão da janela ")    
        img2= mediana(img,int(n))
        cv2.imshow("Mediana", img2)
        cv2.imshow("Original",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.hist(img2) 
        plt.show()
    else: 
        print('Opção inválida')


if __name__ == '__main__':
    main()

