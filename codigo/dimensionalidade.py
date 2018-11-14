# -*- coding: utf-8 -*-

import numpy
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
from matplotlib.patches import Rectangle

from sklearn import manifold
from sklearn.metrics import euclidean_distances

nome = "dengue-temporal"
arquivo_entrada = "./dados/ibge_municipios-completo.csv"
arq_log = "./log/log_" + nome + "_" + time.strftime("%Y%m%d-%Hh%Mm") + ".txt"
arq_fig = "./saida/" + nome + "_" + time.strftime("%Y%m%d-%Hh%Mm")

def label(xy, text):
    y = xy[1] - 0.15  # shift y-value for label so that it's below the artist
    plt.text(xy[0], y, text, ha="center", family='sans-serif', size='x-small')

def draw_rectangle (ax,x1,y1,x2,y2,label):
    rect_subfig_a = Rectangle((x1,y1),x2-x1,y2-y1, fill=False, linestyle='--')   
    ax.add_patch(rect_subfig_a)            
    ax.text(x1+(x2-x1)/2,y1-2, label, ha="center", family='sans-serif', size='x-small')


def cria_imagem (lista, x_min = None, x_max = None, y_min = None, y_max = None, x_nro_locators = None, y_nro_locators = None):

    imagem_completa = False
    if ( (x_min == None) and (x_max == None) and (y_min == None) and (y_max == None) ):
        imagem_completa = True
    
    if ( x_min == None ):
        x_min = min(lista[:,4].astype(int))
    if ( x_max == None ):
        x_max = max(lista[:,4].astype(int))
    if ( y_min == None ):
        y_min = min(lista[:,1].astype(float))
    if ( y_max == None ):
        y_max = max(lista[:,1].astype(float))
                
    fig, ax = plt.subplots()
    bounds = numpy.array([0,5,10,15,20,25,50,75,100,150,200,250,300])
    bound_norm_min_max = colors.BoundaryNorm(boundaries=bounds, ncolors=256, clip=True)
    cm = plt.cm.get_cmap('YlOrBr')
    
    lista_aux = lista[numpy.where(numpy.logical_and(lista[:,4].astype(int) >= x_min, lista[:,4].astype(int) <= x_max))]
    lista_filtrada = lista_aux[numpy.where(numpy.logical_and(lista_aux[:,1].astype(float) >= y_min, lista_aux[:,1].astype(float) <= y_max))]
    
    sc = plt.scatter(lista_filtrada[:,4].astype(int), lista_filtrada[:,1].astype(float), c=lista_filtrada[:,5].astype(float), marker=',', norm=bound_norm_min_max, cmap=cm)
    plt.axis([x_min, x_max, y_min, y_max])

    if ( x_nro_locators == None ):
        ax.xaxis.set_major_locator(mticker.AutoLocator())
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(x_nro_locators))       
    if ( y_nro_locators == None ):
        ax.yaxis.set_major_locator(mticker.AutoLocator())
    else:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(y_nro_locators))
        
    if ( imagem_completa ):
        draw_rectangle(ax,x1=90,y1=0,x2=120,y2=10,label='')

    plt.xlabel('Semana Epidemiológica', fontsize='x-small')
#    plt.ylabel('Municípios')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.3, right=0.9)
    
    ax.tick_params(axis='both', which='minor', length=6, width=2, labelsize='small', direction='out', bottom=True, left=True)
    
    # Precisamos desenhar o canvas para poder resgatar os labels originais
    fig.canvas.draw()
    # Resgatando o label atual plotados para o eixo y
    # Lembrando que esse eixo se refere a projecao da lat long em uma unica dimensao - usando MDS
    y_labels = [item for item in ax.get_yticks()]
    # Resgatando, agora, a UF e nome dos municipios para substituir no label do eixo y
    y_new_idx = [(numpy.abs(lista_filtrada[:,1].astype(float) - item)).argmin() for item in y_labels]
    y_new_labels = [item[2] + ' - ' + item[3] for item in lista_filtrada[y_new_idx]]
    # Ajustando o label do eixo y para o nome dos municipios, pois e mais intuitivo...
    ax.set_yticklabels(y_new_labels)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('x-small')
        
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('x-small')
    
    #plt.legend()
    cb = plt.colorbar(sc, extend='max')
    
    cb.set_label ('Taxa de Incidência por 100 mil hab.', fontsize='x-small')
    
    plt.savefig(arq_fig+"_x-"+str(x_min)+"-"+str(x_max)+"_y-"+str(y_min)+"-"+str(y_max)+"_destaques-"+str(imagem_completa)+".png", dpi=300)
    
    plt.show()


with open(arq_log, 'w', buffering=1) as arq_log:
        
    str_cabecalho = ("Hora inicio: {0:s}\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    arq_log.write(str_cabecalho)
    print (str_cabecalho)

    j = 0
    tempo = time.time()   
    # carrega os dados de latitude e longitude
    # Versao do genfromtxt para Mac
    dados = numpy.genfromtxt(arquivo_entrada, delimiter=",", converters={3:lambda x: x.decode()}, dtype='U50', skip_header=1, names=None, unpack=False)  
#   Versao do genfromtxt para Windows
#    dados = numpy.genfromtxt(arquivo_entrada, delimiter=",", dtype='U50', skip_header=1, names=None, unpack=False, encoding='utf-8')
    
    header = numpy.genfromtxt(arquivo_entrada, delimiter=",", dtype='U50', skip_header=0, names=None, max_rows=1, unpack=False)
    
    lat_long = dados[:,4:6].astype(float)
    municipios = dados[:,0:1].astype(int)
    
    print("{0:s} - Chamando Manifold MDS".format(time.strftime("%Y-%m-%d %H:%M:%S")))    
    mds = manifold.MDS(n_components=1, metric=True, max_iter=1000, verbose=10, eps=1e-22, dissimilarity="euclidean", random_state=None, n_jobs=1, n_init=1)    
    print("{0:s} - Chamando fit transform".format(time.strftime("%Y-%m-%d %H:%M:%S")))    
    pos_final = mds.fit_transform(lat_long)    
    print("{0:s} - Calculando distancia euclidiana da latitude e longitude".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    
    euclidian_distance = euclidean_distances(lat_long)   
    euclidian_distance2 = euclidean_distances(pos_final)
    
    print("{0:s} - Calculando o coeficiente das distancias".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    coeficiente = numpy.divide(euclidian_distance[0,1:], euclidian_distance2[0,1:])
    
    municipios = numpy.hstack((municipios,pos_final))
    municipios = numpy.hstack((municipios,dados[:,2:4]))
    municipios = numpy.hstack((municipios,dados[:,7:]))   
    municipios2 = municipios[municipios[:,1].astype(float).argsort()]
     
    print("{0:s} - Construindo a lista dos resultados".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    semana = 1   
    lista_completa = numpy.vstack((municipios2[:,0], municipios2[:,1], municipios2[:,2], municipios2[:,3],numpy.full(len(municipios2[:,4]),semana), municipios2[:,semana+3])).T
    for x in municipios2[:,5:].T.astype(float):
        semana += 1        
        aux_lista = numpy.vstack((municipios2[:,0], municipios2[:,1], municipios2[:,2], municipios2[:,3],numpy.full(len(x),semana), municipios2[:,semana+3])).T
        lista_completa = numpy.vstack((lista_completa,aux_lista))
        
    print("{0:s} - Criando Visualizacao completa, sem destaques".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    x_minimo = min(lista_completa[:,4].astype(int))
    x_maximo = max(lista_completa[:,4].astype(int))
    y_minimo = min(lista_completa[:,1].astype(float))
    y_maximo = max(lista_completa[:,1].astype(float))
    cria_imagem(lista_completa, x_min=x_minimo, x_max=x_maximo, y_min=y_minimo, y_max=y_maximo, x_nro_locators=15, y_nro_locators=16)
    print("{0:s} - Criando Visualizacao completa, com destaques".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    cria_imagem(lista_completa, x_nro_locators=15, y_nro_locators=16)
    
    print("{0:s} - Criando Visualizacao pequena 1".format(time.strftime("%Y-%m-%d %H:%M:%S")))   
    cria_imagem(lista_completa, x_min=90, x_max=120, y_min=0, y_max=10, y_nro_locators=16)

    str_rodape = ("\nHora termino: {0:s}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    arq_log.write(str_rodape)
    print (str_rodape)
