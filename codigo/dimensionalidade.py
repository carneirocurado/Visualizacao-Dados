# -*- coding: utf-8 -*-

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import cross_val_score

import numpy
#import datetime
import time
import matplotlib.pyplot as plt
from matplotlib import colors

from sklearn import manifold
from sklearn.metrics import euclidean_distances

nome = "mds"
arquivo_entrada = "./dados/ibge_municipios-completo.csv"
#arquivo_saida = "./saida/saida_" + nome + "_" + time.strftime("%Y%m%d-%Hh%Mm") + ".csv"
arq_log = "./log/log_" + nome + "_" + time.strftime("%Y%m%d-%Hh%Mm") + ".txt"
arq_fig = "./saida/fig_" + nome + "_" + time.strftime("%Y%m%d-%Hh%Mm")


with open(arq_log, 'w', buffering=1) as arq_log:
        
    str_cabecalho = ("Hora inicio: {0:s}\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    arq_log.write(str_cabecalho)
    print (str_cabecalho)

    j = 0
    tempo = time.time()
    
    # carrega os dados de latitude e longitude
    dados = numpy.genfromtxt(arquivo_entrada, delimiter=",", dtype='U50', skip_header=1, names=None, unpack=False, encoding='utf-8')
    
    header = numpy.genfromtxt(arquivo_entrada, delimiter=",", dtype='U50', skip_header=0, names=None, max_rows=1, unpack=False)
    
    lat_long = dados[:,4:6].astype(float)
    municipios = dados[:,0:1].astype(int)
    
    print("{0:s} - Chamando Manifold MDS".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    
    mds = manifold.MDS(n_components=1, metric=True, max_iter=1000, verbose=10, eps=1e-18, dissimilarity="euclidean", random_state=None, n_jobs=1, n_init=1)
    
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
        

    print("{0:s} - Criando Visualizacao".format(time.strftime("%Y-%m-%d %H:%M:%S")))   
    vmax = max(lista_completa[:,5].astype(float))
    vmin = min(lista_completa[:,5].astype(float))
    
#    norm_min_max = colors.Normalize(vmin=vmin, vmax=300)
#    log_norm_min_max = colors.SymLogNorm(linthresh=1, linscale=0.1, vmin=vmin, vmax=300)
    
#    plt.figure()   
#    cm = plt.cm.get_cmap('YlOrBr')
#    sc = plt.scatter(lista_completa[:,4].astype(int), lista_completa[:,1].astype(float), c=lista_completa[:,5].astype(float), norm=log_norm_min_max, cmap=cm)
#    plt.axis([min(lista_completa[:,4].astype(float)), max(lista_completa[:,4].astype(float)), min(lista_completa[:,1].astype(float)), max(lista_completa[:,1].astype(float))])
#    plt.xlabel('Semana Epidemiológica')
#    plt.ylabel('Municípios')
#    plt.colorbar(sc, extend='max')
#    plt.show()
    
    
    
    fig, ax = plt.subplots()
    bounds = numpy.array([0,5,10,15,20,25,50,75,100,150,200,250,300])
    bound_norm_min_max = colors.BoundaryNorm(boundaries=bounds, ncolors=256, clip=True)
    cm = plt.cm.get_cmap('YlOrBr')
#    cm = fig.cm.get_cmap('RdBu_r')
    sc = plt.scatter(lista_completa[:,4].astype(int), lista_completa[:,1].astype(float), c=lista_completa[:,5].astype(float), norm=bound_norm_min_max, cmap=cm)
    plt.axis([min(lista_completa[:,4].astype(float)), max(lista_completa[:,4].astype(float)), min(lista_completa[:,1].astype(float)), max(lista_completa[:,1].astype(float))])
    plt.xlabel('Semana Epidemiológica')
#    plt.ylabel('Municípios')
    
    # Precisamos desenhar o canvas para poder resgatar os labels originais
    fig.canvas.draw()
    # Resgatando o label atual plotados para o eixo y
    # Lembrando que esse eixo se refere à projeção da lat long em uma única dimensão - usando MDS
    y_labels = [item for item in ax.get_yticks()]
    # Resgatando, agora, da lista de municipios, os indices que correspondem ao y_label anterior
    y_new_idx = [(numpy.abs(lista_completa[:,1].astype(float) - item)).argmin() for item in y_labels]
    y_new_labels = [item[2] + ' - ' + item[3] for item in lista_completa[y_new_idx]]
    # Ajustando, agora, o label do eixo y para o nome dos municipios, pois e mais intuitivo...
    ax.set_yticklabels(y_new_labels)
    
    #plt.legend()
    plt.colorbar(sc, extend='max')
    
#    plt.savefig(arq_fig+".eps", dpi=150)
#    plt.savefig(arq_fig+".pdf", dpi=150)
    
    plt.show()

    str_rodape = ("\nHora termino: {0:s}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    arq_log.write(str_rodape)
    print (str_rodape)
