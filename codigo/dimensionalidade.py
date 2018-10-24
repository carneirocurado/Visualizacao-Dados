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

#import pandas as pd
#from mlxtend.preprocessing import TransactionEncoder
#from mlxtend.frequent_patterns import apriori
#from mlxtend.frequent_patterns import association_rules
from sklearn import manifold
from sklearn.metrics import euclidean_distances

#nome = "apriori-reduzido-rs"
#nome = "apriori-entrada-label"
#nome = "apriori_rs-maior-500"
#nome = "apriori_rs-completo"
#nome = "apriori_completo"
#nome = "apriori_rs-ae_completo"
#nome = "apriori_estab-saude"
nome = "mds"
#arquivo_entrada = "./dados/ibge_municipios_2010.csv4"
arquivo_entrada = "./dados/ibge_municipios-completo.csv"
#arquivo_saida = "./saida/saida_" + nome + "_" + time.strftime("%Y%m%d-%Hh%Mm") + ".csv"
arq_log = "./log/log_" + nome + "_" + time.strftime("%Y%m%d-%Hh%Mm") + ".txt"


 

#def organiza_antecedentes(antecedentes, lista_antecedentes):
#    for item in antecedentes:
#        lista_antecedentes.append(item)


with open(arq_log, 'w', buffering=1) as arq_log:
        
    str_cabecalho = ("Hora inicio: {0:s}\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    arq_log.write(str_cabecalho)
    print (str_cabecalho)

    j = 0
    tempo = time.time()
    
    # carrega os dados de latitude e longitude
#    dados = numpy.genfromtxt(arquivo_entrada, delimiter=",", converters={3:lambda x: x.decode()}, dtype='U50', skip_header=1, names=None, unpack=False)
    dados = numpy.genfromtxt(arquivo_entrada, delimiter=",", dtype='U50', skip_header=1, names=None, unpack=False, encoding='utf-8')
    
    header = numpy.genfromtxt(arquivo_entrada, delimiter=",", dtype='U50', skip_header=0, names=None, max_rows=1, unpack=False)
    
#    lat_long2 = float(lat_long.copy())
    
#    lat_long2 = lat_long[:,1:3].copy()
    lat_long = dados[:,4:6].astype(float)
    municipios = dados[:,0:1].astype(int)
#    lat_long = arq_original.astype(float)
    
#    lat_long4 = lat_long2.astype(float)
    
#    lat_long4 = list(map(lambda x: float(x[0],:), lat_long2))
    
#    lat_long3 = lat_long2[1].tolist()
    
#    lat_long3[0] = list(map(lambda x: float(x[0]), lat_long2))
    
    print("{0:s} - Chamando Manifold MDS".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    
    mds = manifold.MDS(n_components=1, metric=True, max_iter=1000, verbose=10, eps=1e-18, dissimilarity="euclidean", random_state=None, n_jobs=1, n_init=1)
    
    print("{0:s} - Chamando fit transform".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    
    pos_final = mds.fit_transform(lat_long)
    
    print("{0:s} - Calculando distancia euclidiana da latitude e longitude".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    
    euclidian_distance = euclidean_distances(lat_long)
    
    euclidian_distance2 = euclidean_distances(pos_final)
    
#    teste = numpy.core.records.fromarrays ([municipios, pos_final, lat_long[:,0], lat_long[:,1]], names='munid,latlong,latitude,longitude')
    
    print("{0:s} - Calculando o coeficiente das distancias".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    coeficiente = numpy.divide(euclidian_distance[0,1:], euclidian_distance2[0,1:])
    
    municipios = numpy.hstack((municipios,pos_final))
    municipios = numpy.hstack((municipios,dados[:,2:4]))
    municipios = numpy.hstack((municipios,dados[:,7:]))
    
    municipios2 = municipios[municipios[:,1].astype(float).argsort()]
    
    
#    dados = numpy.hstack((dados,pos_final))
    
#    final = numpy.vstack((municipios.T,lat_long.T))
#    pos_final = numpy.hstack((lat_long,npos))

    
 
    semana = 0
    lista_1 = list()
    lista_2 = list()
    lista_3 = list()
    lista_4 = list()
    lista_completa2 = list()
#    lista = municipios[:,1]
    for x in municipios[:,4:].T.astype(float):
        semana += 1
        
        print("{0:s} - Construindo a lista dos resultados - semana {1:d}".format(time.strftime("%Y-%m-%d %H:%M:%S"), semana))
        
        aux_lista = numpy.vstack((municipios[:,0], municipios[:,1], municipios[:,2], municipios[:,3],numpy.full(len(x),semana), municipios[:,semana+3])).T
        if len(lista_completa2) == 0:
            lista_completa2 = aux_lista
        else:         
            lista_completa2 = numpy.vstack((lista_completa2,aux_lista))
        
        teste_baixo = numpy.where(x <= 25)[0]
        aux_baixo = numpy.vstack((numpy.full(len(teste_baixo),semana),numpy.full(len(teste_baixo),'#fef0d9'),numpy.full(len(teste_baixo),'Baixo'))).T
        if len(lista_1) == 0:
            lista_1 = numpy.hstack((municipios[teste_baixo,0:2].astype(float),aux_baixo))
        else:
            lista_1 = numpy.vstack((lista_1,numpy.hstack((municipios[teste_baixo,0:2].astype(float),aux_baixo))))
            
            
        teste_medio = numpy.where((x > 25) & (x < 75))[0]
        aux_medio = numpy.vstack((numpy.full(len(teste_medio),semana),numpy.full(len(teste_medio),'#fdcc8a'),numpy.full(len(teste_medio),'Medio'))).T
        if len(lista_2) == 0:
            lista_2 = numpy.hstack((municipios[teste_medio,0:2].astype(float),aux_medio))
        else:
            lista_2 = numpy.vstack((lista_2,numpy.hstack((municipios[teste_medio,0:2].astype(float),aux_medio))))
        
        
        teste_alto = numpy.where((x >= 75) & (x < 300))[0]
        aux_alto = numpy.vstack((numpy.full(len(teste_alto),semana),numpy.full(len(teste_alto),'#fc8d59'),numpy.full(len(teste_alto),'Alto'))).T
        if len(lista_3) == 0:
            lista_3 = numpy.hstack((municipios[teste_alto,0:2].astype(float),aux_alto))
        else:
            lista_3 = numpy.vstack((lista_3,numpy.hstack((municipios[teste_alto,0:2].astype(float),aux_alto))))
        
        
        teste_muito_alto = numpy.where(x >= 300)[0]
        aux_muito_alto = numpy.vstack((numpy.full(len(teste_muito_alto),semana),numpy.full(len(teste_muito_alto),'#d7301f'),numpy.full(len(teste_muito_alto),'Muito Alto'))).T
        if len(lista_4) == 0:
            lista_4 = numpy.hstack((municipios[teste_muito_alto,0:2].astype(float),aux_muito_alto))
        else:
            lista_4 = numpy.vstack((lista_4,numpy.hstack((municipios[teste_muito_alto,0:2].astype(float),aux_muito_alto))))
    
    lista_completa = lista_1.copy()
    lista_completa = numpy.vstack((lista_completa, lista_2))
    lista_completa = numpy.vstack((lista_completa, lista_3))
    lista_completa = numpy.vstack((lista_completa, lista_4))
    lista_completa = lista_completa[lista_completa[:,1].astype(float).argsort()]
        
    plt.figure()
#    plt.scatter(lista[:,2].astype(int), lista[:,1].astype(float), c=lista[:,3])
#    plt.scatter(lista_1[:,2].astype(int), lista_1[:,1].astype(float), c=lista_1[:,3], label=lista_1[:,4])
#    plt.scatter(lista_2[:,2].astype(int), lista_2[:,1].astype(float), c=lista_2[:,3], label=lista_2[:,4])
#    plt.scatter(lista_3[:,2].astype(int), lista_3[:,1].astype(float), c=lista_3[:,3], label=lista_3[:,4])
#    plt.scatter(lista_4[:,2].astype(int), lista_4[:,1].astype(float), c=lista_4[:,3], label=lista_4[:,4])
    plt.scatter(lista_1[:,2].astype(int), lista_1[:,1].astype(float), c='#fef0d9', label='Baixo')
    plt.scatter(lista_2[:,2].astype(int), lista_2[:,1].astype(float), c='#fdcc8a', label='Medio')
    plt.scatter(lista_3[:,2].astype(int), lista_3[:,1].astype(float), c='#fc8d59', label='Alto')
    plt.scatter(lista_4[:,2].astype(int), lista_4[:,1].astype(float), c='#d7301f', label='Muito Alto')
    plt.axis([1, 156, lista_completa[0,1].astype(float), lista_completa[len(lista_completa)-1,1].astype(float)])
    plt.xlabel('Semana Epidemiológica')
    plt.ylabel('Municípios')
    plt.colorbar()
#    plt.legend()
    plt.show()
    
    vmax = max(lista_completa2[:,5].astype(float))
    vmin = min(lista_completa2[:,5].astype(float))
    
    plt.figure()
#    plt.scatter(lista_completa[:,2].astype(int), lista_completa[:,1].astype(float), c=lista_completa[:,3], label=lista_completa[:,4])
    plt.scatter(lista_completa[:,2].astype(int), lista_completa[:,1].astype(float), c=lista_completa[:,3])
    plt.axis([1, 156, lista_completa[0,1].astype(float), lista_completa[len(lista_completa)-1,1].astype(float)])
    plt.xlabel('Semana Epidemiológica2')
    plt.ylabel('Municípios2')
    #plt.legend()
#    plt.colorbar()
    plt.show()
    
 
        
#        print(x)
    
    
    # Inicia a lista de antecedentes
#    lista_antecedentes = list()
    
    str_rodape = ("\nHora termino: {0:s}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    arq_log.write(str_rodape)
    print (str_rodape)
