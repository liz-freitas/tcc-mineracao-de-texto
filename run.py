#Importação das bibliotecas
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import random
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk import RegexpTokenizer, stem
from nltk.corpus import stopwords
import unicodedata
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
# import tkinter
import matplotlib.pyplot as plt
import numpy as np


#Inicialização para o pré-processamento
tokenizer = RegexpTokenizer(r'\w+')

stopword_set = set(stopwords.words('portuguese'))

stemmer = stem.RSLPStemmer()

#Função que realiza o tratamento do texto diminindo a dimensionalidade do corpus
def tratar_texto(arq):
    new_data = []
    
    # Iterando linhas/frases do documento 
    for frase in arq:
        # Remove os espaços, tabulações e quebras de linha
        frase = frase.strip()

        # Verifica se linha está vazia, caso não, realiza o proocessamento do texto
        if frase != '':
            # print(' Frase:  ' + str(frase))

            # Ignora as acentuações, coloca todas as palavras em caixa baixa
            new_str = unicodedata.normalize('NFD', frase.lower() ).encode('ASCII', 'ignore').decode('UTF-8')          
            # print('*Frase normalizada:  ' + str(new_str))
            # print('\n \n\n') 

            #Transforma uma frase em token(vetor).
            dlist = tokenizer.tokenize(new_str)
            # print('*Tokens:  ' + str(dlist))
            # print('\n \n\n')  

            #Recebe o vetor dlis, e remove as sopwords do mesmo   
            dlist = list(set(dlist).difference(stopword_set))
            # print('*Frase sem Stopwords:  ' + str(dlist))
            # print('\n \n\n') 
            
            #Pega a lista do vetor e para cada palavra/posição do vetor "s" reduz ao radical da palavra. 
            for s in range(len(dlist)):
                dlist[s] = stemmer.stem(dlist[s])
            
            # print('*Stemmed Tokens:  ' + str(dlist))
            new_data.append(dlist)
    return new_data

#Função que divide as bases de textos em treinamento e validação
def dividir_base(dados):
    quantidade_total = len(dados)
    percentual_treino = 0.75
    treino = []
    validacao = []

    for indice in range(0, quantidade_total):
        if indice < quantidade_total * percentual_treino:
            treino.append(dados[indice])
        else:
            validacao.append(dados[indice])

    return treino, validacao

#Função que treina o modelo Doc2Vec
def treinar_modelo(tagged_data):
    max_epochs = 1000 #Número de iterações sobre o corpus.
    vec_size = 500 # Dimensionalidade dos vetores de recursos.
    alpha = 0.01 #A taxa de aprendizado inicial.

    model = Doc2Vec(vector_size=vec_size, #Dimensionalidade dos vetores de recursos.
                    alpha=alpha, 
                    min_alpha=0.00025, # A taxa de aprendizado cairá linearmente para min_alpha à medida que o treinamento progride.
                    min_count=1,#Ignora todas as palavras com frequência total menor que o valor parametrizado.
                    window = 20,
                    dm =1) #Define o algoritmo de treinamento. Se dm = 1 , 'memória distribuída' (PV-DM) é usada. Caso contrário, o pacote distribuído de palavras (PV-DBOW) é empregado.

    # Constroi o vocabulário
    model.build_vocab(tagged_data)

    # Faz o treinamento da rede neural
    for epoch in range(max_epochs):
        # print('iteration {0}'.format(epoch))
        #Esta é a rotina de treinamento, em que são passadas as informações.
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # Diminui a taxa de aprendizado
        model.alpha -= 0.0002
        # Conserta a taxa a taxa de aprendizado, sem decadância
        model.min_alpha = model.alpha
    return model

#Função que converte o conjunto de frases em vetor caracteristica. 
def gerar_vetor(model, wordVector):
    return model.infer_vector(wordVector, steps=1000, alpha=0.01)

# ---- Função que gera os classificadores ----
def gerar_classificador_reg_logistica(model, suicidas, nao_suicidas):
    # Mesclando tabelas de validação
    array =  [[gerar_vetor(model, suicida), 1] for suicida in suicidas]
    array += [[gerar_vetor(model, nao_suicida), 0] for nao_suicida in nao_suicidas]

    # Randomizando array
    random.shuffle(array)

    # Separando array de teste das labels
    train_array = []
    train_labels = []

    for index in range(len(array)):
        train_array.append(array[index][0])
        train_labels.append(array[index][1])
        
    # Treinando regressão linear
    classificador = LogisticRegression()
    classificador.fit(train_array, train_labels)

    return classificador

#Função que gera os classificadores (Random Forest)
def gerar_classificador_random_forest(model, suicidas, nao_suicidas):
    # Mesclando tabelas de validação
    array =  [[gerar_vetor(model, suicida), 1] for suicida in suicidas]
    array += [[gerar_vetor(model, nao_suicida), 0] for nao_suicida in nao_suicidas]

    # Randomizando array
    random.shuffle(array)

    # Separando array de teste das labels
    train_array = []
    train_labels = []

    for index in range(len(array)):
        train_array.append(array[index][0])
        train_labels.append(array[index][1])
        
    # Treinando regressão linear
    classificador = RandomForestClassifier()
    classificador.fit(train_array, train_labels)

    return classificador


#Função que gera os classificadores ( Naive Bayes¶)
def gerar_classificador_naive_bayes(model, suicidas, nao_suicidas):
    # Mesclando tabelas de validação
    array =  [[gerar_vetor(model, suicida), 1] for suicida in suicidas]
    array += [[gerar_vetor(model, nao_suicida), 0] for nao_suicida in nao_suicidas]

    # Randomizando array
    random.shuffle(array)

    # Separando array de teste das labels
    train_array = []
    train_labels = []

    for index in range(len(array)):
        train_array.append(array[index][0])
        train_labels.append(array[index][1])
        
    # Treinando Naive Bayes
    classificador = GaussianNB()
    classificador.fit(train_array, train_labels)

    return classificador

#Função que gera os classificadores ( Gradient Boosting)
def gerar_classificador_gradient_boosting(model, suicidas, nao_suicidas):
    # Mesclando tabelas de validação
    array =  [[gerar_vetor(model, suicida), 1] for suicida in suicidas]
    array += [[gerar_vetor(model, nao_suicida), 0] for nao_suicida in nao_suicidas]

    # Randomizando array
    random.shuffle(array)

    # Separando array de teste das labels
    train_array = []
    train_labels = []

    for index in range(len(array)):
        train_array.append(array[index][0])
        train_labels.append(array[index][1])

    # Treinando Gradient Boosting
    classificador = GradientBoostingClassifier()
    classificador.fit(train_array, train_labels)

    return classificador

#Função que mede a precisão - REMOVER
def aferir_precisao(model, classifier, suicidas, nao_suicidas):
    total = len(suicidas) + len(nao_suicidas)
    acertos = 0
    qts_suicidas = 0
    qts_nao_suicidas = 0

    for frase in suicidas:
        predicao = classifier.predict([gerar_vetor(model, frase)])
        if predicao[0] == 1:
            acertos += 1
            qts_suicidas += 1

    for frase in nao_suicidas:
        predicao = classifier.predict([gerar_vetor(model, frase)])
        if predicao[0] == 0:
            acertos += 1
            qts_nao_suicidas += 1

    # print("Quantidade de frases classificadas\n\tsuicidas: {}/{}\n\tnão suicidas: {}/{}\n".format(qts_suicidas, len(suicidas), qts_nao_suicidas, len(nao_suicidas)))

    media = acertos * 100.0 / total
    return media

#Função que calcula a curva ROC
def calc_roc( model, classifier, suicidas, nao_suicidas ):
    # Inicializando variaveis
    y = []
    prob = []

    # Classificando frases
    for suicida in suicidas:
        r = classifier.predict([gerar_vetor(model, suicida)])
        p = classifier.predict_proba([gerar_vetor(model, suicida)])
        y.append(r[0])
        prob.append(p[0][r])

    for nao_suicida in nao_suicidas:
        r = classifier.predict([gerar_vetor(model, suicida)])
        p = classifier.predict_proba([gerar_vetor(model, suicida)])
        y.append(r[0])
        prob.append(p[0][r])

    # Calculando o ROC
    fpr, tpr, thresholds = metrics.roc_curve( y, prob )
    return [fpr, tpr, thresholds]
    
#Função que gera o gráfico(curva ROC )
def plot_ROC( nao_suicida, taxa_verdadeiro_positivo, auc ):
    fig = plt.figure()
    fig.set_size_inches( 15, 5 )
    taxa_verdadeiro_positivos = fig.add_subplot( 1, 2, 1 )

    taxa_verdadeiro_positivos.plot( nao_suicida, taxa_verdadeiro_positivo, color = 'darkgreen',
             lw = 2, label = 'ROC curve (area = %0.2f)' % auc )
    taxa_verdadeiro_positivos.plot( [0, 1], [0, 1], color = 'navy', lw = 1, linestyle = '--' )
    taxa_verdadeiro_positivos.grid()
    plt.xlim( [0.0, 1.0] )
    taxa_verdadeiro_positivos.set_xticks( np.arange( -0.1, 1.0, 0.1 ) )
    plt.ylim( [0.0, 1.05] )
    taxa_verdadeiro_positivos.set_yticks( np.arange( 0, 1.05, 0.1 ) )
    plt.xlabel( 'Taxa de Falsos  Positivos' )
    plt.ylabel( 'Taxa de Verdadeiros Positivos' )
    plt.title( 'ROC' )
    taxa_verdadeiro_positivos.legend( loc = "lower right" )

    plt.show()
    return plt

def print_metrics(vetor_esperado, vetor_resultados):
    precisao = metrics.precision_score(vetor_esperado, vetor_resultados)
    print("Taxa de precisão utilizando:\n{:6.2f}%".format(precisao * 100))

    acuracia = metrics.accuracy_score(vetor_esperado, vetor_resultados)
    print("Taxa de acurácia utilizando:\n{:6.2f}%".format(acuracia * 100))

    recall = metrics.recall_score(vetor_esperado, vetor_resultados)
    print("Taxa de Recall utilizando:\n{:6.2f}%".format(recall * 100))

    f1 = metrics.f1_score(vetor_esperado, vetor_resultados)
    print("Taxa de F1 utilizando:\n{:6.2f}%".format(f1 * 100))


# Utilização dos dados de treinamento
arq = open('suicidas.txt', 'r')
suicida_texts = tratar_texto(arq)
arq.close()
arq = open('nao_suicidas.txt', 'r')
nao_suicida_texts = tratar_texto(arq)
arq.close()

# Divisão da base de treinamento e validação
treino_suicida, validacao_suicida = dividir_base(suicida_texts)
treino_nao_suicida, validacao_nao_suicida = dividir_base(nao_suicida_texts)

# Cria a label para o vetor. Esta é a forma que o doc2vec aceita
tagged_data = [TaggedDocument(words=linha, tags=['0','NÃO_SUICIDA_'+str(index)]) for index, linha in enumerate(treino_nao_suicida)]
tagged_data += [TaggedDocument(words=linha, tags=['1','SUICIDA_'+str(index)]) for index, linha in enumerate(treino_suicida)]
#print(tagged_data)


#Inicializando e treinando modelo
model = treinar_modelo(tagged_data)
model.save("2d2v.model")
print("Model Saved")

#carrega o modelo treinamento.
model= Doc2Vec.load("2d2v.model")

# Gerando classificador
classificador_reg_logistica     = gerar_classificador_reg_logistica(model, treino_suicida, treino_nao_suicida)
classificador_random_forest     = gerar_classificador_random_forest(model, treino_suicida, treino_nao_suicida)
classificador_naive_bayes       = gerar_classificador_naive_bayes(model, treino_suicida, treino_nao_suicida)
classificador_gradient_boosting = gerar_classificador_gradient_boosting(model, treino_suicida, treino_nao_suicida)

# precisao = aferir_precisao(model, classificador, validacao_suicida, validacao_nao_suicida)
# print('Taxa de acertos: {:02.2f}'.format(precisao))

###################### GERAÇÃO DA AUC E CURVA ROC #############################

fpr, tpr, thresholds = calc_roc(model, classificador_reg_logistica, validacao_suicida, validacao_nao_suicida)
auc_reg_logistica = metrics.auc(fpr, tpr)
print(auc_reg_logistica)
plot_ROC(fpr, tpr, auc_reg_logistica)


fpr, tpr, thresholds = calc_roc(model, classificador_random_forest, validacao_suicida, validacao_nao_suicida)
auc_random_forest = metrics.auc(fpr, tpr)
print(auc_random_forest)
plot_ROC(fpr, tpr, auc_random_forest)


fpr, tpr, thresholds = calc_roc(model, classificador_naive_bayes, validacao_suicida, validacao_nao_suicida)
auc_naive_bayes = metrics.auc(fpr, tpr)
print(auc_naive_bayes)
plot_ROC(fpr, tpr, auc_naive_bayes)

fpr, tpr, thresholds = calc_roc(model, classificador_gradient_boosting, validacao_suicida, validacao_nao_suicida)
auc_gradient_boosting = metrics.auc(fpr, tpr)
print(auc_gradient_boosting)
plot_ROC(fpr, tpr, auc_gradient_boosting)


###############################
#####  Testando Precisão  #####
###############################

# Vetores de validação: validacao_suicida, validacao_nao_suicida

vetores_frases_suicidas = []
for frase in validacao_suicida:
    vetores_frases_suicidas.append(gerar_vetor(model, frase))

vetores_frases_nao_suicidas = []
for frase in validacao_nao_suicida:
    vetores_frases_nao_suicidas.append(gerar_vetor(model, frase))

vetor_resultado_esperado = [1 for i in range(len(vetores_frases_suicidas))]
vetor_resultado_esperado += [0 for i in range(len(vetores_frases_nao_suicidas))]

#Atribuição dos valores aos vetores 

vetor_reg_logistica = []
for frase in vetores_frases_suicidas:
    resultado = classificador_reg_logistica.predict_proba([frase])
    if resultado[0][0] < resultado[0][1]:
        vetor_reg_logistica.append(1)
    else:
        vetor_reg_logistica.append(0)
for frase in vetores_frases_nao_suicidas:
    resultado = classificador_reg_logistica.predict_proba([frase])
    if resultado[0][0] < resultado[0][1]:
        vetor_reg_logistica.append(1)
    else:
        vetor_reg_logistica.append(0)

print("\n\n####  Regressão Logistica  ####")
print_metrics(vetor_resultado_esperado, vetor_reg_logistica)
###########################################################################

vetor_random_forest = []
for frase in vetores_frases_suicidas:
    resultado = classificador_random_forest.predict_proba([frase])
    if resultado[0][0] < resultado[0][1]:
        vetor_random_forest.append(1)
    else:
        vetor_random_forest.append(0)
for frase in vetores_frases_nao_suicidas:
    resultado = classificador_random_forest.predict_proba([frase])
    if resultado[0][0] < resultado[0][1]:
        vetor_random_forest.append(1)
    else:
        vetor_random_forest.append(0)

print("\n\n####  Random Forest  ####")
print_metrics(vetor_resultado_esperado, vetor_random_forest)

#############################################################################

vetor_naive_bayes = []
for frase in vetores_frases_suicidas:
    resultado = classificador_naive_bayes.predict_proba([frase])
    if resultado[0][0] < resultado[0][1]:
        vetor_naive_bayes.append(1)
    else:
        vetor_naive_bayes.append(0)
for frase in vetores_frases_nao_suicidas:
    resultado = classificador_naive_bayes.predict_proba([frase])
    if resultado[0][0] < resultado[0][1]:
        vetor_naive_bayes.append(1)
    else:
        vetor_naive_bayes.append(0)

print("\n\n####  Naive Bayes  ####")
print_metrics(vetor_resultado_esperado, vetor_naive_bayes)

#############################################################################

vetor_gradient_boosting = []
for frase in vetores_frases_suicidas:
    resultado = classificador_gradient_boosting.predict_proba([frase])
    if resultado[0][0] < resultado[0][1]:
        vetor_gradient_boosting.append(1)
    else:
        vetor_gradient_boosting.append(0)
for frase in vetores_frases_nao_suicidas:
    resultado = classificador_gradient_boosting.predict_proba([frase])
    if resultado[0][0] < resultado[0][1]:
        vetor_gradient_boosting.append(1)
    else:
        vetor_gradient_boosting.append(0)

print("\n\n####  Gradient Boosting  ####")
print_metrics(vetor_resultado_esperado, vetor_gradient_boosting)



############################
#####  Testando frase  #####
############################

while(0):
    frase = input('Favor informar frase a ser testada: ')
    print('\nFrase a ser testada: \"' + frase + '\"')
    frase = tratar_texto([frase])
    print('\nFrase após tratamento:')
    print(frase)
    vetor = gerar_vetor(model, frase[0])
    print('Vetor gerado a partir da frase:')
    print(vetor)
    resultado = classificador_reg_logistica.predict_proba([vetor])
    print('\nResultado: ')
    print(' * {:6.2f}% de chance da frase possuir ideação suicida'.format(resultado[0][1]     * 100))
    print(' * {:6.2f}% de chance da frase não possuir ideação suicida'.format(resultado[0][0] * 100))
    print('\n * Frase com ideação suicida' if resultado[0][0] < resultado[0][1] else '\n * Frase sem ideção suicida!')

    if (input('\n\nDeseja testar uma nova frase?(sim)(não)\n') == 'não'):
        break
