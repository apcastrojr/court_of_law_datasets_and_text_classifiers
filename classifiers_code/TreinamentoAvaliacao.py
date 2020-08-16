#By Antonio Pires de Castro Jr
#apcastrojr
#April/2020
class TreinamentoAvaliacao:
  import sys
  import numpy as np
  #from TreinamentoAvaliacao import TreinamentoAvaliacao

  def ler_dataset(self, arquivo_dataset):
    #f = open("dataset_termos_combinados_coseno_0.5binario.txt", "r")
    f = open(arquivo_dataset, "r")
    dataset_array=[]
    label_array=[]
    for x in f:
        data = []
        label= 0
        for dado in x.split(';'):
            if "#" in dado:
                data.append(int(dado.split('#')[0]))
                label=int(dado.split('#')[1].split('\n')[0])
            else:
                data.append(int(dado))
        dataset_array.append(data)
        label_array.append(label)  
    f.close()
    #print(dataset_array)
    #print(label_array)
    return dataset_array,label_array

  def tree(self, dataset_array, label_array, data_teste):
    from sklearn import tree

    clf = tree.DecisionTreeClassifier()
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def naivebayes(self, dataset_array, label_array, data_teste):
    from sklearn.naive_bayes import GaussianNB

    #Create a Gaussian Classifier
    clf = GaussianNB()
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def mlp(self, dataset_array, label_array, data_teste):
    from sklearn.neural_network import MLPClassifier

    #analise 2 resultados
    clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50), random_state=1, max_iter=400)
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)
    #clf.score(dataset_array, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

  def svm(self, dataset_array, label_array, data_teste):
    from sklearn import svm

    #analise 2 resultados
    clf = svm.SVC()
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def avaliar_acuracia(self, array_predicao, array_label_corretos):
    acuracia=cont=0
    for i in array_predicao:
      if i == array_label_corretos[cont]:
        acuracia+=1
      cont+=1
    return (acuracia / cont) * 100

  if __name__ == '__main__':
    if len(sys.argv) != 4:
       print("Erro na passagem dos argumentos")
       print("Correto:")
       print("        python3 <nome_arquivo_dataset> <nome_arquivo_teste> <nome_metodo_predicao [tree, naivebayes, mlp, svm]>")
    else:
       #constantes
       arquivo_dataset  =sys.argv[1]
       arquivo_datateste=sys.argv[2]
       metodo_predicao  =sys.argv[3]
       from TreinamentoAvaliacao import TreinamentoAvaliacao

       print(metodo_predicao)

       ta = TreinamentoAvaliacao()
       dataset_array, label_array = ta.ler_dataset(arquivo_dataset)
       data_teste, label_teste = ta.ler_dataset(arquivo_datateste)

       resultado_predicao=[]
       if metodo_predicao == "tree":
          resultado_predicao = ta.tree(dataset_array,label_array,data_teste)
       elif metodo_predicao == "naivebayes":
          resultado_predicao = ta.naivebayes(dataset_array,label_array,data_teste)
       elif metodo_predicao == "mlp":
          resultado_predicao = ta.mlp(dataset_array,label_array,data_teste)
       elif metodo_predicao == "svm":
          resultado_predicao = ta.svm(dataset_array,label_array,data_teste)

       if resultado_predicao == []:
         print("***************************************")
         print("  Erro:")
         print("       Nome método de predição errado. Passar o tipo correto: [tree, naivebayes, mlp, svm]")
         print("***************************************")
       else:
         acuracia = ta.avaliar_acuracia(resultado_predicao,label_teste)
         print("***************************************")
         print(acuracia)
         #print(resultado_predicao)
         #print(resultado_datateste)
         print("***************************************")
