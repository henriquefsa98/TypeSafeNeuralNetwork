import sys

import torch

from torch import nn
from torch import optim




class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

       
    def forward(self, x):
        
        x = torch.sigmoid(self.layer_1(x))
        #x = torch.sigmoid(self.layer_2(x))
        #x = torch.round(self.layer_2(x))
        x = self.layer_2(x)


        return x

    
    def train(self, inputs, outputs, criterion, optimizer, learning_rate=0.025, steps=1):
        
        loss_list = []

        if inputs is None or outputs is None or (inputs == [] or outputs == []):
            print("Input e/ou output nao definidos! Nao e possivel treinar a rede")
            return

        if criterion is None or optimizer is None:
            print("Criterio e/ou otimizador nao definidos! Nao e possivel treinar a rede")
            return


        for t in range(steps):

            y_pred = self(inputs)
            loss = criterion(y_pred, outputs)
            loss_list.append(loss.item())
            self.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in self.parameters():
                    param -= learning_rate * param.grad



    def predict(self, inputs):

        return self(inputs)




def plot_pred_out(inputs, preds, outputs):
    
    if not(len(inputs) == len(preds) and len(inputs) == len(outputs)):
        print("Tamanhos de input, pred e output incompativeis! Verificar se argumentos estao corretos")
        return

    counter = 0

    if type(preds) == torch.Tensor:
        preds = preds.tolist()

    for i in range(len(inputs)):

        print("Input = {}, Predict = {}, Expected = {}".format(inputs[i], preds[i], outputs[i]))

        if preds[i] == outputs[i]:
            counter+=1

    print("\nSummary: {}% precision\n".format((counter/len(inputs)) * 100))
    


def main(args):

    # parametros inicias da rede:
    input_dim = 2
    hidden_dim = 10
    output_dim = 1
    batch_size = 0
    learning_rate = 0.025

    
    model = NeuralNetwork(input_dim, hidden_dim, output_dim)
    print(model)

    # lendo os inputs e outputs a ser usado:
    with open("inputs.txt", 'r') as f:
        inputstxt = f.readlines()

    samples = [[float(j) for j in x.split(" ")] for x in inputstxt ]

    #print("Samples: {}".format(samples))

    inputs  = [[x[0], x[1]] for x in samples]
    outputs = [[x[2]] for x in samples]

    inputs_t  = torch.FloatTensor(inputs)
    outputs_t = torch.FloatTensor(outputs)

    # definicao dos parametros de otimizacao:
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # predicao nao treinada:
    y_pred = model.predict(inputs_t)
    plot_pred_out(inputs, y_pred, outputs)
    print("\n\n")
    
    # definicao do treino da rede:
    model.train(inputs_t, outputs_t, loss_function, optimizer, learning_rate, 1000)


    # predicao treinada:
    y_pred = model.predict(inputs_t)

    plot_pred_out(inputs, y_pred, outputs)



if __name__ == "__main__":
    main(sys.argv[1:])