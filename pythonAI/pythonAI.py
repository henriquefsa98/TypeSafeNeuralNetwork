import sys

import torch
import torch.nn as nn


import matplotlib.pyplot as plt



def main(args):


    with open("inputs.txt", 'r') as f:
        inputstxt = f.readlines()


    #samples = [ [float(y)] for y in x.split(" ") for x in inputstxt]

    samples = [[float(j) for j in x.split(" ")] for x in inputstxt ]

    #print("Samples: {}".format(samples))

    inputs = [[x[0], x[1]] for x in samples]

    outputs = [[x[2]] for x in samples]

    inputs_t = torch.FloatTensor(inputs)
    outputs_t = torch.FloatTensor(outputs)

    n_input, n_hidden, n_out, batch_size, learning_rate = len(inputs[0]), 30, len(outputs[0]), 1, 0.0025    # verificar o batch size

    # declarar o modelo com todas as funcs de ativacao como sigmoid quebra a rede!
    model = nn.Sequential(nn.Linear(n_input, n_hidden),
                            nn.Sigmoid(), 
                            #nn.Linear(n_hidden, int(n_hidden/2)),
                                #nn.ReLU(),
                                #nn.Linear(int(n_hidden/2), n_out), 
                                    nn.Sigmoid())

    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("ninp: {}, nhid: {}, nout: {}\n".format(n_input, n_hidden, n_out))

    #print("Pesos randomicos iniciais:")
    #print(model[0].bias)
    #print(model[0].weight)

    #model[0].weight.data = torch.FloatTensor([[-6.068413195232125e-2, 0.5821336030876887]])
    #model[0].bias.data = torch.FloatTensor([[0.30156576973459015]])

    #print("\n\nPesos setados:")
    #print(model[0].bias)
    #print(model[0].weight)

    #input()

    losses = []
    for epoch in range(50000):
        pred_y = model(inputs_t)
        loss = loss_function(pred_y, outputs_t)
        losses.append(loss.item())

        model.zero_grad()
        loss.backward()

        optimizer.step()

    #model.train()

    #plt.plot(losses)
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.title("Learning rate %f"%(learning_rate))
    #plt.show()

    for y in range(len(inputs_t)):

        if(outputs_t[y] == 0):
            print("Valor 0 encontrado!")
            print("\nvalores de entrada: {}, valor previsto: {}, valor esperado: {}".format(inputs_t[y], pred_y[y], outputs_t[y]))


    for x in range(1):

        print("\nvalores de entrada: {}, valor previsto: {}, valor esperado: {}".format(inputs_t[x], pred_y[x], outputs_t[x]))

    print(model)
    #print(model.parameters())
    print(model[0].weight)
    print(model[0].bias)

if __name__ == "__main__":
    main(sys.argv[1:])

