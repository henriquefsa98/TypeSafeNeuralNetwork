import sys

import torch
import torch.nn as nn


import matplotlib.pyplot as plt



def main(args):


    with open("inputs.txt", 'r') as f:
        inputstxt = f.readlines()


    #samples = [ [float(y)] for y in x.split(" ") for x in inputstxt]

    samples = [[float(j) for j in x.split(" ")] for x in inputstxt ]

    inputs = [[x[0], x[1]] for x in samples]

    outputs = [[x[2]] for x in samples]

    inputs_t = torch.FloatTensor(inputs)
    outputs_t = torch.FloatTensor(outputs)

    n_input, n_hidden, n_out, batch_size, learning_rate = len(inputs[0]), 15, len(outputs[0]), 100, 0.01    # verificar o batch size

    # declarar o modelo com todas as funcs de ativacao como sigmoid quebra a rede!
    model = nn.Sequential(nn.Linear(n_input, n_hidden),
                          #nn.Sigmoid(),
                          nn.Linear(n_hidden, n_hidden),
                          #nn.Sigmoid(n_input, n_hidden),
                          #nn.Sigmoid(n_hidden, n_hidden),
                          #nn.Sigmoid(),
                          nn.Linear(n_hidden, n_out),
                          nn.Sigmoid())

    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(5000):
        pred_y = model(inputs_t)
        loss = loss_function(pred_y, outputs_t)
        losses.append(loss.item())

        model.zero_grad()
        loss.backward()

        optimizer.step()


    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f"%(learning_rate))
    plt.show()

    for x in range(50):

        print("valores de entrada: {}, valor previsto: {}, valor esperado: {}".format(inputs_t[x], pred_y[x], outputs_t[x]))

if __name__ == "__main__":
    main(sys.argv[1:])

