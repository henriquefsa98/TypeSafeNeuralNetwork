# generate a txt file with n inputs and its respective expected output, using a arg as n
# to use as input to TypeSafeNN 



# get sys args
import sys

# get random numbers to make an input
import random



# get imc output with some weight and height input
def imc_output(weight, height):

    imc = weight / (height * height)

    if imc >= 25:
        return 1
    else:
        return 0


def imc_sample():
    
    w = random.SystemRandom().uniform(40.00, 200)
    h = random.SystemRandom().uniform(1.00, 2.20)

    output = imc_output(w, h)

    return [w, h, output]


# only return samples that output = 1
def imc_sample_1():
    
    w = random.SystemRandom().uniform(40.00, 200)
    h = random.SystemRandom().uniform(1.00, 2.20)

    output = imc_output(w, h)

    while(output != 1):

        w = random.SystemRandom().uniform(40.00, 200)
        h = random.SystemRandom().uniform(1.00, 2.20)

        output = imc_output(w, h)

    return [w, h, output]


# only return samples that output = 0
def imc_sample_0():
    
    w = random.SystemRandom().uniform(40.00, 200)
    h = random.SystemRandom().uniform(1.00, 2.20)

    output = imc_output(w, h)

    while(output != 0):

        w = random.SystemRandom().uniform(40.00, 200)
        h = random.SystemRandom().uniform(1.00, 2.20)

        output = imc_output(w, h)

    return [w, h, output]


def main(args):

    if args == [] or not args[0].isnumeric():
        print("Invalid argument. Please set some integer number of inputs!")
        return

    inputs_samples = int(args[0])


    samples = []

    for x in range(round(inputs_samples/2)):
        samples.append(imc_sample_0())

    for x in range((round(inputs_samples/2)), inputs_samples):
        samples.append(imc_sample_1())

    random.shuffle(samples)

    #TODO salvar samples num arquivo txt, para ser lido pela IA em haskell

    print(samples)

    with open("inputs.txt", 'w') as f:

        linhas = [str(x[0]) + " " + str(x[1]) + " " + str(x[2]) + "\n" for x in samples]

        f.writelines(linhas)


    f.close()


if __name__ == "__main__":
    
    main(sys.argv[1:])

