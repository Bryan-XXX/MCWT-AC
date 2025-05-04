import os
import sys
import random

fat = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
density = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

mindata = int(100 * (1 * 1024.0 * 1024.0))
maxdata = int(500 * (1 * 1024.0 * 1024.0))


def main():
    for i in range(1000):
        command = './daggen --dot -n 10'
        file_name = 'random.10.' + str(i) + '.gv'
        command = (command + ' --fat ' + str(fat[random.randint(0, len(fat) - 1)]) +
                   ' --density ' + str(density[random.randint(0, len(density) - 1)]) +
                   ' --mindata ' + str(mindata) +
                   ' --maxdata ' + str(maxdata))
        print(command)

        os.system(command + '> ' + file_name)


if __name__ == '__main__':
    main()
