# Convolutional encoder/decoder
# BMS project - FIT VUT - 2020
# Marek Salon (xsalon00)

import sys
import argparse
import re
from functools import reduce
from itertools import groupby

# path tuple indexes
STATE = 2
ERR_C = 1
PATH = 0

def main():
    """ 
        Main function. Parses command line arguments, calls appropriate functions and prints results to stdout.
    """
    # parsing arguments using argparse module
    parser = argparse.ArgumentParser(prog='bms', description='Convolutional encoder/decoder', epilog="Author: Marek Salon (xsalon)", formatter_class=argparse.RawTextHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-e', action='store_true', help='encoding mode (input: stdin)')
    group.add_argument('-d', action='store_true', help='decoding mode (input: stdin)')
    parser.add_argument('-p', '--params', nargs=3, metavar=('X', 'Y', 'Z'), type=int, help="X - number of delay cells\nY - upper feedback scheme\nZ - lower feedback scheme")
    args = parser.parse_args()

    # filter not allowed characters from stdin and start encoding/decoding
    if args.e:
        input = re.sub(r"[^0-9A-Za-z]*", '', ''.join(sys.stdin))
        if not input:
            print('')
            return
        if args.params != None:
            if not all(p > 0 for p in args.params):
                parser.error("params must be higher than 0.")
            print(''.join(encode(input, args.params)))
        else:
            print(''.join(encode(input)))
    elif args.d:
        input = re.sub(r"[^0-1]*", '', ''.join(sys.stdin))
        if not input:
            print('')
            return
        if args.params != None:
            if not all(p > 0 for p in args.params):
                parser.error("params must be higher than 0.")
            print(''.join(decode(input, args.params)))
        else:
            print(''.join(decode(input)))


def encode(input_str, config=[5,53,46]):
    """
        Encode text input using convolutional encoder with defined configuration.

        If the argument `config` isn't passed in, the default configuration is used.

        Parameters
        ----------
        input_str : str
            Input string to encode

        config : [X,Y,Z], optional
            Configuration of encoder where:
                X : int, number of delay cells (memory blocks)
                Y : int, *upper feedback scheme
                Z : int, *lower feedback scheme 
        
        Returns
        ----------
        Encoded input in form of list of ones and zeroes as chars

        * - ones define connection of specific index
    """
    
    # result list buffer initialization
    res = []
    # initialize state with given X value from config
    state = [0 for x in range(config[0])]
    # make list of parsed input characters in corresponding 8-bit representation 
    input_bin = state + list(map(int, ''.join(('00000000' + bin(ord(c))).replace('b','')[-8:] for c in input_str)))
    # get scheme indexes from config
    indexes_y, indexes_z = getIndexes(config)
    
    # simulates shifting register and computes output into `res` buffer
    for branch in input_bin[::-1]:
        res = calculateOutput(branch, state, indexes_y, indexes_z, res)
        state.pop()
        state.insert(0, branch)

    return list(map(str, res))


def getIndexes(config):
    """
        Compute indexes for feedback schemes.

        Parameters
        ----------
        config : [X,Y,Z]
        
        Returns
        ----------
        Two values where:
            1st value is a list of indexes for upper scheme
            2nd value is a list of indexes for lower scheme
    """

    # get required number of bits from Y and Z values to match size of encoder buffer
    conf_y = (['0' for x in range(config[0])] + list(bin(config[1]).replace('b','')))[(-(config[0]+1)):]
    conf_z = (['0' for x in range(config[0])] + list(bin(config[2]).replace('b','')))[(-(config[0]+1)):]
    
    # save indexes to lists
    indexes_y = [idx for idx, val in enumerate(conf_y) if val == '1']
    indexes_z = [idx for idx, val in enumerate(conf_z) if val == '1']
    
    return indexes_y, indexes_z


def calculateOutput(branch, state, indexes_y, indexes_z, res):
    """
        Calculate encoder output for given branch, state and schemes.

        Parameters
        ----------
        branch : int, [0,1]
            MSB bit of encoder buffer

        state : list of [0,1]
            State of encoder buffer (delay cells)

        indexes_y : list of int
            Upper scheme indexes
        
        indexes_z : list of int
            Lower scheme indexes

        res : list of [0,1]
            Result output buffer
        
        Returns
        ----------
        Modified `res` buffer with new 2-bit encoder output at the front
    """
    
    # create encoder buffer
    buffer = [branch] + state

    # get indexed values to temporary lists
    tmp_y = []
    tmp_z = []

    for idx in indexes_y:
        tmp_y.append(buffer[idx])

    for idx in indexes_z:
        tmp_z.append(buffer[idx])

    # make xor operation over temporary lists
    res_y = reduce(lambda x, y: x ^ y, tmp_y)
    res_z = reduce(lambda x, y: x ^ y, tmp_z)

    # update result buffer
    res.insert(0, res_z)
    res.insert(0, res_y)

    return res


def decode(input_str, config=[5,53,46]):
    """
        Decode binary input using convolutional decoder with defined configuration.

        If the argument `config` isn't passed in, the default configuration is used.

        Parameters
        ----------
        input_str : str
            Input binary string to decode

        config : [X,Y,Z], optional
            Configuration of encoder where:
                X : int, number of delay cells (memory blocks)
                Y : int, *upper feedback scheme
                Z : int, *lower feedback scheme 
        
        Returns
        ----------
        Decoded input in form of list of chars.

        * - ones define connection of specific index
    """

    # initialize state with given X value from config
    state = [0 for x in range(config[0])]
    # parse input binary string into pair tuples
    input_tuples = list(zip(input_str[::2], input_str[1::2]))
    # create list of pairs from pair tuples
    input_pairs = list(''.join(pair) for pair in input_tuples)
    # get scheme indexes from config
    indexes_y, indexes_z = getIndexes(config)

    # initialize first path - structure to help simulate trellis traversal and memorize needed data
    # path is defined by tuple (path, error_count, state) where:
    #   path - decoded portion of input
    #   error_count - sum of Hamming distances of this path
    #   state - latest state of this path
    # paths are stored in list `paths`
    paths = [('', 0, state)]

    # process input from LSB pairs of bits
    for pair in input_pairs[::-1]:
        # create new paths by finding new possible states and calculate error for new states
        new_paths = []
        for path in paths:
            new_paths += eval_step(path, indexes_y, indexes_z, pair)

        # group new paths into lists according to state value
        grouped_paths = []
        for key, group in groupby(sorted(new_paths, key = lambda x: x[STATE]), key = lambda x: x[STATE]): 
            grouped_paths.append(list(group))

        # reduce paths - remove all paths with same state but higher error_count
        reduced_paths = []
        for item in grouped_paths:
            reduced_paths.append(min(item, key = lambda t: t[ERR_C]))
        
        # update `paths` with relevant paths
        paths = reduced_paths.copy()

    # find best path with the lowest error_count
    min_err = sys.maxsize
    best_path = ()
    for path in paths:
        if path[ERR_C] < min_err:
            best_path = path
            min_err = path[ERR_C]
    
    # segment best path into list of 8-bit values
    res_bin = re.findall('........', best_path[PATH][config[0]:])
    
    # create list of corresponding chars for each element from `res_bin`
    # chars are limited by ASCII encoding
    #   - every char with ASCII value > 127 is ignored on output
    #   - printing ASCII value > 127 on merlin leads to errors (otherwise it works fine above 127 on other systems)
    res = map(lambda x: chr(int(x,2)) if int(x,2) < 128 else '', res_bin)
    
    return res


def eval_step(path, indexes_y, indexes_z, pair):
    """
        Evaluate trellis step. Take `path` and calculate all possible new states with error.

        Parameters
        ----------
        path : (PATH, ERR_C, STATE)
            Path tuple where:
                PATH : string, decoded part of input
                ERR_C : int, sum of Hamming distances of this path
                STATE : list of int, latest state of this path
        
        indexes_y : list of int
            Upper scheme indexes
        
        indexes_z : list of int
            Lower scheme indexes

        pair : str
            Pair of bits from input
        
        Returns
        ----------
        List of new possible paths.
    """

    res = []

    # create new states from latest state
    state_a = path[STATE].copy()
    state_b = path[STATE].copy()

    state_a.pop()
    state_b.pop()

    state_a.insert(0, 0)
    state_b.insert(0, 1)

    # calculate error of step to new state (Hamming distance)
    err_a = 0
    err_b = 0
    err_a = calculateHammingDist(0, path[STATE], indexes_y, indexes_z, pair)
    err_b = calculateHammingDist(1, path[STATE], indexes_y, indexes_z, pair)

    # create new paths from new states and errors
    res.append(('0' + path[PATH], path[ERR_C] + err_a, state_a))
    res.append(('1' + path[PATH], path[ERR_C] + err_b, state_b))

    return res


def calculateHammingDist(branch, state, indexes_y, indexes_z, pair):
    """
        Calculate Hamming distance.

        Parameters
        ----------
        branch : int, [0,1]
            MSB bit of encoder buffer

        state : list of [0,1]
            State of encoder buffer (delay cells)
        
        indexes_y : list of int
            Upper scheme indexes
        
        indexes_z : list of int
            Lower scheme indexes

        pair : str
            Pair of bits from input
        
        Returns
        ----------
        Hamming distance/number of different bits.
    """
    
    # simulate encoder step to find out expected encoder output for new state
    res = []
    res = calculateOutput(branch, state, indexes_y, indexes_z, res)

    # calculate Hamming distance between input pair and expected pair
    expected_val = int(''.join(map(str, res)), 2)
    input_val = int(pair, 2)
    dist = str(bin(expected_val ^ input_val)).count('1')

    return dist
    

if __name__ == "__main__":
    main()