#!/usr/bin/env python3

import argparse
from PixelMMV import PixelMMV

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-dir", '--directory', required=True,
        help="path to the input images directory")
    args = vars(ap.parse_args())

    pmmv = PixelMMV(args['directory'])

if __name__ == '__main__':
    main()


