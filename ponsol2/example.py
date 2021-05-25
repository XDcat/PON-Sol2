import sys
import os
from model import PonSol2

if __name__ == "__main__":
    ponsol2 = PonSol2()
    seq = "MAKFEDKVDDNSPKVLCESSNQPVKEHS"
    aa = "M1A"
    print(ponsol2.predict(seq, aa)[0])

