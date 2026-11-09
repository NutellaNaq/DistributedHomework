"""Genereaza imaginile de test (PPM binar P6, maxval 255) pentru studiu.

Utilizare: python3 genereaza_imagini.py [director=imagini]

Continutul este pseudo-aleator (seed fix), deci imaginile sunt identice la fiecare
generare. Pentru gray/blur/rotate timpul nu depinde de valorile pixelilor, doar de
dimensiuni. Fara dependinte externe.
"""
import os
import random
import sys

# latime, inaltime -> 4 MP, 16 MP, 36 MP
REZOLUTII = [(2000, 2000), (4000, 4000), (6000, 6000)]


def genereaza(cale, latime, inaltime):
    rnd = random.Random(latime * 100003 + inaltime)
    # Un rand de baza mai lung; randul y este o fereastra deplasata din el.
    baza = bytes(rnd.getrandbits(8) for _ in range(3 * (latime + inaltime)))
    with open(cale, "wb") as f:
        f.write(f"P6\n{latime} {inaltime}\n255\n".encode())
        for y in range(inaltime):
            f.write(baza[3 * y: 3 * (y + latime)])


def main():
    director = sys.argv[1] if len(sys.argv) > 1 else "imagini"
    os.makedirs(director, exist_ok=True)
    for latime, inaltime in REZOLUTII:
        cale = os.path.join(director, f"input_{latime}x{inaltime}.ppm")
        if os.path.exists(cale):
            print(f"exista deja: {cale}")
            continue
        genereaza(cale, latime, inaltime)
        print(f"generat: {cale} ({latime * inaltime / 1e6:.0f} MP)")


if __name__ == "__main__":
    main()
