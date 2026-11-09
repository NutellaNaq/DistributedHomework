"""Analiza rezultatelor: citeste liniile "CSV,..." produse de studiu si face graficele.

Utilizare: python3 analiza.py [rezultate.txt ...]
Necesita: matplotlib (pip install matplotlib).

Iesire:
  grafice/timp_vs_rezolutie.png  - timp vs. rezolutie, o serie pe numar de procese
  grafice/timp_vs_procese.png    - timp vs. p, cele 4 variante MPI + secvential
  grafice/accelerare.png         - accelerare S = Tseq / Tmpi vs. p, cu linia ideala
  grafice/tabel.md               - tabel cu toate mediile, accelerarea si eficienta

Baza secventiala: varianta "seq" din jobul cu p=1, pentru fiecare operatie si
rezolutie. Daca lipseste, se foloseste media variantelor "seq" din toate joburile.
Eficienta E = S / p (p include si procesul coordonator).
"""
import os
import sys
from collections import defaultdict
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OPERATII = ["gray", "blur", "rotate"]
VARIANTE = [("1d", "static"), ("1d", "dynamic"), ("2d", "static"), ("2d", "dynamic")]
NUME = {("1d", "static"): "1D static", ("1d", "dynamic"): "1D dinamic",
        ("2d", "static"): "2D static", ("2d", "dynamic"): "2D dinamic"}
# Paleta categoriala validata (ordine fixa) + markere diferite ca a doua codificare.
CULORI = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
MARKERE = ["o", "s", "^", "D", "v", "P"]
GRI = "#52514e"
SUPRAFATA = "#fcfcfb"
TEXT = "#0b0b0b"


def citeste(fisiere):
    # cheie: (op, megapixeli, p, mod, divizare, balansare) -> lista de (medie, deviatie)
    date = defaultdict(list)
    calculatoare = defaultdict(set)
    for fisier in fisiere:
        with open(fisier, encoding="utf-8") as f:
            for linie in f:
                if not linie.startswith("CSV,"):
                    continue
                c = linie.strip().split(",")[1:]
                op, w, h, p, hosts, mod, div, bal = c[0], int(c[1]), int(c[2]), int(c[3]), int(c[4]), c[5], c[6], c[7]
                medie, dev = float(c[11]), float(c[12])
                mp = round(w * h / 1e6, 1)
                date[(op, mp, p, mod, div, bal)].append((medie, dev))
                calculatoare[p].add(str(hosts))
    # Daca aceeasi configuratie a fost rulata de mai multe ori, se face media.
    return {k: (mean(m for m, _ in v), mean(d for _, d in v)) for k, v in date.items()}, calculatoare


def baza_secventiala(date, op, mp):
    if (op, mp, 1, "seq", "-", "-") in date:
        return date[(op, mp, 1, "seq", "-", "-")][0]
    valori = [v[0] for k, v in date.items() if k[0] == op and k[1] == mp and k[3] == "seq"]
    return mean(valori) if valori else None


def stil_axe(ax):
    ax.set_facecolor(SUPRAFATA)
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#a8a7a2")
    ax.tick_params(colors=TEXT, labelsize=9)


def figura(titlu):
    fig, axe = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SUPRAFATA)
    fig.suptitle(titlu, color=TEXT, fontsize=13)
    for ax in axe:
        stil_axe(ax)
    return fig, axe


def salveaza(fig, axe, cale, nota):
    handles, labels = axe[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=len(labels), frameon=False, fontsize=9)
    fig.text(0.01, 0.01, nota, color=GRI, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    fig.savefig(cale, dpi=150, facecolor=SUPRAFATA)
    plt.close(fig)
    print(f"scris: {cale}")


def main():
    fisiere = sys.argv[1:] or ["rezultate.txt"]
    date, calculatoare = citeste(fisiere)
    if not date:
        sys.exit("Nu am gasit linii CSV in " + ", ".join(fisiere))
    os.makedirs("grafice", exist_ok=True)
    rezolutii = sorted({k[1] for k in date})
    procese = sorted({k[2] for k in date})
    mp_max = rezolutii[-1]
    nota = ("Medii din 5 rulari; bare = ±1 deviatie standard. p include coordonatorul. "
            "Calculatoare per p: " + ", ".join(f"p={p}: {'/'.join(sorted(calculatoare[p]))}" for p in procese))

    def cea_mai_buna(op, mp, p):
        # Cea mai rapida dintre cele 4 variante MPI la (op, mp, p).
        opt = [(date[k][0], v) for v in VARIANTE if (k := (op, mp, p, "mpi", *v)) in date]
        return min(opt) if opt else (None, None)

    # 1) Timp vs. rezolutie: secvential + cea mai buna varianta MPI pentru fiecare p.
    fig, axe = figura("Timp de executie vs. rezolutia imaginii (cea mai rapida varianta MPI pentru fiecare p)")
    for ax, op in zip(axe, OPERATII):
        seq = [baza_secventiala(date, op, mp) for mp in rezolutii]
        ax.plot(rezolutii, [s * 1000 for s in seq], "--", color=GRI, marker="X", markersize=8,
                linewidth=2, label="secvential")
        for i, p in enumerate(q for q in procese if q > 1):
            t = [cea_mai_buna(op, mp, p)[0] for mp in rezolutii]
            ax.plot(rezolutii, [x * 1000 if x else None for x in t], color=CULORI[i % 6],
                    marker=MARKERE[i % 6], markersize=8, linewidth=2, label=f"MPI p={p}")
        ax.set_title(op, color=TEXT)
        ax.set_xlabel("Rezolutie (megapixeli)", color=TEXT)
        ax.set_ylabel("Timp (ms)", color=TEXT)
        ax.set_xticks(rezolutii)
        ax.set_ylim(bottom=0)
    salveaza(fig, axe, "grafice/timp_vs_rezolutie.png", nota)

    # 2) Timp vs. p la rezolutia maxima: cele 4 variante + secvential.
    fig, axe = figura(f"Timp de executie vs. numarul de procese ({mp_max:g} MP)")
    for ax, op in zip(axe, OPERATII):
        seq = baza_secventiala(date, op, mp_max)
        ax.axhline(seq * 1000, color=GRI, linestyle="--", linewidth=2, label="secvential (p=1)")
        for i, v in enumerate(VARIANTE):
            ps = [p for p in procese if (op, mp_max, p, "mpi", *v) in date]
            ax.errorbar(ps, [date[(op, mp_max, p, "mpi", *v)][0] * 1000 for p in ps],
                        yerr=[date[(op, mp_max, p, "mpi", *v)][1] * 1000 for p in ps],
                        color=CULORI[i], marker=MARKERE[i], markersize=8, linewidth=2,
                        capsize=3, label=NUME[v])
        ax.set_title(op, color=TEXT)
        ax.set_xlabel("Numar de procese p", color=TEXT)
        ax.set_ylabel("Timp (ms)", color=TEXT)
        ax.set_xscale("log", base=2)
        ax.set_xticks(procese, [str(p) for p in procese])
        ax.set_ylim(bottom=0)
    salveaza(fig, axe, "grafice/timp_vs_procese.png", nota)

    # 3) Accelerare vs. p la rezolutia maxima.
    fig, axe = figura(f"Accelerare S = Tsecvential / Tmpi ({mp_max:g} MP)")
    for ax, op in zip(axe, OPERATII):
        seq = baza_secventiala(date, op, mp_max)
        ax.plot(procese, procese, ":", color=GRI, linewidth=1.5, label="ideal (S = p)")
        for i, v in enumerate(VARIANTE):
            ps = [p for p in procese if (op, mp_max, p, "mpi", *v) in date]
            ax.plot(ps, [seq / date[(op, mp_max, p, "mpi", *v)][0] for p in ps], color=CULORI[i],
                    marker=MARKERE[i], markersize=8, linewidth=2, label=NUME[v])
        ax.axhline(1, color="#a8a7a2", linewidth=1)
        ax.set_title(op, color=TEXT)
        ax.set_xlabel("Numar de procese p", color=TEXT)
        ax.set_ylabel("Accelerare S", color=TEXT)
        ax.set_xscale("log", base=2)
        ax.set_xticks(procese, [str(p) for p in procese])
        ax.set_ylim(bottom=0)
    salveaza(fig, axe, "grafice/accelerare.png", nota)

    # 4) Tabel complet.
    with open("grafice/tabel.md", "w", encoding="utf-8") as f:
        f.write("| Operatie | MP | p | Varianta | Timp mediu (ms) | Dev. std (ms) | S | E |\n")
        f.write("|---|---:|---:|---|---:|---:|---:|---:|\n")
        for op in OPERATII:
            for mp in rezolutii:
                seq = baza_secventiala(date, op, mp)
                f.write(f"| {op} | {mp:g} | 1 | secvential | {seq * 1000:.1f} | - | 1.00 | - |\n")
                for p in procese:
                    for v in VARIANTE:
                        k = (op, mp, p, "mpi", *v)
                        if k in date:
                            t, d = date[k]
                            f.write(f"| {op} | {mp:g} | {p} | {NUME[v]} | {t * 1000:.1f} | {d * 1000:.1f} "
                                    f"| {seq / t:.2f} | {100 * seq / t / p:.0f}% |\n")
    print("scris: grafice/tabel.md")


if __name__ == "__main__":
    main()
