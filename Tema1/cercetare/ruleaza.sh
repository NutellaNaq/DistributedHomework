# Ruleaza toate experimentele pe cluster si adauga iesirea in rezultate.txt.
#
# Pregatire (pe AMBELE calculatoare, la aceeasi cale):
#   mpic++ -O3 -std=c++17 studiu.cpp -o studiu
# Doar pe calculatorul de pe care lansati (rank 0):
#   python3 genereaza_imagini.py
#   cp hosts.exemplu hosts   # si completati numele/IP-urile si numarul de core-uri
#
# Rulare: ./ruleaza.sh
# Variabilele de mai jos pot fi suprascrise din linia de comanda, de exemplu:
#   PROCESE="1 2 4 8 16 32" ./ruleaza.sh
# Comenzile sunt pentru Open MPI (--hostfile, --map-by node, --oversubscribe).
set -euo pipefail
cd "$(dirname "$0")"

# Ultima valoare trebuie sa depaseasca totalul de core-uri din cluster (ex. 2x),
# ca sa se vada limitarea data de numarul de core-uri.
PROCESE=${PROCESE:-"1 2 4 8 16"}
IMAGINI=${IMAGINI:-"imagini/input_2000x2000.ppm imagini/input_4000x4000.ppm imagini/input_6000x6000.ppm"}
REPETARI=${REPETARI:-5}
BLOCURI=${BLOCURI:-64}
RAZA=${RAZA:-4}
HOSTFILE=${HOSTFILE:-hosts}
IESIRE=${IESIRE:-rezultate.txt}

[ -x ./studiu ] || { echo "Lipseste ./studiu - compilati mai intai."; exit 1; }
[ -f "$HOSTFILE" ] || { echo "Lipseste $HOSTFILE - vezi hosts.exemplu."; exit 1; }

# Toate combinatiile (imagine, p), in ordine amestecata, ca efectele de ordine
# (incalzire, alte procese) sa nu favorizeze sistematic o configuratie.
combinatii=()
for img in $IMAGINI; do
    [ -f "$img" ] || { echo "Lipseste $img - rulati genereaza_imagini.py."; exit 1; }
    for p in $PROCESE; do combinatii+=("$img $p"); done
done
amestecate=()
while IFS= read -r linie; do amestecate+=("$linie"); done < <(
    printf '%s\n' "${combinatii[@]}" | awk 'BEGIN{srand()} {print rand() "\t" $0}' | sort -n | cut -f2-)
combinatii=("${amestecate[@]}")

mkdir -p iesiri
echo "### Start $(date '+%m/%d/%Y %H:%M:%S') pe $(hostname)" >> "$IESIRE"
for c in "${combinatii[@]}"; do
    read -r img p <<< "$c"
    echo ">>> $(date '+%H:%M:%S')  $img  p=$p"
    echo "### Imagine=$img; p=$p" >> "$IESIRE"
    mpirun --hostfile "$HOSTFILE" --map-by node --oversubscribe -np "$p" \
        ./studiu "$img" iesiri/rezultat "$REPETARI" "$BLOCURI" "$RAZA" | tee -a "$IESIRE" | grep '^CSV'
done
echo "### Final $(date '+%m/%d/%Y %H:%M:%S')" >> "$IESIRE"
echo "Gata. Urmeaza: python3 analiza.py $IESIRE"
