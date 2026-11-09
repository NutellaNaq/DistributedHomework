# Studiu preliminar MPI — imagine 1920 × 1080

## Date și verificări

Au fost analizate fișierele timpi_1.txt, timpi_2.txt și timpi_4.txt: 45 de configurații, fiecare cu 5 rulări, adică 225 de măsurători. Mediile și deviațiile standard au fost recalculate din rulările individuale și coincid cu valorile afișate, în limita preciziei din fișiere. Toate verificările rezultatului sunt OK. Această verificare confirmă egalitatea variantelor cu secvențialul implementat, nu este un test independent al formulelor operațiilor.

Toate procesele au rulat pe calculatorul denis. Imaginea are 1920 × 1080 = 2.073.600 pixeli, aproximativ 2,07 MP. Sunt rezultate locale pentru o singură rezoluție, nu un studiu complet al clusterului cu două calculatoare.

## Metodă

Timpul principal include pregătirea blocurilor, comunicarea, calculul și reasamblarea; exclude citirea/scrierea imaginilor și verificarea rezultatului. Secvențialul procesează un singur bloc. Variantele MPI procesează 64 de blocuri. Schimbarea granulației modifică și copiile, alocările și accesul la memorie.

Accelerarea S = Tseq/Tmpi folosește secvențialul din același fișier/job. Eficiența E = S/p include toate procesele, inclusiv coordonatorul. Secvențialele au fost măsurate din nou în joburile cu p=1, p=2 și p=4; de aceea baza de comparație variază. Nu combinați aceste rapoarte cu o bază comună fără a le recalcula și a preciza metoda.

La p=1 toate blocurile se procesează local: static și dinamic parcurg același cod. Diferențele lor măsurate provin din variabilitatea execuțiilor, nu din planificare dinamică. La p=2 există un coordonator și un singur worker; dinamicul nu poate echilibra sarcina între mai mulți workeri. La p=4 există trei workeri. Staticul folosește loturi sincronizate, în timp ce dinamicul trimite blocul următor primului worker liber. Comparația include și costul așteptării dintre loturi.

## Cele mai mici medii observate la 4 procese

| Operație | Variantă | Timp mediu (ms) | Tseq în jobul p=4 (ms) | S | E |
|---|---|---:|---:|---:|---:|
| gray | 2D dinamic | 3.633 | 9.627 | 2.65 | 66.2% |
| blur | 1D dinamic | 31.133 | 97.848 | 3.14 | 78.6% |
| rotate | 2D dinamic | 5.152 | 17.976 | 3.49 | 87.2% |

În acest set de măsurători, cele patru variante dinamice/statice de la p=4 sunt mai rapide decât secvențialele asociate. Cele mai mici medii sunt obținute cu distribuire dinamică: 2D la gray și rotate, respectiv 1D la blur. Aceasta este o clasificare după medii, nu dovada unei diferențe statistic semnificative. De exemplu, la gray diferența dintre 1D dinamic și 2D dinamic este mică în raport cu deviațiile standard; nici la blur 1D dinamic versus 2D dinamic nu rezultă un câștig clar doar din aceste cinci rulări. Nu există o singură împărțire câștigătoare pentru toate operațiile.

La p=4, variantele dinamice au timpi medii mai mici decât staticele corespunzătoare în toate cele trei operații. Eliminarea așteptărilor dintre loturi poate contribui, dar datele nu izolează această cauză de alte costuri și fluctuații.

Valorile S peste 1 la p=1 și unele valori S peste numărul de workeri la p=4 nu trebuie atribuite exclusiv paralelizării. Secvențialul cu un bloc și varianta pe blocuri au comportamente de memorie diferite; bazele secvențiale variază și ele. Cache-ul, alocările, frecvența procesorului și încărcarea sistemului sunt explicații posibile, care nu au fost măsurate separat.

## Ce mai trebuie pentru cerința completă

1. Repetați același protocol pentru alte rezoluții, inclusiv imagini de zeci de megapixeli, de exemplu 4000 × 4000 (16 MP) și 6000 × 6000 (36 MP).
2. Rulați pe cele două calculatoare ale clusterului și verificați că plasarea include ambele nume de gazdă. Păstrați aceleași rezoluții, numere de procese și 64 de blocuri.
3. Notați pentru fiecare calculator CPU, core-uri fizice/logice, RAM, OS, MPI, tipul rețelei și dacă rulați în WSL/VM/Docker. Aceste detalii nu se pot deduce din timpi.
4. Folosiți numere de procese relevante pentru core-urile disponibile; includeți și un caz de depășire a resurselor dacă îl studiați, etichetat explicit.
5. Construiți graficul timp–rezoluție abia după culegerea datelor pentru mai multe rezoluții. Nu există suficiente date acum pentru acel grafic sau pentru concluzii despre comunicația între două noduri.

Păstrați toate cele cinci rulări, inclusiv valorile mai mari. O încălzire este deja exclusă de program. La măsurătorile următoare, reduceți activitatea de fundal și alternați/variați ordinea configurațiilor pentru a reduce efectele sistematice ale ordinii. Nu înlocuiți măsurătorile existente.
