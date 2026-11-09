// Studiu simplificat: C++17 + MPI, fara generare de tabele sau grafice.
// Compilare: mpic++ -O3 -std=c++17 studiu.cpp -o studiu
// Rulare: mpirun -np 4 ./studiu input.ppm rezultat [repetari=5] [blocuri=64]
// Exemplu cluster Open MPI:
// mpirun --hostfile hosts --map-by node -np 4 ./studiu input.ppm rezultat
// Input: imagine PPM binara P6, maxval 255. Calea input este citita de rank 0.
// Programul testeaza toate operatiile: gray, blur, rotate.
// Pentru fiecare: secvential + MPI 1D/2D, static/dinamic, cu incalzire si repetari.
// Repeta comanda cu alte imagini si alte valori -np. Graficele se fac separat.
// Exemplu rezolutii de pregatit: 2000x2000, 4000x4000, 6000x6000.
// Rezultatele sunt afisate; copiaza-le in Excel. Pentru log: adauga > timpi.txt
// Fisierele rezultat_gray.pgm, rezultat_blur.ppm, rezultat_rotate.ppm sunt
// suprascrise la fiecare varianta; rezultatul este verificat fata de secvential.
// Timpul exclude I/O si verificarea; include pregatirea blocurilor, transferul,
// calculul, reasamblarea si sincronizarea. Kernel=maximul sumelor locale de calcul.
// La p>=2: rank 0 coordonator, p-1 workeri. p=1: toate blocurile local.
// Static: loturi ciclice fixe, cu sincronizare intre loturi. Dinamic: primul liber.
// Numarul tinta de blocuri este fix intre rulari; grila 2D poate rotunji in sus.
// Blur: halo de raza 1 din imaginea originala, inclusiv colturi; media si rotunjirea
// sunt ca in original. Gray: formula originala, fara prag. Rotate: 90 grade orar.
// Acestea sunt variante experimentale, distincte de distributia MPI originala.
// Secventialul nu distribuie date; MPI_Init si citirea sunt in afara masuratorii.
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <climits>
#include <iterator>
#include <utility>

double last_seconds = 0, last_kernel = 0;
int last_tiles = 0;
bool show_hosts = true;

using Bytes = std::vector<unsigned char>;
struct Tile {
    int y, x, h, w, ey, ex, eh, ew;
};
int W, H, C;
std::string op;
void integer(std::istream &f, int &v) {
    while (true) {
        f >> std::ws;
        if (f.peek() != '#')
            break;
        std::string s;
        std::getline(f, s);
    }
    if (!(f >> v))
        throw std::runtime_error("Antet invalid");
}

Bytes read(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    std::string m;
    f >> m;
    C = m == "P6" ? 3 : m == "P5" ? 1 : 0;
    int maxv;
    integer(f, W);
    integer(f, H);
    integer(f, maxv);
    if (!C || W <= 0 || H <= 0 || maxv != 255 || 1LL * W * H * C > INT_MAX)
        throw std::runtime_error("Imagine invalida: foloseste P6/P5 maxval=255, counts MPI int");
    char sep;
    f.get(sep);
    if (sep == '\r' && f.peek() == '\n')
        f.get();
    Bytes a(size_t(W) * H * C);
    f.read((char *)a.data(), a.size());
    if (size_t(f.gcount()) != a.size())
        throw std::runtime_error("Imagine trunchiata");
    return a;
}

Bytes pack(const Bytes &a, const Tile &t) {
    Bytes b(size_t(t.eh) * t.ew * C);
    for (int y = 0; y < t.eh; y++)
        std::memcpy(b.data() + size_t(y) * t.ew * C, a.data() + (size_t(t.ey + y) * W + t.ex) * C,
                    size_t(t.ew) * C);
    return b;
}

Bytes calc(const Bytes &a, const Tile &t) {
    int oc = op == "gray" ? 1 : C;
    Bytes b(size_t(t.h) * t.w * oc);
    for (int y = 0; y < t.h; y++)
        for (int x = 0; x < t.w; x++) {
            size_t src = (size_t(t.y + y - t.ey) * t.ew + t.x + x - t.ex) * C;
            if (op == "gray")
                b[size_t(y) * t.w + x] =
                    (unsigned char)(.299 * a[src] + .587 * a[src + 1] + .114 * a[src + 2] + .5);
            else if (op == "rotate")
                for (int c = 0; c < C; c++)
                    b[(size_t(x) * t.h + t.h - 1 - y) * C + c] = a[src + c];
            else
                for (int c = 0; c < C; c++) {
                    int sum = 0, n = 0;
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++) {
                            int gy = t.y + y + dy, gx = t.x + x + dx;
                            if (gy < 0 || gy >= H || gx < 0 || gx >= W)
                                continue;
                            sum += a[(size_t(gy - t.ey) * t.ew + gx - t.ex) * C + c];
                            n++;
                        }
                    b[(size_t(y) * t.w + x) * C + c] = (unsigned char)((sum + n / 2) / n);
                }
        }
    return b;
}

void place(Bytes &out, const Bytes &b, const Tile &t) {
    int oc = op == "gray" ? 1 : C;
    if (op == "rotate")
        for (int y = 0; y < t.w; y++)
            std::memcpy(out.data() + (size_t(t.x + y) * H + H - t.y - t.h) * C,
                        b.data() + size_t(y) * t.h * C, size_t(t.h) * C);
    else
        for (int y = 0; y < t.h; y++)
            std::memcpy(out.data() + (size_t(t.y + y) * W + t.x) * oc,
                        b.data() + size_t(y) * t.w * oc, size_t(t.w) * oc);
}

int experiment(int argc, char **argv) {
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    try {
        if (argc != 9)
            throw std::runtime_error(
                "engine op input output seq|mpi 1d|2d static|dynamic chunks metadata");
        op = argv[1];
        std::string mode = argv[4], split = argv[5], sched = argv[6];
        int chunks = std::stoi(argv[7]);
        Bytes image;
        if (!rank) {
            image = read(argv[2]);
            if (op == "gray" && C != 3)
                throw std::runtime_error("Gray necesita P6");
        }
        int dims[3] = {W, H, C};
        MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
        W = dims[0];
        H = dims[1];
        C = dims[2];
        char host[MPI_MAX_PROCESSOR_NAME] = {0};
        int len;
        MPI_Get_processor_name(host, &len);
        std::vector<char> hosts(size_t(p) * MPI_MAX_PROCESSOR_NAME);
        MPI_Gather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hosts.data(), MPI_MAX_PROCESSOR_NAME,
                   MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime(), kernel = 0;
        Bytes output;
        std::vector<Tile> tiles;
        if (!rank) {
            output.resize(size_t(W) * H * (op == "gray" ? 1 : C));
            int gy = 1, gx = 1;
            if (mode != "seq") {
                if (split == "1d")
                    gy = std::min(H, chunks);
                else {
                    gy = std::min(H, std::max(1, int(std::sqrt(chunks))));
                    gx = std::min(W, (chunks + gy - 1) / gy);
                }
            }
            for (int iy = 0; iy < gy; iy++)
                for (int ix = 0; ix < gx; ix++) {
                    Tile t;
                    t.y = 1LL * iy * H / gy;
                    t.x = 1LL * ix * W / gx;
                    t.h = 1LL * (iy + 1) * H / gy - t.y;
                    t.w = 1LL * (ix + 1) * W / gx - t.x;
                    int halo = op == "blur";
                    t.ey = std::max(0, t.y - halo);
                    t.ex = std::max(0, t.x - halo);
                    t.eh = std::min(H, t.y + t.h + halo) - t.ey;
                    t.ew = std::min(W, t.x + t.w + halo) - t.ex;
                    tiles.push_back(t);
                }
        }
        auto local = [&](const Bytes &a, const Tile &t) {
            double s = MPI_Wtime();
            Bytes b = calc(a, t);
            kernel += MPI_Wtime() - s;
            return b;
        };
        if (mode == "seq" || p == 1) {
            if (!rank)
                for (auto &t : tiles) {
                    Bytes a = pack(image, t);
                    Bytes b = local(a, t);
                    place(output, b, t);
                }
        } else if (rank) {
            while (true) {
                int header[9];
                MPI_Recv(header, 9, MPI_INT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (header[0] < 0)
                    break;
                Tile t{header[1], header[2], header[3], header[4],
                       header[5], header[6], header[7], header[8]};
                Bytes a(size_t(t.eh) * t.ew * C);
                MPI_Recv(a.data(), a.size(), MPI_UNSIGNED_CHAR, 0, 11, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                Bytes b = local(a, t);
                MPI_Send(header, 1, MPI_INT, 0, 12, MPI_COMM_WORLD);
                MPI_Send(b.data(), b.size(), MPI_UNSIGNED_CHAR, 0, 13, MPI_COMM_WORLD);
            }
        } else {
            auto send = [&](int worker, int id) {
                const auto &t = tiles[id];
                int h[9] = {id, t.y, t.x, t.h, t.w, t.ey, t.ex, t.eh, t.ew};
                Bytes a = pack(image, t);
                MPI_Send(h, 9, MPI_INT, worker, 10, MPI_COMM_WORLD);
                MPI_Send(a.data(), a.size(), MPI_UNSIGNED_CHAR, worker, 11, MPI_COMM_WORLD);
            };
            auto recv = [&](int worker) {
                int id;
                MPI_Status status;
                MPI_Recv(&id, 1, MPI_INT, worker, 12, MPI_COMM_WORLD, &status);
                const auto &t = tiles[id];
                Bytes b(size_t(t.h) * t.w * (op == "gray" ? 1 : C));
                MPI_Recv(b.data(), b.size(), MPI_UNSIGNED_CHAR, status.MPI_SOURCE, 13,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                place(output, b, t);
                return status.MPI_SOURCE;
            };
            int n = tiles.size();
            if (sched == "static") {
                for (int base = 0; base < n; base += p - 1) {
                    int count = std::min(p - 1, n - base);
                    for (int i = 0; i < count; i++)
                        send(i + 1, base + i);
                    for (int i = 0; i < count; i++)
                        recv(i + 1);
                }
            } else {
                int next = 0, active = 0;
                for (int w = 1; w < p && next < n; w++) {
                    send(w, next++);
                    active++;
                }
                while (active) {
                    int w = recv(MPI_ANY_SOURCE);
                    active--;
                    if (next < n) {
                        send(w, next++);
                        active++;
                    }
                }
            }
            int stop[9] = {-1};
            for (int w = 1; w < p; w++)
                MPI_Send(stop, 9, MPI_INT, w, 10, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = MPI_Wtime() - start, maxelapsed, maxkernel;
        MPI_Reduce(&elapsed, &maxelapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&kernel, &maxkernel, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!rank) {
            int ow = op == "rotate" ? H : W, oh = op == "rotate" ? W : H, oc = op == "gray" ? 1 : C;
            std::ofstream f(argv[3], std::ios::binary);
            f << (oc == 1 ? "P5\n" : "P6\n") << ow << " " << oh << "\n255\n";
            f.write((char *)output.data(), output.size());
            if (!f)
                throw std::runtime_error("Scriere esuata");
            if (show_hosts) {
                std::cout << "Plasare procese (rank: calculator):\n";
                for (int i = 0; i < p; i++)
                    std::cout << i << ": " << hosts.data() + i * MPI_MAX_PROCESSOR_NAME << "\n";
                show_hosts = false;
            }
            last_seconds = maxelapsed;
            last_kernel = maxkernel;
            last_tiles = tiles.size();
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}

// Verificarea se face in afara intervalului cronometrat.
Bytes loadFile(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Nu pot citi rezultatul pentru verificare");
    return Bytes(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    try {
        if (argc < 3 || argc > 5) {
            if (!rank)
                std::cout << "Utilizare: mpirun -np P ./studiu input.ppm rezultat [repetari=5] "
                             "[blocuri=64]\n";
            MPI_Finalize();
            return 1;
        }
        int repeats = argc > 3 ? std::stoi(argv[3]) : 5;
        int chunks = argc > 4 ? std::stoi(argv[4]) : 64;
        if (repeats < 1 || chunks < 1)
            throw std::runtime_error("Repetarile si numarul de blocuri trebuie sa fie pozitive");
        std::string input = argv[1], prefix = argv[2];
        if (input == prefix + "_gray.pgm" || input == prefix + "_blur.ppm" ||
            input == prefix + "_rotate.ppm")
            throw std::runtime_error("Inputul nu poate fi si fisier de iesire");

        for (std::string operation : {"gray", "blur", "rotate"}) {
            Bytes reference;
            double sequentialMean = 0;
            std::string output = prefix + "_" + operation + (operation == "gray" ? ".pgm" : ".ppm");
            // Primul caz: secvential. Urmatoarele: cele patru variante MPI.
            for (int variant = 0; variant < 5; ++variant) {
                std::string mode = variant == 0 ? "seq" : "mpi";
                std::string split = variant <= 2 ? "1d" : "2d";
                std::string schedule = (variant == 2 || variant == 4) ? "dynamic" : "static";
                std::vector<double> times;
                double kernelSum = 0;
                if (!rank)
                    std::cout << "\nOperatie: " << operation << "; " << mode << "; " << split
                              << "; " << schedule << "\n";
                for (int repetition = 0; repetition <= repeats; ++repetition) {
                    // experiment() foloseste acelasi MPI deja initializat.
                    std::vector<std::string> args = {
                        "engine", operation, input,    output,
                        mode,     split,     schedule, std::to_string(chunks),
                        "unused"};
                    std::vector<char *> pointers;
                    for (auto &arg : args)
                        pointers.push_back(arg.data());
                    experiment(pointers.size(), pointers.data());
                    if (!rank) {
                        Bytes actual = loadFile(output);
                        if (variant == 0 && repetition == 0)
                            reference = std::move(actual);
                        else if (actual != reference)
                            throw std::runtime_error("Rezultat diferit de varianta secventiala");
                        if (repetition > 0) {
                            times.push_back(last_seconds);
                            kernelSum += last_kernel;
                            std::cout << std::setprecision(9) << "Rulare " << repetition << ": "
                                      << last_seconds << " s; kernel=" << last_kernel << " s\n";
                        }
                    }
                    // Verificarea si afisarea de la rank 0 se termina inaintea urmatoarei rulari.
                    MPI_Barrier(MPI_COMM_WORLD);
                }
                if (!rank) {
                    double mean = 0, squared = 0;
                    for (double value : times)
                        mean += value;
                    mean /= repeats;
                    for (double value : times)
                        squared += (value - mean) * (value - mean);
                    double deviation = repeats > 1 ? std::sqrt(squared / (repeats - 1)) : 0;
                    if (variant == 0)
                        sequentialMean = mean;
                    std::cout << "Rezolutie: " << W << "x" << H
                              << "; procese alocate jobului=" << processes
                              << "; procese active=" << (variant == 0 ? 1 : processes)
                              << "; blocuri=" << last_tiles << "\n"
                              << "MEDIA=" << mean << " s; deviatia standard=" << deviation
                              << " s; media kernel=" << kernelSum / repeats << " s\n";
                    if (variant != 0) {
                        double speedup = sequentialMean / mean;
                        std::cout << "Accelerare=" << speedup
                                  << "; eficienta=" << 100 * speedup / processes << "%\n";
                    }
                    std::cout << "Verificare rezultat: OK (incalzirea nu intra in medie)\n";
                }
            }
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Rank " << rank << ": " << error.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
