// studiu.cpp
// Program MPI pentru studiul de performanta al celor trei operatii (gray, blur, rotate).
//
// Nu face o operatie noua: foloseste aceleasi formule ca ../gray.cpp, ../blur.cpp si
// ../rotate.cpp, dar imparte munca in mai multe moduri si masoara timpul fiecaruia:
//   - secvential: un singur proces calculeaza toata imaginea
//   - MPI, impartire 1D (fasii orizontale) sau 2D (grila de dreptunghiuri)
//   - MPI, load-balancing static (blocurile sunt impartite dinainte)
//            sau dinamic (blocul urmator merge la primul worker liber)
//
// Compilare: mpic++ -O3 -std=c++17 studiu.cpp -o studiu
// Rulare:    mpirun -np N ./studiu <input.ppm> <prefix_iesire> [repetari=5] [blocuri=64] [raza_blur=2]
// Cluster:   mpirun --hostfile hosts --map-by node -np N ./studiu input.ppm iesiri/rezultat
//
// La N >= 2, rank 0 este coordonatorul (trimite blocurile si reasambleaza imaginea),
// iar rank-urile 1..N-1 sunt workerii (calculeaza blocurile). La N = 1 rank 0 face tot.
#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

// ============================ Imagine ============================

// Pixelii sunt stocati rand cu rand, cate `channels` bytes pe pixel (3 = RGB, 1 = gri).
struct Image {
    int width = 0, height = 0, channels = 0;
    std::vector<unsigned char> pixels;

    // pozitia in `pixels` a primului byte al pixelului (row, col)
    size_t index(int row, int col) const { return (static_cast<size_t>(row) * width + col) * channels; }
};

static Image blankImage(int width, int height, int channels) {
    Image image;
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.pixels.resize(static_cast<size_t>(width) * height * channels);
    return image;
}

// Citeste urmatorul numar din header, sarind peste spatii si comentarii (#)
static bool readHeaderInt(std::istream& in, int& value) {
    while (true) {
        in >> std::ws;
        if (in.peek() != '#') break;
        std::string comment_line;
        std::getline(in, comment_line);
    }
    return static_cast<bool>(in >> value);
}

// Accepta doar P6 (color, 3 bytes/pixel), pentru ca gray are nevoie de R, G, B
static bool readImage(const std::string& path, Image& image) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    std::string format;
    file >> format;
    if (format != "P6") {
        std::cerr << "Error: input must be binary PPM (P6)\n";
        return false;
    }

    int width = 0, height = 0, max_value = 0;
    if (!readHeaderInt(file, width) || !readHeaderInt(file, height) || !readHeaderInt(file, max_value)) {
        std::cerr << "Error: malformed header\n";
        return false;
    }
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: invalid image size\n";
        return false;
    }
    if (max_value != 255) {
        std::cerr << "Error: only 8-bit images are supported\n";
        return false;
    }
    // MPI numara elementele in int
    if (static_cast<long long>(width) * height * 3 > INT_MAX) {
        std::cerr << "Error: image too large for MPI int counts\n";
        return false;
    }

    file.get();  // un singur whitespace intre header si pixeli

    image = blankImage(width, height, 3);
    file.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    if (file.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
        std::cerr << "Error: file is truncated\n";
        return false;
    }
    return true;
}

// Scrie P6 daca e color, P5 daca e gri
static bool writeImage(const std::string& path, const Image& image) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    file << (image.channels == 1 ? "P5" : "P6") << "\n" << image.width << " " << image.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    return static_cast<bool>(file);
}

// ============================ Experimentul ============================

// Tot ce descrie varianta testata acum. Toate procesele au aceeasi copie.
struct Experiment {
    std::string operation;        // gray | blur | rotate
    bool sequential = false;      // true = un singur proces, toata imaginea
    std::string split;            // 1d | 2d
    std::string schedule;         // static | dynamic
    int block_count = 64;         // cate blocuri vrem (la 2D grila poate rotunji putin in sus)
    int blur_radius = 2;
    std::vector<double> weights;  // ponderile gaussiene, (2r+1) x (2r+1)
};

// ============================ Blocuri ============================

// Un dreptunghi din imagine
struct Rect {
    int first_row, first_col;  // coltul stanga-sus
    int row_count, col_count;  // dimensiunile
};

// Un bloc = dreptunghiul pe care il calculeaza un worker (area).
// La blur fiecare pixel are nevoie de vecinii pana la distanta `raza`, deci worker-ul
// primeste un dreptunghi putin mai mare: blocul + o margine (with_halo).
// La gray si rotate nu e nevoie de vecini, deci with_halo == area.
struct Block {
    Rect area;
    Rect with_halo;
};

// Imparte imaginea in blocuri:
//   secvential: 1 bloc (toata imaginea)
//   1D:         block_count fasii orizontale
//   2D:         grila de aproximativ sqrt(block_count) x sqrt(block_count) dreptunghiuri
// Coordonatorul si workerii apeleaza aceeasi functie, deci au aceeasi lista de blocuri.
static std::vector<Block> splitIntoBlocks(int width, int height, const Experiment& experiment) {
    int block_rows = 1, block_cols = 1;
    if (!experiment.sequential && experiment.split == "1d") {
        block_rows = std::min(height, experiment.block_count);
    } else if (!experiment.sequential) {
        block_rows = std::min(height, std::max(1, static_cast<int>(std::sqrt(experiment.block_count))));
        block_cols = std::min(width, (experiment.block_count + block_rows - 1) / block_rows);
    }
    const int halo = experiment.operation == "blur" ? experiment.blur_radius : 0;

    // impartim randurile si coloanele cat mai egal
    std::vector<Block> blocks;
    for (int block_row = 0; block_row < block_rows; ++block_row) {
        for (int block_col = 0; block_col < block_cols; ++block_col) {
            Rect area;
            area.first_row = static_cast<long long>(block_row) * height / block_rows;
            area.first_col = static_cast<long long>(block_col) * width / block_cols;
            area.row_count = static_cast<long long>(block_row + 1) * height / block_rows - area.first_row;
            area.col_count = static_cast<long long>(block_col + 1) * width / block_cols - area.first_col;

            // marginea nu iese din imagine
            Rect with_halo;
            with_halo.first_row = std::max(0, area.first_row - halo);
            with_halo.first_col = std::max(0, area.first_col - halo);
            with_halo.row_count = std::min(height, area.first_row + area.row_count + halo) - with_halo.first_row;
            with_halo.col_count = std::min(width, area.first_col + area.col_count + halo) - with_halo.first_col;

            blocks.push_back({area, with_halo});
        }
    }
    return blocks;
}

// Decupeaza din imagine dreptunghiul `rect`, ca o imagine mai mica
static Image cutRect(const Image& image, const Rect& rect) {
    Image piece = blankImage(rect.col_count, rect.row_count, image.channels);
    const size_t row_bytes = static_cast<size_t>(piece.width) * piece.channels;
    for (int row = 0; row < piece.height; ++row)
        std::memcpy(&piece.pixels[piece.index(row, 0)],
                    &image.pixels[image.index(rect.first_row + row, rect.first_col)], row_bytes);
    return piece;
}

// Lipeste `piece` in `image`, cu coltul stanga-sus in (first_row, first_col)
static void pasteRect(Image& image, const Image& piece, int first_row, int first_col) {
    const size_t row_bytes = static_cast<size_t>(piece.width) * piece.channels;
    for (int row = 0; row < piece.height; ++row)
        std::memcpy(&image.pixels[image.index(first_row + row, first_col)],
                    &piece.pixels[piece.index(row, 0)], row_bytes);
}

// Cum arata rezultatul unei operatii aplicate pe un dreptunghi de row_count x col_count:
// gray -> 1 canal; rotate -> latimea si inaltimea se inverseaza; blur -> la fel ca inputul.
static Image blankResult(const std::string& operation, int row_count, int col_count, int channels) {
    if (operation == "gray") return blankImage(col_count, row_count, 1);
    if (operation == "rotate") return blankImage(row_count, col_count, channels);
    return blankImage(col_count, row_count, channels);
}

// Pune rezultatul unui bloc la locul lui in imaginea finala
static void pasteResult(Image& output, const Image& result, const Block& block, const std::string& operation) {
    if (operation == "rotate") {
        // dupa rotire: coloana blocului devine rand, iar blocurile de sus ajung in dreapta
        pasteRect(output, result, block.area.first_col, output.width - block.area.first_row - block.area.row_count);
    } else {
        pasteRect(output, result, block.area.first_row, block.area.first_col);
    }
}

// ============================ Cele trei operatii, pe un bloc ============================
// `piece` = pixelii blocului cu margine (block.with_halo), decupati din imagine.
// Rezultatul are dimensiunile blocului (block.area).

// Aceeasi formula ca in gray.cpp: ochiul e mai sensibil la verde.
static Image grayBlock(const Image& piece) {
    Image gray = blankImage(piece.width, piece.height, 1);  // la gray nu exista margine
    for (int row = 0; row < piece.height; ++row) {
        for (int col = 0; col < piece.width; ++col) {
            const size_t source = piece.index(row, col);
            const double r = piece.pixels[source + 0];
            const double g = piece.pixels[source + 1];
            const double b = piece.pixels[source + 2];
            gray.pixels[gray.index(row, col)] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b + 0.5);
        }
    }
    return gray;
}

// Aceeasi rotire ca in rotate.cpp: 90 de grade in sensul acelor de ceasornic.
// Pixelul (row, col) ajunge pe randul col, coloana (height - 1 - row).
static Image rotateBlock(const Image& piece) {
    Image rotated = blankImage(piece.height, piece.width, piece.channels);  // latime <-> inaltime
    for (int row = 0; row < piece.height; ++row) {
        for (int col = 0; col < piece.width; ++col) {
            const size_t source = piece.index(row, col);
            const size_t target = rotated.index(col, piece.height - 1 - row);
            for (int channel = 0; channel < piece.channels; ++channel)
                rotated.pixels[target + channel] = piece.pixels[source + channel];
        }
    }
    return rotated;
}

// Aceeasi pondere ca in blur.cpp
static double gaussianWeight(int row_offset, int col_offset, double sigma) {
    const double row_distance = row_offset;
    const double col_distance = col_offset;
    const double distance_squared = row_distance * row_distance + col_distance * col_distance;
    return std::exp(-distance_squared / (2.0 * sigma * sigma));
}

// Acelasi blur gaussian ca in blur.cpp. Singura diferenta: ponderile sunt calculate
// o singura data (experiment.weights), nu la fiecare pixel; valorile sunt identice.
static Image blurBlock(const Image& piece, const Block& block, int image_width, int image_height,
                       const Experiment& experiment) {
    const int radius = experiment.blur_radius;
    const int weights_per_row = 2 * radius + 1;
    Image blurred = blankImage(block.area.col_count, block.area.row_count, piece.channels);

    for (int row = 0; row < block.area.row_count; ++row) {
        const int image_row = block.area.first_row + row;
        // cat de sus / jos putem merge fara sa iesim din imagine
        const int first_row_offset = std::max(-radius, -image_row);
        const int last_row_offset = std::min(radius, image_height - 1 - image_row);

        for (int col = 0; col < block.area.col_count; ++col) {
            const int image_col = block.area.first_col + col;
            // cat de la stanga / dreapta putem merge fara sa iesim din imagine
            const int first_col_offset = std::max(-radius, -image_col);
            const int last_col_offset = std::min(radius, image_width - 1 - image_col);

            for (int channel = 0; channel < piece.channels; ++channel) {
                double weighted_sum = 0.0, weight_sum = 0.0;

                for (int row_offset = first_row_offset; row_offset <= last_row_offset; ++row_offset) {
                    for (int col_offset = first_col_offset; col_offset <= last_col_offset; ++col_offset) {
                        // vecinul (image_row + row_offset, image_col + col_offset), cautat in piece
                        const int piece_row = image_row + row_offset - block.with_halo.first_row;
                        const int piece_col = image_col + col_offset - block.with_halo.first_col;
                        const double weight = experiment.weights[(row_offset + radius) * weights_per_row + col_offset + radius];
                        weighted_sum += weight * piece.pixels[piece.index(piece_row, piece_col) + channel];
                        weight_sum += weight;
                    }
                }

                blurred.pixels[blurred.index(row, col) + channel] =
                    static_cast<unsigned char>(std::lround(weighted_sum / weight_sum));
            }
        }
    }
    return blurred;
}

// Aplica operatia experimentului pe un bloc; adauga timpul de calcul la compute_seconds
static Image processBlock(const Image& piece, const Block& block, int image_width, int image_height,
                          const Experiment& experiment, double& compute_seconds) {
    const double start = MPI_Wtime();
    Image result;
    if (experiment.operation == "gray") result = grayBlock(piece);
    else if (experiment.operation == "rotate") result = rotateBlock(piece);
    else result = blurBlock(piece, block, image_width, image_height, experiment);
    compute_seconds += MPI_Wtime() - start;
    return result;
}

// ============================ Cine face ce ============================
// Comunicarea coordonator <-> worker este un singur mesaj in fiecare sens, iar
// TAG-UL MESAJULUI ESTE INDEXUL BLOCULUI. Nu trimitem coordonatele blocului, pentru ca
// workerii au aceeasi lista de blocuri (splitIntoBlocks).
//   coordonator -> worker: pixelii blocului cu margine   (tag = indexul blocului)
//   worker -> coordonator: pixelii calculati             (tag = indexul blocului)
//   coordonator -> worker: mesaj gol cu tag STOP_TAG      = "nu mai sunt blocuri"
const int STOP_TAG = 30000;
const int MAX_BLOCKS = 10000;  // ca indexul blocului sa nu ajunga la STOP_TAG

// Un singur proces face tot: varianta secventiala sau N = 1
static Image runOnOneProcess(const Image& image, const Experiment& experiment, double& compute_seconds) {
    Image output = blankResult(experiment.operation, image.height, image.width, image.channels);
    for (const Block& block : splitIntoBlocks(image.width, image.height, experiment)) {
        const Image piece = cutRect(image, block.with_halo);
        const Image result = processBlock(piece, block, image.width, image.height, experiment, compute_seconds);
        pasteResult(output, result, block, experiment.operation);
    }
    return output;
}

// Worker: primeste blocuri, le calculeaza si le trimite inapoi, pana la STOP_TAG
static void runWorker(const Image& image, const Experiment& experiment, double& compute_seconds) {
    const std::vector<Block> blocks = splitIntoBlocks(image.width, image.height, experiment);
    while (true) {
        // ne uitam la tag-ul mesajului urmator ca sa stim ce bloc vine
        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == STOP_TAG) {
            MPI_Recv(nullptr, 0, MPI_UNSIGNED_CHAR, 0, STOP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        }
        const int block_index = status.MPI_TAG;
        const Block& block = blocks[block_index];

        Image piece = blankImage(block.with_halo.col_count, block.with_halo.row_count, image.channels);
        MPI_Recv(piece.pixels.data(), static_cast<int>(piece.pixels.size()), MPI_UNSIGNED_CHAR, 0, block_index,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        const Image result = processBlock(piece, block, image.width, image.height, experiment, compute_seconds);
        MPI_Send(result.pixels.data(), static_cast<int>(result.pixels.size()), MPI_UNSIGNED_CHAR, 0, block_index,
                 MPI_COMM_WORLD);
    }
}

// Coordonator (rank 0): trimite blocurile la workeri, static sau dinamic, si reasambleaza imaginea
static Image runCoordinator(const Image& image, const Experiment& experiment, int worker_count) {
    const std::vector<Block> blocks = splitIntoBlocks(image.width, image.height, experiment);
    const int total_blocks = static_cast<int>(blocks.size());
    Image output = blankResult(experiment.operation, image.height, image.width, image.channels);

    auto sendBlock = [&](int worker, int block_index) {
        const Image piece = cutRect(image, blocks[block_index].with_halo);
        MPI_Send(piece.pixels.data(), static_cast<int>(piece.pixels.size()), MPI_UNSIGNED_CHAR, worker, block_index,
                 MPI_COMM_WORLD);
    };

    // primeste un bloc calculat de la `worker` (sau de la oricine, cu MPI_ANY_SOURCE),
    // il pune in imagine si intoarce worker-ul care a raspuns
    auto receiveBlock = [&](int worker) {
        MPI_Status status;
        MPI_Probe(worker, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        const Block& block = blocks[status.MPI_TAG];
        Image result = blankResult(experiment.operation, block.area.row_count, block.area.col_count, image.channels);
        MPI_Recv(result.pixels.data(), static_cast<int>(result.pixels.size()), MPI_UNSIGNED_CHAR, status.MPI_SOURCE,
                 status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pasteResult(output, result, block, experiment.operation);
        return status.MPI_SOURCE;
    };

    if (experiment.schedule == "static") {
        // STATIC: blocul i merge mereu la worker-ul 1 + (i % worker_count), stabilit dinainte.
        // Lucram pe runde: cate un bloc la fiecare worker, apoi asteptam toate rezultatele.
        // Un worker rapid asteapta dupa cel mai lent din runda.
        for (int round_start = 0; round_start < total_blocks; round_start += worker_count) {
            const int blocks_in_round = std::min(worker_count, total_blocks - round_start);
            for (int i = 0; i < blocks_in_round; ++i) sendBlock(1 + i, round_start + i);
            for (int i = 0; i < blocks_in_round; ++i) receiveBlock(1 + i);
        }
    } else {
        // DINAMIC: dam cate un bloc fiecarui worker, apoi blocul urmator merge
        // la primul worker care termina. Nimeni nu sta degeaba.
        int next_block = 0, blocks_in_progress = 0;
        for (int worker = 1; worker <= worker_count && next_block < total_blocks; ++worker) {
            sendBlock(worker, next_block++);
            ++blocks_in_progress;
        }
        while (blocks_in_progress > 0) {
            const int free_worker = receiveBlock(MPI_ANY_SOURCE);
            --blocks_in_progress;
            if (next_block < total_blocks) {
                sendBlock(free_worker, next_block++);
                ++blocks_in_progress;
            }
        }
    }

    for (int worker = 1; worker <= worker_count; ++worker)
        MPI_Send(nullptr, 0, MPI_UNSIGNED_CHAR, worker, STOP_TAG, MPI_COMM_WORLD);
    return output;
}

// ============================ Masurare ============================

struct Measurement {
    double total_seconds = 0.0;    // de la start pana au terminat toate procesele
    double compute_seconds = 0.0;  // doar calculul (fara comunicare); maximul dintre procese
};

// O rulare cronometrata a experimentului. Pe rank 0, `output` primeste imaginea calculata.
static Measurement measureOnce(const Image& image, const Experiment& experiment, Image& output) {
    int my_rank = 0, process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    double my_compute_seconds = 0.0;

    // pornim cronometrul cand toate procesele sunt pregatite
    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    if (experiment.sequential || process_count == 1) {
        if (my_rank == 0) output = runOnOneProcess(image, experiment, my_compute_seconds);
    } else if (my_rank == 0) {
        output = runCoordinator(image, experiment, process_count - 1);
    } else {
        runWorker(image, experiment, my_compute_seconds);
    }

    // oprim cronometrul cand au terminat toate procesele
    MPI_Barrier(MPI_COMM_WORLD);
    const double my_seconds = MPI_Wtime() - start_time;

    Measurement measurement;
    MPI_Reduce(&my_seconds, &measurement.total_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_compute_seconds, &measurement.compute_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return measurement;
}

struct Statistics {
    double mean = 0.0;
    double deviation = 0.0;  // deviatia standard de selectie
};

static Statistics computeStatistics(const std::vector<double>& values) {
    Statistics statistics;
    for (double value : values) statistics.mean += value;
    statistics.mean /= values.size();
    double squared_sum = 0.0;
    for (double value : values) squared_sum += (value - statistics.mean) * (value - statistics.mean);
    if (values.size() > 1) statistics.deviation = std::sqrt(squared_sum / (values.size() - 1));
    return statistics;
}

// ============================ Main ============================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int my_rank = 0, process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    char host_name[MPI_MAX_PROCESSOR_NAME] = {0};
    int host_name_length = 0;
    MPI_Get_processor_name(host_name, &host_name_length);

    if (argc < 3 || argc > 6) {
        if (my_rank == 0)
            std::cerr << "Usage: mpirun -np N ./studiu <input.ppm> <prefix_iesire> [repetari=5] [blocuri=64] [raza_blur=2]\n";
        MPI_Finalize();
        return 1;
    }
    const std::string input_path = argv[1];
    const std::string output_prefix = argv[2];
    const int repetitions = argc > 3 ? std::atoi(argv[3]) : 5;
    const int block_count = argc > 4 ? std::atoi(argv[4]) : 64;
    const int blur_radius = argc > 5 ? std::atoi(argv[5]) : 2;
    if (repetitions < 1 || block_count < 1 || block_count > MAX_BLOCKS || blur_radius < 0) {
        if (my_rank == 0) std::cerr << "Error: repetari >= 1, 1 <= blocuri <= " << MAX_BLOCKS << ", raza_blur >= 0\n";
        MPI_Finalize();
        return 1;
    }

    // doar rank 0 citeste fisierul
    Image image;
    if (my_rank == 0) {
        if (!readImage(input_path, image)) {
            std::cerr << "Could not read " << input_path << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "[Rank 0 on " << host_name << "] Loaded " << input_path << " (" << image.width << "x"
                  << image.height << "), " << process_count << " procese\n";
    }

    // restul proceselor afla dimensiunile de la rank 0 (pixelii ii primesc pe blocuri)
    int image_info[3] = {image.width, image.height, image.channels};
    MPI_Bcast(image_info, 3, MPI_INT, 0, MPI_COMM_WORLD);
    image.width = image_info[0];
    image.height = image_info[1];
    image.channels = image_info[2];

    // rank 0 afla pe ce calculatoare ruleaza procesele (ca sa verificam ca folosim ambele)
    std::vector<char> all_host_names(static_cast<size_t>(process_count) * MPI_MAX_PROCESSOR_NAME);
    MPI_Gather(host_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_host_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               0, MPI_COMM_WORLD);
    int computer_count = 0;
    if (my_rank == 0) {
        std::set<std::string> computers;
        std::cout << "Plasare procese (rank: calculator):\n";
        for (int p = 0; p < process_count; ++p) {
            const std::string name = &all_host_names[static_cast<size_t>(p) * MPI_MAX_PROCESSOR_NAME];
            std::cout << "  " << p << ": " << name << "\n";
            computers.insert(name);
        }
        computer_count = static_cast<int>(computers.size());
    }

    // ponderile gaussiene, calculate o singura data (aceeasi sigma ca in blur.cpp)
    std::vector<double> weights;
    const double sigma = std::max(blur_radius / 2.0, 0.5);
    for (int row_offset = -blur_radius; row_offset <= blur_radius; ++row_offset)
        for (int col_offset = -blur_radius; col_offset <= blur_radius; ++col_offset)
            weights.push_back(gaussianWeight(row_offset, col_offset, sigma));

    // cele 5 variante testate pentru fiecare operatie: {secvential, impartire, balansare}
    struct Variant { bool sequential; std::string split, schedule; };
    const std::vector<Variant> variants = {{true, "-", "-"},
                                           {false, "1d", "static"}, {false, "1d", "dynamic"},
                                           {false, "2d", "static"}, {false, "2d", "dynamic"}};

    for (const std::string operation : {"gray", "blur", "rotate"}) {
        Image sequential_result;  // referinta pentru verificare

        for (const Variant& variant : variants) {
            Experiment experiment;
            experiment.operation = operation;
            experiment.sequential = variant.sequential;
            experiment.split = variant.split;
            experiment.schedule = variant.schedule;
            experiment.block_count = block_count;
            experiment.blur_radius = blur_radius;
            experiment.weights = weights;
            const int used_blocks = static_cast<int>(splitIntoBlocks(image.width, image.height, experiment).size());

            if (my_rank == 0)
                std::cout << "\nOperatie: " << operation << "; "
                          << (variant.sequential ? "secvential" : "MPI " + variant.split + " " + variant.schedule) << "\n";

            // rularea 0 = incalzire (nu intra in medie), apoi `repetitions` rulari masurate
            std::vector<double> total_times, compute_times;
            for (int run = 0; run <= repetitions; ++run) {
                Image output;
                const Measurement measurement = measureOnce(image, experiment, output);
                if (my_rank != 0) continue;

                // fiecare varianta MPI trebuie sa dea exact aceeasi imagine ca secventialul
                if (variant.sequential && run == 0) {
                    sequential_result = output;
                } else if (output.pixels != sequential_result.pixels) {
                    std::cerr << "Error: rezultat diferit de varianta secventiala\n";
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (run == 0) continue;

                total_times.push_back(measurement.total_seconds);
                compute_times.push_back(measurement.compute_seconds);
                std::cout << std::setprecision(9) << "  Rulare " << run << ": " << measurement.total_seconds
                          << " s; calcul=" << measurement.compute_seconds << " s\n";
            }
            if (my_rank != 0) continue;

            const Statistics total = computeStatistics(total_times);
            const Statistics compute = computeStatistics(compute_times);
            std::cout << "  MEDIA=" << total.mean << " s; deviatie standard=" << total.deviation
                      << " s; calcul mediu=" << compute.mean << " s; blocuri=" << used_blocks << "; verificare: OK\n";

            // linia citita de analiza.py:
            // operatie,latime,inaltime,procese,calculatoare,mod,divizare,balansare,
            // blocuri,raza,repetari,medie_s,deviatie_s,calcul_s
            std::cout << "CSV," << operation << "," << image.width << "," << image.height << "," << process_count
                      << "," << computer_count << "," << (variant.sequential ? "seq" : "mpi") << "," << variant.split
                      << "," << variant.schedule << "," << used_blocks << "," << (operation == "blur" ? blur_radius : 0)
                      << "," << repetitions << "," << total.mean << "," << total.deviation << "," << compute.mean
                      << "\n";
        }

        // salvam o data rezultatul secvential, ca sa se poata verifica vizual
        if (my_rank == 0) {
            const std::string extension = operation == "gray" ? ".pgm" : ".ppm";
            const std::string output_path = output_prefix + "_" + operation + extension;
            if (!writeImage(output_path, sequential_result))
                std::cerr << "[Rank 0] Could not write " << output_path << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
