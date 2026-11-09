#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

// Accepta P6 (color, 3 bytes/pixel) si P5 (gri, 1 byte/pixel),
// ca sa putem da ca input si iesirea de la ./gray
static bool readImage(const std::string& path, int& width, int& height, int& channels,
                      std::vector<unsigned char>& pixels) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    std::string format;
    file >> format;
    if (format == "P6") {
        channels = 3;
    } else if (format == "P5") {
        channels = 1;
    } else {
        std::cerr << "Error: input must be binary PPM (P6) or PGM (P5)\n";
        return false;
    }

    int max_value = 0;
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
    // Modificam calculul pentru a evita overflow
    if (width > INT_MAX / channels ||
    height > INT_MAX / (width * channels)) {
        std::cerr << "Error: image too large for MPI int counts\n";
        return false;
    }

    file.get();  // un singur whitespace intre header si pixeli

    pixels.resize(static_cast<size_t>(width) * height * channels);
    file.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    if (file.gcount() != static_cast<std::streamsize>(pixels.size())) {
        std::cerr << "Error: file is truncated\n";
        return false;
    }
    return true;
}

// Scrie P6 daca e color, P5 daca e gri
static bool writeImage(const std::string& path, int width, int height, int channels,
                       const std::vector<unsigned char>& pixels) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    file << (channels == 1 ? "P5" : "P6") << "\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    return static_cast<bool>(file);
}

static double gaussianWeight(int row_offset, int col_offset, double sigma) {
    const double row_distance = row_offset;
    const double col_distance = col_offset;
    const double distance_squared = row_distance * row_distance + col_distance * col_distance;
    return std::exp(-distance_squared / (2.0 * sigma * sigma));
}

// strip_with_halo = fasia + halo_rows randuri halo de fiecare parte.
// Randul 'row' din fasie se afla la indexul row + halo_rows in strip_with_halo.
// first_image_row ne spune unde e fasia in imagine, ca sa stim unde sunt marginile reale.
static void blurStrip(const std::vector<unsigned char>& strip_with_halo, std::vector<unsigned char>& blurred_strip,
                      int width, int image_height, int channels, int first_image_row, int row_count,
                      int blur_radius, int halo_rows) {
    const size_t bytes_per_row = static_cast<size_t>(width) * channels;
    const double sigma = std::max(blur_radius / 2.0, 0.5);

    for (int row = 0; row < row_count; ++row) {
        const int image_row = first_image_row + row;

        for (int col = 0; col < width; ++col) {
            for (int channel = 0; channel < channels; ++channel) {
                double weighted_sum = 0.0, weight_sum = 0.0;

                // Calculam cat de sus si cat de jos putem merge pe randuri si coloane

                // ------ Randuri ------
                // Primiele (Limita stanga)
                const int first_row_offset = std::max(-blur_radius, -image_row);
                // Ultimele (Limita Dreapta)
                const int last_row_offset = std::min(blur_radius, image_height - 1 - image_row);
                
                
                // ------ Coloane ------
                // Primele (Limita sus)
                const int first_col_offset = std::max(-blur_radius, -col);
                // Ultimele (Limita jos)
                const int last_col_offset = std::min(blur_radius, width - 1 - col);


                for (int row_offset = first_row_offset; row_offset <= last_row_offset; ++row_offset) {
                    const size_t halo_row = row + halo_rows + row_offset;

                    for (int col_offset = first_col_offset; col_offset <= last_col_offset; ++col_offset) {
                        const int neighbor_col = col + col_offset;

                        const double weight = gaussianWeight(row_offset, col_offset, sigma);
                        weighted_sum += weight * strip_with_halo[halo_row * bytes_per_row + static_cast<size_t>(neighbor_col) * channels + channel];
                        weight_sum += weight;
                    }
                }

                blurred_strip[row * bytes_per_row + static_cast<size_t>(col) * channels + channel] =
                    static_cast<unsigned char>(std::lround(weighted_sum / weight_sum));
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int my_rank = 0, process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    char host_name[MPI_MAX_PROCESSOR_NAME];
    int host_name_length = 0;
    MPI_Get_processor_name(host_name, &host_name_length);

    if (argc != 3 && argc != 4) {
        if (my_rank == 0) std::cerr << "Usage: mpirun -np N ./blur <input.ppm|.pgm> <output> [blur-radius]\n";
        MPI_Finalize();
        return 1;
    }

    int blur_radius = 2;
    if (my_rank == 0 && argc == 4) {
        char* end = nullptr;
        errno = 0;
        const long parsed_radius = std::strtol(argv[3], &end, 10);
        if (errno == ERANGE || end == argv[3] || *end != '\0' || parsed_radius < 0 || parsed_radius > INT_MAX) {
            std::cerr << "Error: blur radius must be a non-negative integer\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        blur_radius = static_cast<int>(parsed_radius);
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    // doar rank 0 citeste fisierul
    int width = 0, height = 0, channels = 0;
    std::vector<unsigned char> full_image;
    if (my_rank == 0) {
        if (!readImage(input_path, width, height, channels, full_image)) {
            std::cerr << "Could not read " << input_path << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "[Rank 0 on " << host_name << "] Loaded " << input_path
                  << " (" << width << "x" << height << "), operation: Gaussian blur\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // restul proceselor afla dimensiunile de la rank 0
    int image_info[3] = {width, height, channels};
    MPI_Bcast(image_info, 3, MPI_INT, 0, MPI_COMM_WORLD);
    width = image_info[0];
    height = image_info[1];
    channels = image_info[2];
    MPI_Bcast(&blur_radius, 1, MPI_INT, 0, MPI_COMM_WORLD);
    blur_radius = std::min(blur_radius, std::max(width, height) - 1);
    const int halo_rows = std::min(blur_radius, height - 1);
    if (my_rank == 0) std::cout << "Using blur radius: " << blur_radius << "\n";

    // impartim randurile cat mai egal, primele procese iau cate un rand in plus
    std::vector<int> rows_per_process(process_count), first_row_of_process(process_count);
    const int min_rows = height / process_count;
    const int extra_rows = height % process_count;
    int next_row = 0;
    for (int p = 0; p < process_count; ++p) {
        rows_per_process[p] = min_rows + (p < extra_rows ? 1 : 0);
        first_row_of_process[p] = next_row;
        next_row += rows_per_process[p];
    }

    // acelasi lucru in bytes, asa cum le vrea Scatterv/Gatherv
    const int bytes_per_row = width * channels;
    std::vector<int> bytes_per_process(process_count), byte_offset_of_process(process_count);
    for (int p = 0; p < process_count; ++p) {
        bytes_per_process[p] = rows_per_process[p] * bytes_per_row;
        byte_offset_of_process[p] = first_row_of_process[p] * bytes_per_row;
    }

    const int my_row_count = rows_per_process[my_rank];
    const int my_first_row = first_row_of_process[my_rank];
    const int my_byte_count = bytes_per_process[my_rank];

    // fiecare proces primeste fasia lui de randuri
    std::vector<unsigned char> my_strip(my_byte_count);
    MPI_Scatterv(full_image.data(), bytes_per_process.data(), byte_offset_of_process.data(), MPI_UNSIGNED_CHAR,
                 my_strip.data(), my_byte_count, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    std::cout << "  -> Rank " << my_rank << " (" << host_name << ") processing "
              << my_row_count << " rows starting at row " << my_first_row << "\n";

    // Copiem fasia intre randurile halo necesare razei alese.
    std::vector<unsigned char> strip_with_halo(static_cast<size_t>(my_row_count + 2 * halo_rows) * bytes_per_row, 0);

    if (!my_strip.empty()) {
        std::memcpy(&strip_with_halo[static_cast<size_t>(halo_rows) * bytes_per_row],
                    my_strip.data(), my_strip.size());
    }

    // Cautam vecinii care au efectiv randuri, inclusiv daca unele procese au fasii goale.
    const bool has_rows = my_row_count > 0;
    int rank_above = MPI_PROC_NULL, rank_below = MPI_PROC_NULL;
    if (has_rows) {
        for (int p = my_rank - 1; p >= 0; --p) {
            if (rows_per_process[p] > 0) {
                rank_above = p;
                break;
            }
        }
        for (int p = my_rank + 1; p < process_count; ++p) {
            if (rows_per_process[p] > 0) {
                rank_below = p;
                break;
            }
        }
    }

    // Schimbam cate un rand pe nivel
    // randurile halo primite sunt retransmise
    // la nivelul urmator, astfel raza poate traversa fasii de orice dimensiune.

    // Optiunea aceasta e in teorie mai rapida 
    // deoarece fiecare proces are grija sa imparta ce este necesar
    // Optiunea 2 este sa triimitem cu Scatterv, dar asta ar fi o actiune mai mare facut doar de un process la Rank 0
    for (int depth = 1; has_rows && depth <= halo_rows; ++depth) {
        const size_t own_start = static_cast<size_t>(halo_rows);
        unsigned char* row_to_below = &strip_with_halo[(own_start + static_cast<size_t>(my_row_count) - depth) * bytes_per_row];
        unsigned char* top_halo = &strip_with_halo[(own_start - depth) * bytes_per_row];
        MPI_Sendrecv(row_to_below, bytes_per_row, MPI_UNSIGNED_CHAR, rank_below, 0,
                     top_halo, bytes_per_row, MPI_UNSIGNED_CHAR, rank_above, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        unsigned char* row_to_above = &strip_with_halo[(own_start + depth - 1) * bytes_per_row];
        unsigned char* bottom_halo = &strip_with_halo[(own_start + my_row_count - 1 + depth) * bytes_per_row];
        MPI_Sendrecv(row_to_above, bytes_per_row, MPI_UNSIGNED_CHAR, rank_above, 1,
                     bottom_halo, bytes_per_row, MPI_UNSIGNED_CHAR, rank_below, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // rezultatul se scrie peste my_strip; citim doar din strip_with_halo
    blurStrip(strip_with_halo, my_strip, width, height, channels, my_first_row, my_row_count,
              blur_radius, halo_rows);

    // blur-ul nu schimba dimensiunile, deci adunam fasiile exact cum le-am impartit
    std::vector<unsigned char> output_image;
    if (my_rank == 0) output_image.resize(static_cast<size_t>(width) * height * channels);

    MPI_Gatherv(my_strip.data(), my_byte_count, MPI_UNSIGNED_CHAR,
                output_image.data(), bytes_per_process.data(), byte_offset_of_process.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        if (writeImage(output_path, width, height, channels, output_image))
            std::cout << "[Rank 0] Saved " << output_path << " (" << width << "x" << height << ")\n";
        else
            std::cerr << "[Rank 0] Could not write " << output_path << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    if (my_rank == 0)
        std::cout << "Total parallel execution time: " << (end_time - start_time) << " seconds.\n";

    MPI_Finalize();
    return 0;
}