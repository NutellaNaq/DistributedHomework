#include <mpi.h>

#include <climits>
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
    if (max_value <= 0 || max_value > 255) {
        std::cerr << "Error: only 8-bit images are supported\n";
        return false;
    }
    // MPI numara elementele in int
    if (static_cast<long long>(width) * height * channels > INT_MAX) {
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

// strip_with_halo = fasia + un rand halo deasupra (index 0) si unul dedesubt.
// Randul 'row' din fasie se afla deci la indexul row + 1 in strip_with_halo.
// first_image_row ne spune unde e fasia in imagine, ca sa stim unde sunt marginile reale.
static void blurStrip(const std::vector<unsigned char>& strip_with_halo, std::vector<unsigned char>& blurred_strip,
                      int width, int image_height, int channels, int first_image_row, int row_count) {
    const size_t bytes_per_row = static_cast<size_t>(width) * channels;

    for (int row = 0; row < row_count; ++row) {
        const int image_row = first_image_row + row;

        for (int col = 0; col < width; ++col) {
            for (int channel = 0; channel < channels; ++channel) {
                int neighbor_sum = 0, neighbor_count = 0;

                for (int row_offset = -1; row_offset <= 1; ++row_offset) {
                    // la marginea de sus/jos a imaginii nu exista vecini
                    // (halo-ul de acolo e doar zerouri)
                    const int neighbor_image_row = image_row + row_offset;
                    if (neighbor_image_row < 0 || neighbor_image_row >= image_height) continue;

                    const size_t halo_row = row + 1 + row_offset;

                    for (int col_offset = -1; col_offset <= 1; ++col_offset) {
                        const int neighbor_col = col + col_offset;
                        if (neighbor_col < 0 || neighbor_col >= width) continue;

                        neighbor_sum += strip_with_halo[halo_row * bytes_per_row
                                                        + static_cast<size_t>(neighbor_col) * channels + channel];
                        ++neighbor_count;
                    }
                }

                // pe margini sunt mai putin de 9 vecini, de asta impartim la neighbor_count
                // (+ neighbor_count / 2 face rotunjire in loc de trunchiere)
                blurred_strip[row * bytes_per_row + static_cast<size_t>(col) * channels + channel] =
                    static_cast<unsigned char>((neighbor_sum + neighbor_count / 2) / neighbor_count);
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

    if (argc != 3) {
        if (my_rank == 0) std::cerr << "Usage: mpirun -np N ./blur <input.ppm|.pgm> <output>\n";
        MPI_Finalize();
        return 1;
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
                  << " (" << width << "x" << height << "), operation: blur\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // restul proceselor afla dimensiunile de la rank 0
    int image_info[3] = {width, height, channels};
    MPI_Bcast(image_info, 3, MPI_INT, 0, MPI_COMM_WORLD);
    width = image_info[0];
    height = image_info[1];
    channels = image_info[2];

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

    // copiem fasia intre cele doua randuri halo
    std::vector<unsigned char> strip_with_halo(static_cast<size_t>(my_row_count + 2) * bytes_per_row, 0);
    if (!my_strip.empty()) std::memcpy(&strip_with_halo[bytes_per_row], my_strip.data(), my_strip.size());

    unsigned char* top_halo       = &strip_with_halo[0];
    unsigned char* first_own_row  = &strip_with_halo[bytes_per_row];
    unsigned char* last_own_row   = &strip_with_halo[static_cast<size_t>(my_row_count) * bytes_per_row];
    unsigned char* bottom_halo    = &strip_with_halo[static_cast<size_t>(my_row_count + 1) * bytes_per_row];

    // prima si ultima fasie nu au vecin; cu MPI_PROC_NULL comunicarea nu face nimic
    const bool has_rows = my_row_count > 0;
    const int rank_above = (has_rows && my_first_row > 0) ? my_rank - 1 : MPI_PROC_NULL;
    const int rank_below = (has_rows && my_first_row + my_row_count < height) ? my_rank + 1 : MPI_PROC_NULL;

    // ultimul meu rand -> vecinul de jos; primesc halo-ul de sus de la vecinul de sus
    MPI_Sendrecv(last_own_row, bytes_per_row, MPI_UNSIGNED_CHAR, rank_below, 0,
                 top_halo, bytes_per_row, MPI_UNSIGNED_CHAR, rank_above, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // primul meu rand -> vecinul de sus; primesc halo-ul de jos de la vecinul de jos
    MPI_Sendrecv(first_own_row, bytes_per_row, MPI_UNSIGNED_CHAR, rank_above, 1,
                 bottom_halo, bytes_per_row, MPI_UNSIGNED_CHAR, rank_below, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // rezultatul se scrie peste my_strip; citim doar din strip_with_halo
    blurStrip(strip_with_halo, my_strip, width, height, channels, my_first_row, my_row_count);

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