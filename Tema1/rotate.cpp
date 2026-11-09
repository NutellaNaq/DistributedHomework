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

// Fasia are row_count randuri si width coloane; dupa rotire are width randuri si row_count coloane.
// Pixelul (rand r, coloana c) ajunge la (rand c, coloana row_count - 1 - r).
static void rotateStrip(const std::vector<unsigned char>& strip, std::vector<unsigned char>& rotated_strip,
                        int width, int channels, int row_count) {
    for (int row = 0; row < row_count; ++row) {
        for (int col = 0; col < width; ++col) {
            const size_t source = (static_cast<size_t>(row) * width + col) * channels;
            const size_t target = (static_cast<size_t>(col) * row_count + (row_count - 1 - row)) * channels;
            for (int channel = 0; channel < channels; ++channel)
                rotated_strip[target + channel] = strip[source + channel];
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
        if (my_rank == 0) std::cerr << "Usage: mpirun -np N ./rotate <input.ppm|.pgm> <output>\n";
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
                  << " (" << width << "x" << height << "), operation: rotate\n";
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

    // rotirea nu are nevoie de vecini, deci nu e nevoie de comunicare intre procese
    std::vector<unsigned char> my_rotated_strip(my_strip.size());
    rotateStrip(my_strip, my_rotated_strip, width, channels, my_row_count);

    // fiecare fasie are acelasi numar de bytes ca inainte, doar aranjati altfel
    std::vector<unsigned char> gathered_strips;
    if (my_rank == 0) gathered_strips.resize(static_cast<size_t>(width) * height * channels);

    MPI_Gatherv(my_rotated_strip.data(), my_byte_count, MPI_UNSIGNED_CHAR,
                gathered_strips.data(), bytes_per_process.data(), byte_offset_of_process.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        const int rotated_width = height;
        const int rotated_height = width;
        std::vector<unsigned char> rotated_image(gathered_strips.size());

        for (int p = 0; p < process_count; ++p) {
            const int strip_height = rows_per_process[p];
            if (strip_height == 0) continue;

            // fasia de sus ajunge in dreapta, cea de jos in stanga
            const int target_first_col = height - first_row_of_process[p] - strip_height;
            const unsigned char* rotated_strip = &gathered_strips[byte_offset_of_process[p]];
            const size_t strip_row_bytes = static_cast<size_t>(strip_height) * channels;

            // fiecare rand din fasia rotita e o bucata dintr-un rand al imaginii finale
            for (int out_row = 0; out_row < rotated_height; ++out_row)
                std::memcpy(&rotated_image[(static_cast<size_t>(out_row) * rotated_width + target_first_col) * channels],
                            &rotated_strip[out_row * strip_row_bytes],
                            strip_row_bytes);
        }

        if (writeImage(output_path, rotated_width, rotated_height, channels, rotated_image))
            std::cout << "[Rank 0] Saved " << output_path << " (" << rotated_width << "x" << rotated_height << ")\n";
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