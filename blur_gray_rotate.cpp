// Cele trei implementari originale, izolate in namespace-uri.
// mpic++ -O3 -std=c++17 operatii.cpp -o operatii
// mpirun -np 4 ./operatii gray input.ppm output.pgm [threshold]
// mpirun -np 4 ./operatii blur input.ppm output.ppm
// mpirun -np 4 ./operatii rotate input.ppm output.ppm
#include <climits>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>
namespace gray_original {
// gray.cpp
// Program MPI care transformă o imagine color (PPM) în tonuri de gri (PGM).


// Citire / scriere fișiere 
// Citește următorul număr întreg din antetul PPM.
// Sare peste spații albe și peste comentarii (liniile care încep cu '#').
static bool readHeaderInt(std::istream& in, int& value) {
    while (true) {
        in >> std::ws;                    // remove whitespaces
        if (in.peek() == '#') {           // verifica file comments
            std::string comment;
            std::getline(in, comment);    // trecem peste linie
        } else {
            break;                        
        }
    }
    return static_cast<bool>(in >> value);
}


// Forma unui ppm file
// P6     <-- Citim acel magic number
// # creat cu GIMP <-- Verificam mereu sa nu avem un comentariu la mijloc
// 640 480    <--- latimea x inaltimea
// 255    <--- Valoarea maxima a culorii


// Încarcă o imagine PPM binară (P6). Pixelii sunt stocați rând cu rând,
// 3 octeti (R, G, B) pentru fiecare pixel.
static bool readPPM(const std::string& input_file, int& width, int& height,
                    std::vector<unsigned char>& full_rgb) {
    std::ifstream file(input_file, std::ios::binary); // std::ios:binary e pasata citirea Binary pentru a evita conversia caracterelor
    if (!file.is_open()) return false;

    // "Numărul magic" spune tipul fișierului; aici acceptăm doar P6 (color, binar)
    std::string magic;
    file >> magic;
    if (magic != "P6") {
        std::cerr << "Error: Input image must be in binary PPM (P6) format!\n";
        return false;
    }

    // Antetul conține: lățime, înălțime, valoarea maximă a unei culori
    int max_val = 0;
    if (!readHeaderInt(file, width) || !readHeaderInt(file, height) || !readHeaderInt(file, max_val)) {
        std::cerr << "Error: Malformed PPM header!\n";
        return false;
    }
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Invalid image dimensions!\n";
        return false;
    }
    if (max_val <= 0 || max_val > 255) {
        std::cerr << "Error: Only 8-bit PPM files (max value <= 255) are supported!\n";
        return false;
    }

    // MPI folosește int pentru numărul de elemente, deci nu putem depăși INT_MAX
    if (static_cast<long long>(width) * height * 3 > INT_MAX) {
        std::cerr << "Error: image too large for int-based MPI counts.\n";
        return false;
    }

    file.get();  // după max_val urmează exact UN caracter alb, apoi datele binare

    full_rgb.resize(static_cast<size_t>(width) * height * 3);
    file.read(reinterpret_cast<char*>(full_rgb.data()), static_cast<std::streamsize>(full_rgb.size())); // Facem compiler sa creada ca avem char in full_rgb si stocam data in el 
    if (file.gcount() != static_cast<std::streamsize>(full_rgb.size())) { // Verifica daca am citit destul din fisier si a facyt match cu nr total de pixeli al acestei imaginini WxH
        std::cerr << "Error: PPM file is truncated!\n";
        return false;
    }
    return true;
}

// Salvează o imagine gri ca PGM binar (P5): 1 octet pe pixel.
static bool writePGM(const std::string& output_file, int width, int height,
                     const std::vector<unsigned char>& data) {
    std::ofstream file(output_file, std::ios::binary);
    if (!file.is_open()) return false;
    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(file);
}


// Transformă pixelii RGB în tonuri de gri.
// Formula standard : ochiul e mai sensibil la verde, deci verdele contează mai mult.
static void convertToGray(const std::vector<unsigned char>& rgb, std::vector<unsigned char>& gray) {
    for (size_t i = 0; i < gray.size(); ++i) {
        const double r = rgb[i * 3 + 0];
        const double g = rgb[i * 3 + 1];
        const double b = rgb[i * 3 + 2];
        gray[i] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b + 0.5); // +0.5 = rotunjire
        // Formula luata de pe net
    }
}

// Alb-negru: sub prag -> 0 (negru), altfel -> 255 (alb). Pragul -1 = nu facem nimic.
static void applyThreshold(std::vector<unsigned char>& gray, int threshold) {
    if (threshold < 0) return;
    for (auto& p : gray) p = (p < threshold) ? 0 : 255;
}

// Main app 

int run(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // rank = numărul acestui proces 0, 1, 2... || num_procs = câte procese rulează în total
    int rank = 0, num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Numele calculatorului pe care rulează procesul
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int name_len = 0;
    MPI_Get_processor_name(proc_name, &name_len);

    // -- Argumentele necesare pentru rulare --
    const std::string usage =
        "Usage: mpirun -np N ./gray <input.ppm> <output.pgm> [threshold]\n"
        "  threshold: -1..255 (default -1 = grayscale, no black & white)\n";

    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() < 2 || args.size() > 3) {
        if (rank == 0) std::cerr << usage;   // doar procesul 0 afișează, ca să nu apară de N ori
        MPI_Finalize();
        return 1;
    }
    const std::string input_file = args[0];
    const std::string output_file = args[1];

    int threshold = -1;
    if (args.size() == 3) {
        bool ok = true;
        try {
            size_t pos = 0;
            threshold = std::stoi(args[2], &pos);
            if (pos != args[2].size()) ok = false;   // nu e un numar valid
        } catch (const std::exception&) {
            ok = false;
        }
        if (!ok || threshold < -1 || threshold > 255) {
            if (rank == 0) std::cerr << "Error: invalid threshold '" << args[2] << "'\n" << usage;
            MPI_Finalize();
            return 1;
        }
    }

    // ---------- Citirea imaginii (doar procesul 0) ----------
    int width = 0, height = 0;
    std::vector<unsigned char> full_rgb;

    if (rank == 0) {
        if (!readPPM(input_file, width, height, full_rgb)) {
            std::cerr << "Failed to open or parse " << input_file << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);   // oprește TOATE procesele
        }
        std::cout << "[Rank 0 on " << proc_name << "] Loaded " << input_file
                  << " (" << width << "x" << height << " pixels), operation: gray\n";
    }

    // Așteptăm ca toate procesele să ajungă aici, apoi pornim cronometrul
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Procesul 0 trimite tuturor dimensiunile imaginii (Broadcast)
    int dims[2] = {width, height};
    MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
    width = dims[0];
    height = dims[1];

    // ---------- Împărțirea rândurilor între procese ----------
    // Ex.: 10 rânduri, 3 procese -> 4, 3, 3 (primele procese primesc câte un rând în plus)
    std::vector<int> rows(num_procs), row_starts(num_procs);
    const int base_rows = height / num_procs;
    const int remainder_rows = height % num_procs;
    int offset = 0;
    for (int i = 0; i < num_procs; ++i) {
        rows[i] = base_rows + (i < remainder_rows ? 1 : 0);  // câte rânduri primește procesul i
        row_starts[i] = offset;                              // de la ce rând începe
        offset += rows[i];
    }

    // Transformă rândurile în octeți pentru MPI:
    // counts[i] = câți octeți are procesul i; displs[i] = de unde încep în imaginea completă
    auto make_counts = [&](int channels, std::vector<int>& counts, std::vector<int>& displs) {
        counts.resize(num_procs);
        displs.resize(num_procs);
        for (int i = 0; i < num_procs; ++i) {
            counts[i] = rows[i] * width * channels;
            displs[i] = row_starts[i] * width * channels;
        }
    };

    const int local_rows = rows[rank];
    const int local_pixels = local_rows * width;

    // Scatterv: procesul 0 trimite fiecărui proces bucata lui de rânduri RGB (3 octeți/pixel)
    std::vector<int> counts_rgb, displs_rgb;
    make_counts(3, counts_rgb, displs_rgb);
    std::vector<unsigned char> local(static_cast<size_t>(local_pixels) * 3); // Numarul de valori RGB (Nr Pixeli x 3 -- R,G,B)
    MPI_Scatterv(full_rgb.data(), counts_rgb.data(), displs_rgb.data(), MPI_UNSIGNED_CHAR,
                 local.data(), local_pixels * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    std::cout << "  -> Rank " << rank << " (" << proc_name << ") processing "
              << local_rows << " rows (" << local_pixels << " pixels)...\n";

    // ---------- Prelucrarea: fiecare proces lucrează doar pe bucata lui ----------
    // Pixelii sunt independenți între ei, deci NU e nevoie de comunicare între procese.
    std::vector<unsigned char> gray(local_pixels);
    convertToGray(local, gray);
    applyThreshold(gray, threshold);

    // ---------- Adunarea rezultatelor la procesul 0 ----------
    // Acum fiecare pixel are 1 octet (gri), deci recalculăm counts/displs cu 1 canal
    std::vector<int> counts, displs;
    make_counts(1, counts, displs);
    std::vector<unsigned char> final_img;
    if (rank == 0) final_img.resize(static_cast<size_t>(width) * height);

    // Gatherv = inversul lui Scatterv: toate bucățile vin înapoi la procesul 0
    MPI_Gatherv(gray.data(), local_pixels, MPI_UNSIGNED_CHAR,
                final_img.data(), counts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (writePGM(output_file, width, height, final_img))
            std::cout << "[Rank 0] Saved result to " << output_file << " (PGM, " << width << "x" << height << ")\n";
        else
            std::cerr << "[Rank 0] Error writing output file!\n";
    }

    // Oprim cronometrul abia după ce au terminat toate procesele
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    if (rank == 0)
        std::cout << "Total parallel execution time: " << (end_time - start_time) << " seconds.\n";

    MPI_Finalize();
    return 0;
}
}
namespace blur_original {


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

int run(int argc, char** argv) {
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
}
namespace rotate_original {


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

int run(int argc, char** argv) {
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
}

int main(int argc, char** argv) {
    if (argc >= 2) {
        const std::string op = argv[1];
        // argv+1 pastreaza conventia: primul argument este numele programului.
        if (op == "gray") return gray_original::run(argc-1, argv+1);
        if (op == "blur") return blur_original::run(argc-1, argv+1);
        if (op == "rotate") return rotate_original::run(argc-1, argv+1);
    }
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank) std::cerr << "Usage: operatii gray|blur|rotate input output [threshold doar gray]\n";
    MPI_Finalize(); return 1;
}
