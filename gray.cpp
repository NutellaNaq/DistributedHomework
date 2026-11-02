// gray.cpp
// Program MPI care transformă o imagine color (PPM) în tonuri de gri (PGM).
#include <mpi.h>

#include <climits>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

int main(int argc, char** argv) {
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