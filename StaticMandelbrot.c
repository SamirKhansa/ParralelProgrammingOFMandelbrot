#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <mpi.h>

int mandelbrot(double x, double y, int max_iter) {
    double real = x;
    double imag = y;
    int iter = 0;

    while (iter < max_iter) {
        double real2 = real * real;
        double imag2 = imag * imag;

        if (real2 + imag2 > 4.0) {
            return iter;
        }

        imag = 2 * real * imag + y;
        real = real2 - imag2 + x;

        iter++;
    }

    return max_iter;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    double start_time;
    double wait_time;
    double end_time;
    double parallel_execution_time;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

   

    

    const double xmin = 0.27085;
    const double xmax = 0.27100;
    const double ymin = 0.004640;
    const double ymax = 0.004810;
    const int maxiter = 1000 ;
    const int xres = 1024;
    const int yres = (xres * (ymax - ymin)) / (xmax - xmin);

    const char* filename = "pic.ppm";

    FILE *fp;
    if (rank == 0) {
        fp = fopen(filename, "wb");
        if (fp == NULL) {
            printf("Error: Cannot open file for writing.\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    const char* comment = "# Mandelbrot set"; // comment should start with #

    if (rank == 0) {
        // Only rank 0 writes the PPM header
        fprintf(fp, "P6\n%s\n%d\n%d\n%d\n", comment, xres, yres, (maxiter < 256 ? 256 : maxiter));
    }

    // Calculate the portion of the image to generate for each process
    const int rows_per_process = yres / size;
    const int start_row = rank * rows_per_process;
    const int end_row = (rank + 1) * rows_per_process;

    unsigned char* local_image = (unsigned char*)malloc(3 * xres * rows_per_process);
    start_time = MPI_Wtime();
    for (int j = start_row; j < end_row; j++) {
        for (int i = 0; i < xres; i++) {
            double x = xmin + i * (xmax - xmin) / xres;
            double y = ymin + j * (ymax - ymin) / yres;

            int color = mandelbrot(x, y, maxiter);

            // Map color to RGB
            unsigned char r = (color >> 8) & 0xFF;
            unsigned char g = (color >> 4) & 0xFF;
            unsigned char b = color & 0xFF;

            local_image[(j - start_row) * 3 * xres + i * 3] = r;
            local_image[(j - start_row) * 3 * xres + i * 3 + 1] = g;
            local_image[(j - start_row) * 3 * xres + i * 3 + 2] = b;
        }
    }

    // Gather the local images to rank 0
    if (rank == 0) {
        unsigned char* image = (unsigned char*)malloc(3 * xres * yres);

        MPI_Gather(local_image, 3 * xres * rows_per_process, MPI_UNSIGNED_CHAR,
                   image, 3 * xres * rows_per_process, MPI_UNSIGNED_CHAR,
                   0, MPI_COMM_WORLD);

        // Write the image data to the file
        fwrite(image, 1, 3 * xres * yres, fp);
        fclose(fp);
        free(image);
    } else {
        MPI_Gather(local_image, 3 * xres * rows_per_process, MPI_UNSIGNED_CHAR,
                   NULL, 0, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }
    end_time = MPI_Wtime();
    if(rank==0){
        
        parallel_execution_time = end_time - start_time;
        printf("My Parralele execution time %f\n",parallel_execution_time);
    
    
}
    free(local_image);

    MPI_Finalize();

    return 0;
}
