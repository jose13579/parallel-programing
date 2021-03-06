#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define MASK_WIDTH 5
#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255
#define TILE_WIDTH 32
#define SHARED_WIDTH (MASK_WIDTH-1 + TILE_WIDTH)


typedef struct {
    unsigned char red, green, blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


static PPMImage *readPPM(const char *filename) {
    char buff[16];
    PPMImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }

    img = (PPMImage *) malloc(sizeof(PPMImage));
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n')
            ;
        c = getc(fp);
    }

    ungetc(c, fp);
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        exit(1);
    }

    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
                filename);
        exit(1);
    }

    if (rgb_comp_color != RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        exit(1);
    }

    while (fgetc(fp) != '\n')
        ;
    img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(PPMImage *img) {

    fprintf(stdout, "P6\n");
    fprintf(stdout, "# %s\n", COMMENT);
    fprintf(stdout, "%d %d\n", img->x, img->y);
    fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

    fwrite(img->data, 3 * img->x, img->y, stdout);
    fclose(stdout);
}

__global__ void Smothing_Paralelo(PPMImage* image_input, PPMImage* image_output)
{
	// Creating variables
	int i, j, row, col;
	int total_red = 0, total_blue = 0, total_green = 0;

	// Calculate the row and col # of the image 
	row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	col = blockIdx.x * TILE_WIDTH + threadIdx.x;

	// Create private copies for each block of the image
	// save each block of the image in a shared memory
	// each block of the image has SHARED_WIDTH*SHARED_WIDTH as size
	// Is considered the border generated by the mask
	// SHARED_WIDTH = TILE_WIDTH + MASK-1
	// Size of the SHARED_WIDTH is the size the thread defined
	// and the mask - 1 because is considered the left and right border as well
	// up and down border
	__shared__ PPMPixel shared_image[SHARED_WIDTH*SHARED_WIDTH];

	// Checks if the col and row position in the image is less than the width and height of the image 
    	if(row < image_output->y && col < image_output->x)
	{
		for(int i = -(MASK_WIDTH-1)/2; i<=(MASK_WIDTH-1)/2; i++)
		{
			for(int j = -(MASK_WIDTH-1)/2; j<=(MASK_WIDTH-1)/2; j++)
			{
				int new_col= col + i; 
				int new_row = row + j;
				
				// The filtering operation produces a output image from the input image.
				// the mask is shifted to a size of MASK_WIDTH where each pixel 
				// value is multiplied by each pixel value of the mask and then added in the 
				// each pixel of the output image

				// As the mask is a mask with ones, then only is necessary
				// save each pixel of the input image in the share block image
				// Finally sum all pixels by MASK_WIDTH and save it in the output image
				// index_mask calculate the position of each pixel convoluted by the mask 
				int index_mask = new_row * image_output->x  + new_col;
					
				int shared_col = threadIdx.x + ((MASK_WIDTH-1)/2) + i;
				int shared_row = threadIdx.y + ((MASK_WIDTH-1)/2) + j;
				
				// index_shared calculate the position all pixels in the shared block memory
				int index_shared = shared_row * SHARED_WIDTH + shared_col;
					
				if(index_shared < SHARED_WIDTH*SHARED_WIDTH)
				{	

					// If the position of each pixel convoluted by the mask is smaller than zero 	
					// and greater than WIDTH and HEIGTH add zero in the shared block memory
					// else add the current position of the input image
					if(new_row >= 0 && new_row < image_output->y && new_col >= 0 && new_col < image_output->x)
					{
						shared_image[index_shared].red = image_input->data[index_mask].red;
						shared_image[index_shared].green = image_input->data[index_mask].green;
						shared_image[index_shared].blue = image_input->data[index_mask].blue;
					}
					else
					{
						shared_image[index_shared].red = 0;
						shared_image[index_shared].green = 0;
						shared_image[index_shared].blue = 0;				
					}
				}
			}
		}
	}

	// Wait for all other threads in the block to finish
	__syncthreads();

	// Checks if row and col stay into image proceed with convolution
	if (row < image_output->y && col < image_output->x){
		for (i = 0; i < MASK_WIDTH; i++){
		     for (j = 0; j < MASK_WIDTH; j++) {
			// Sum all pixels by MASK_WIDTH and save it in the output channel image
			total_red += shared_image[((threadIdx.y + j) * SHARED_WIDTH) + (threadIdx.x + i)].red;
			total_blue += shared_image[((threadIdx.y + j) * SHARED_WIDTH) + (threadIdx.x + i)].blue;
			total_green += shared_image[((threadIdx.y + j) * SHARED_WIDTH) + (threadIdx.x + i)].green;
		     }
		}

		// Save convolution input data with the mask into output image
		image_output->data[(row * image_output->x) + col].red = total_red / (MASK_WIDTH*MASK_WIDTH);
		image_output->data[(row * image_output->x) + col].blue = total_blue / (MASK_WIDTH*MASK_WIDTH);
		image_output->data[(row * image_output->x) + col].green = total_green / (MASK_WIDTH*MASK_WIDTH);
	}
}

int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }
    //Timers
    cudaEvent_t start, stop;
    cudaEventCreate ( &start );
    cudaEventCreate ( &stop );

    //double t_start, t_end;
    char *filename = argv[1]; //Recebendo o arquivo!;

    // Initialize variables
    unsigned int rows, cols, img_size;
    PPMImage *image_output, *image_output_par, *image_par;
    PPMPixel *pixels_output_par, *pixels_par, *pixels_output;

    // Read output Image
    image_output = readPPM(filename);

    // Get data
    cols = image_output->x;
    rows = image_output->y;
    img_size = cols * rows;

    // Alloc CPU memory
    pixels_output = (PPMPixel *) malloc(img_size * sizeof(PPMPixel));

    // Alloc space for device structures copies to input and output image
    cudaMalloc((void **)&image_output_par, sizeof(PPMImage));
    cudaMalloc((void **)&image_par, sizeof(PPMImage));

    // Alloc space for device pixels copies to input and output image
    cudaMalloc((void **)&pixels_output_par, sizeof(PPMPixel) * img_size);
    cudaMalloc((void **)&pixels_par, sizeof(PPMPixel) * img_size);

    // Copy outputs to device
    cudaMemcpy(image_output_par, image_output, sizeof(PPMImage), cudaMemcpyHostToDevice);
    cudaMemcpy(pixels_output_par, image_output->data, sizeof(PPMPixel) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(image_output_par->data), &pixels_output_par, sizeof(PPMPixel *), cudaMemcpyHostToDevice);

    // Copy inputs to device
    cudaMemcpy(image_par, image_output, sizeof(PPMImage), cudaMemcpyHostToDevice);
    cudaMemcpy(pixels_par, image_output->data, sizeof(PPMPixel) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(image_par->data), &pixels_par, sizeof(PPMPixel *), cudaMemcpyHostToDevice);

    // Initialize dimGrid and dimBlocks
    // create a grid with (columns / TILE_WIDTH) number of columns and (lines / TILE_WIDTH) number of rows
    // the ceiling function makes sure there are enough to cover all elements 
    dim3 dimGrid(ceil((float)cols / TILE_WIDTH), ceil((float)rows / TILE_WIDTH), 1);

    // Create a block with TILE_WIDTH columns and TILE_WIDTH rows
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    cudaEventRecord ( start );
    // Launch Smothing_Paralelo() kernel on GPU with a grid and block as input
    Smothing_Paralelo<<<dimGrid, dimBlock>>>(image_par,image_output_par);
    //////////////////////////////////////////////////////////////////////////
    cudaEventRecord ( stop );
    cudaEventSynchronize ( stop );
    float tempo_GPU_kernel = 0;
    cudaEventElapsedTime ( &tempo_GPU_kernel,start,stop );

    // Copy results to local output image
    cudaMemcpy(image_output, image_output_par, sizeof(PPMImage), cudaMemcpyDeviceToHost);
    cudaMemcpy(pixels_output, pixels_output_par, sizeof(PPMPixel) * img_size, cudaMemcpyDeviceToHost);
    image_output->data = pixels_output;

    // Free memory
    cudaFree(image_output_par);
    cudaFree(image_par);
    cudaFree(pixels_output_par);
    cudaFree(pixels_par);

    // Write the output image and show it
    writePPM(image_output);

    printf ( "\n Time GPU kernel %.2f\n",tempo_GPU_kernel);
    printf("\n");

    //fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);
    free(image_output);
}
