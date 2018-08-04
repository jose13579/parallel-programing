#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define MASK_WIDTH 5
#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255
#define TILE_WIDTH 32
#define SHARED_WIDTH TILE_WIDTH+(MASK_WIDTH-1)

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

__global__ void smothing_filter_Paralelo(PPMImage* image_input, PPMImage* image_output, int width, int height, int dim )
{
	__shared__ PPMPixel shared_image[TILE_WIDTH*TILE_WIDTH];
	//__shared__ int private_green[TILE_WIDTH*TILE_WIDTH];
	//__shared__ int private_blue[TILE_WIDTH*TILE_WIDTH];

	int y, x;
	int total_red = 0, total_green = 0, total_blue = 0;

	int bx = blockIdx.x;  
	int by = blockIdx.y; 

	int tx= threadIdx.x; 
	int ty = threadIdx.y;

	/*if((tx * TILE_WIDTH + ty) < TILE_WIDTH * TILE_WIDTH){
		private_red[tx * TILE_WIDTH + ty] = 0;
		private_green[tx * TILE_WIDTH + ty] = 0;
		private_blue[tx * TILE_WIDTH + ty] = 0;
	}

	__syncthreads();*/

	//Compute the current element index
	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	// Calculate the index of the image
    	int index = row * width + col;


	if(row < height && col < width)
	{
		
		for(int i = -(MASK_WIDTH-1)/2; i<=(MASK_WIDTH-1)/2; i++)
		{
			for(int j = -(MASK_WIDTH-1)/2; j<=(MASK_WIDTH-1)/2; j++)
			{
				int new_col= col + i; 
				int new_row = row + j;
				
				int index_mask = new_row * width  + new_col;
					
				int shared_col = tx + ((MASK_WIDTH-1)/2) + i;
				int shared_row = ty + ((MASK_WIDTH-1)/2) + j;
				
				int index_shared = shared_row * SHARED_WIDTH + shared_col;
					
				if(index_shared < SHARED_WIDTH*SHARED_WIDTH)
				{	
					if(new_row >= 0 && new_row < height && new_col >= 0 && new_col < width)
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

	__syncthreads();

	if(row < height && col < width)
	{
		for (int i = 0; i <= MASK_WIDTH; i++) {
			for (int j = 0; j <= MASK_WIDTH; j++) {
				int index_shared = (ty + j) * SHARED_WIDTH + (tx + i);
				total_red += shared_image[index_shared].red;
				total_blue += shared_image[index_shared].blue;
				total_green += shared_image[index_shared].green;
			} //for z
		} //for y

		//__syncthreads();

		image_output->data[index].red = total_red / (MASK_WIDTH*MASK_WIDTH);
		image_output->data[index].blue = total_blue / (MASK_WIDTH*MASK_WIDTH);
		image_output->data[index].green = total_green / (MASK_WIDTH*MASK_WIDTH);
	}
}

int main(int argc, char *argv[]) {

	if( argc != 2 ) {
		printf("Too many or no one arguments supplied.\n");
	}

	//double t_start, t_end;
	char *filename = argv[1]; //Recebendo o arquivo!;

	PPMImage *image = readPPM(filename);
	PPMImage *image_output = readPPM(filename);

	unsigned int rows, cols, dim;
	PPMImage *d_image_output, *d_image;
	PPMPixel *d_pixels_output, *d_pixels, *new_pixels;

	// Get data
	cols = image_output->x;
	rows = image_output->y;
	dim = cols * rows;

	// Alloc structure to devise
	cudaMalloc((void **)&d_image_output, sizeof(PPMImage));
	cudaMalloc((void **)&d_image, sizeof(PPMImage));

	// Alloc image to devise
	cudaMalloc((void **)&d_pixels_output, sizeof(PPMPixel) * dim);
	cudaMalloc((void **)&d_pixels, sizeof(PPMPixel) * dim);

	// cpy stucture to devise
	cudaMemcpy(d_image_output, image_output, sizeof(PPMImage), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixels_output, image_output->data, sizeof(PPMPixel) * dim, cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_image_output->data), &d_pixels_output, sizeof(PPMPixel *), cudaMemcpyHostToDevice);

	cudaMemcpy(d_image, image, sizeof(PPMImage), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixels, image->data, sizeof(PPMPixel) * dim, cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_image->data), &d_pixels, sizeof(PPMPixel *), cudaMemcpyHostToDevice);

	// Initialize dimGrid and dimBlocks
	// create a grid with (32 / columns) number of columns and (32 / lines) number of rows
	// the ceiling function makes sure there are enough to cover all elements 
	dim3 dimGrid(ceil((float)cols / TILE_WIDTH), ceil((float)rows / TILE_WIDTH), 1);

	// create a block with 32 columns and 32 rows
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	// Launch smothing_filter_Paralelo() kernel on GPU with a grid and block as input ////
	smothing_filter_Paralelo<<<dimGrid, dimBlock>>>(d_image, d_image_output,cols,rows,dim);

	// Copy result to local array //////////////////////////////////////////////
	new_pixels = (PPMPixel *) malloc(dim * sizeof(PPMPixel));

	// Copy result to local array
	cudaMemcpy(image_output, d_image_output, sizeof(PPMImage), cudaMemcpyDeviceToHost);
	cudaMemcpy(new_pixels, d_pixels_output, sizeof(PPMPixel) * dim, cudaMemcpyDeviceToHost);
	image_output->data = new_pixels;

	writePPM(image_output);

	//Free memory
	cudaFree(d_image_output);
	cudaFree(d_image);
	cudaFree(d_pixels_output);
	cudaFree(d_pixels);

	free(image);
	free(image_output);	
}
