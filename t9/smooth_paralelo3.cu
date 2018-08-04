#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define MASK_WIDTH 5
#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255
#define TILE_WIDTH 32
#define SHARED_WIDTH TILE_WIDTH+(MASK_WIDTH-1)/2

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


void Smoothing_CPU_Serial(PPMImage *image, PPMImage *image_copy) {
    int i, j, y, x;
    int total_red, total_blue, total_green;

    for (i = 0; i < image->y; i++) {
        for (j = 0; j < image->x; j++) {
            total_red = total_blue = total_green = 0;
            for (y = i - ((MASK_WIDTH-1)/2); y <= (i + ((MASK_WIDTH-1)/2)); y++) {
                for (x = j - ((MASK_WIDTH-1)/2); x <= (j + ((MASK_WIDTH-1)/2)); x++) {
                    if (x >= 0 && y >= 0 && y < image->y && x < image->x) {
                        total_red += image_copy->data[(y * image->x) + x].red;
                        total_blue += image_copy->data[(y * image->x) + x].blue;
                        total_green += image_copy->data[(y * image->x) + x].green;
                    } //if
                } //for z
            } //for y
            image->data[(i * image->x) + j].red = total_red / (MASK_WIDTH*MASK_WIDTH);
            image->data[(i * image->x) + j].blue = total_blue / (MASK_WIDTH*MASK_WIDTH);
            image->data[(i * image->x) + j].green = total_green / (MASK_WIDTH*MASK_WIDTH);
        }
    }
}

void vector_Image(PPMImage *image_output, unsigned char* image_output_vector, int dim )
{
        for ( int i = 0; i < dim; i++ ) {
                image_output->data[i].red = image_output_vector[i*3];
                image_output->data[i].green = image_output_vector[i*3+1];
                image_output->data[i].blue = image_output_vector[i*3+2];
        }
}

unsigned char* Image_vector(PPMImage* image_input, int dim )
{
        unsigned char* image_input_vector = ( unsigned char* ) malloc ( sizeof ( unsigned char ) * dim*3 );

        //Fills the vector with the already normalized data.
        for ( int i = 0; i < dim; i++) {
                image_input_vector[i*3] = image_input->data[i].red;
                image_input_vector[i*3+1] = image_input->data[i].green;
                image_input_vector[i*3+2] = image_input->data[i].blue;
        }
        return image_input_vector;
}

__global__ void smothing_filter_Paralelo(unsigned char* image_input, unsigned char* image_output, int width, int height, int dim )
{
	__shared__ int private_red[TILE_WIDTH*TILE_WIDTH];
	__shared__ int private_green[TILE_WIDTH*TILE_WIDTH];
	__shared__ int private_blue[TILE_WIDTH*TILE_WIDTH];

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

	if((row < width && col < height) && (index < dim))
	{
		private_red[ty * TILE_WIDTH + tx] = image_input[index * 3];
		private_green[ty * TILE_WIDTH + tx] = image_input[index * 3 + 1];
		private_blue[ty * TILE_WIDTH + tx] = image_input[index * 3 + 2];
	}
	else
	{
		private_red[ty * TILE_WIDTH + tx] = 0;
		private_green[ty * TILE_WIDTH + tx] = 0;
		private_blue[ty * TILE_WIDTH + tx] = 0;				
	}

	__syncthreads();

	if(row < width && col < height)
	{
		for (y = row - ((MASK_WIDTH-1)/2); y <= (row + ((MASK_WIDTH-1)/2)); y++) {
			for (x = col - ((MASK_WIDTH-1)/2); x <= (col + ((MASK_WIDTH-1)/2)); x++) {
				if (x >= 0 && y >= 0 && y < width && x < height) {
					total_red += private_red[y * TILE_WIDTH + x];
					total_blue += private_green[y * TILE_WIDTH + x];
					total_green += private_blue[y * TILE_WIDTH + x];
				} //if
			} //for z
		} //for y

		__syncthreads();

		image_output[index * 3] = total_red / (MASK_WIDTH*MASK_WIDTH);
		image_output[index * 3 + 1] = total_blue / (MASK_WIDTH*MASK_WIDTH);
		image_output[index * 3 + 2] = total_green / (MASK_WIDTH*MASK_WIDTH);
	}
	/*
	int y, x;
	int total_red = 0, total_green = 0, total_blue = 0;

	int bx = blockIdx.x;  
	int by = blockIdx.y;

	int tx= threadIdx.x; 
	int ty = threadIdx.y;

	//Compute the current element index
	int row = by * blockDim.y + ty;
	int col = bx* blockDim.x+ tx;
	
        //Computes the histogram position and increments it.
        //if ( ( row < width && col < height ) && ( index < dim ) ) {

	total_red = total_blue = total_green = 0;

	
	for(int p=0; p<ceil((float)dim/TILE_WIDTH); ++p)
	{
		int i = row;
		int j = p*TILE_WIDTH+tx;
		
		if(i < width && j < height)
		{
			private_red[ty][tx] = image_input[(i*width + j+tx)*3];
			private_green[ty][tx] = image_input[(i*width + j+tx)*3+1];
			private_blue[ty][tx] = image_input[(i*width + j+tx)*3+2];
		}
		else
		{
			private_red[ty][tx] = 0;
			private_green[ty][tx] = 0;
			private_blue[ty][tx] = 0;				
		}

		__syncthreads();

		
		for (y = i - ((MASK_WIDTH-1)/2); y <= (i + ((MASK_WIDTH-1)/2)); y++) {
			for (x = j - ((MASK_WIDTH-1)/2); x <= (j + ((MASK_WIDTH-1)/2)); x++) {
				int curr_indx = (y * height) + x;

				if (x >= 0 && y >= 0 && y < height && x < width) {
					total_red += private_red[ty][tx];
					total_blue += private_green[ty][tx];
					total_green += private_blue[ty][tx];
				} //if
			} //for z
		} //for y

		__syncthreads();

		image_output[(i*width + j+tx)*3] = total_red / (MASK_WIDTH*MASK_WIDTH);
    		image_output[(i*width + j+tx)*3+1] = total_blue / (MASK_WIDTH*MASK_WIDTH);
    		image_output[(i*width + j+tx)*3+2] = total_green / (MASK_WIDTH*MASK_WIDTH);
	//}
        }
	*/

	
}

int main(int argc, char *argv[]) {

	if( argc != 2 ) {
		printf("Too many or no one arguments supplied.\n");
	}

	double t_start, t_end;
	int i;
	char *filename = argv[1]; //Recebendo o arquivo!;

	PPMImage *image = readPPM(filename);
	PPMImage *image_output = readPPM(filename);

	int x = image->x;
	int y = image->y; 
	int dim = x*y;

	unsigned char* image_vect = Image_vector(image,dim);
	
	// Iniatialize variables
	unsigned char* image_par;
	unsigned char* image_output_par;

	// Alloc CPU memory
	unsigned char* image_output_vect = Image_vector(image,dim);

	// Alloc space for device copies of a, b, c //////////////////
	cudaMalloc((void**)&image_par, sizeof(unsigned char)*dim*3);
	cudaMalloc((void**)&image_output_par, sizeof(unsigned char)*dim*3);

	// Copy inputs to device ////////////////////////////////////////////////////////////////
	cudaMemcpy(image_par, image_vect, sizeof(unsigned char)*dim*3 ,cudaMemcpyHostToDevice );
	cudaMemcpy(image_output_par, image_output_vect, sizeof(unsigned char)*dim*3 ,cudaMemcpyHostToDevice );

	// Initialize dimGrid and dimBlocks
	// create a grid with (32 / columns) number of columns and (32 / lines) number of rows
	// the ceiling function makes sure there are enough to cover all elements 
	dim3 dimGrid(ceil((float)y / TILE_WIDTH), ceil((float)x / TILE_WIDTH), 1);

	// create a block with 32 columns and 32 rows
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	// Launch smothing_filter_Paralelo() kernel on GPU with a grid and block as input ////
	smothing_filter_Paralelo<<<dimGrid, dimBlock>>>(image_par, image_output_par,x,y,dim);

	// Copy result to local array //////////////////////////////////////////////
	cudaMemcpy(image_output_vect, image_output_par, sizeof(unsigned char)*dim*3, cudaMemcpyDeviceToHost);

	vector_Image(image_output,image_output_vect,dim);

	writePPM(image_output);

	cudaFree(image_par);
	cudaFree(image_output_par);

	free(image);
	free(image_output);	
}
