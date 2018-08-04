#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


// MASK SIZE
#define MASK_WIDTH 13
// MASK RADIO
#define MASK_R (MASK_WIDTH-1)/2

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

// SIZE OF TILE
#define TILE_WIDTH 32
// SIZE OF SHARE MATRIX
#define SHARED_SIZE (MASK_WIDTH-1 + TILE_WIDTH)


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

__global__ void smothing_filter_Paralelo(unsigned char* image_input, unsigned char* image_output, int height, int width, int dim)
{
	// Creating variables
	int i, j, row, col;
	int total_red = 0, total_blue = 0, total_green = 0;

	// Get Row and COl
	row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	col = blockIdx.x * TILE_WIDTH + threadIdx.x;

	// Create Shared block of data
	__shared__ int private_red[SHARED_SIZE*SHARED_SIZE];
	__shared__ int private_green[SHARED_SIZE*SHARED_SIZE];
	__shared__ int private_blue[SHARED_SIZE*SHARED_SIZE];


    	if(row < height && col < width)
	{
		for(int i = -(MASK_WIDTH-1)/2; i<=(MASK_WIDTH-1)/2; i++)
		{
			for(int j = -(MASK_WIDTH-1)/2; j<=(MASK_WIDTH-1)/2; j++)
			{
				int new_col= col + i; 
				int new_row = row + j;
				
				int index_mask = new_row * width  + new_col;
					
				int shared_col = threadIdx.x + ((MASK_WIDTH-1)/2) + i;
				int shared_row = threadIdx.y + ((MASK_WIDTH-1)/2) + j;
				
				int index_shared = shared_row * SHARED_SIZE + shared_col;
					
				if(index_shared < SHARED_SIZE*SHARED_SIZE)
				{	
					if(new_row >= 0 && new_row < height && new_col >= 0 && new_col < width)
					{
						private_red[index_shared] = image_input[index_mask*3];
                				private_green[index_shared] = image_input[index_mask*3+1];
                				private_blue[index_shared] = image_input[index_mask*3+2];
					}
					else
					{
						private_red[index_shared] = 0;
						private_green[index_shared] = 0;
						private_blue[index_shared] = 0;				
					}
				}
			}
		}
	}

	// sync threads
	__syncthreads();

	// if row and col stay into image proceed with convolution
	if (row < height && col < width){
		for (i = 0; i < MASK_WIDTH; i++){
		     for (j = 0; j < MASK_WIDTH; j++) {
			total_red += private_red[((threadIdx.y + j) * SHARED_SIZE) + (threadIdx.x + i)];
			total_blue += private_green[((threadIdx.y + j) * SHARED_SIZE) + (threadIdx.x + i)];
			total_green += private_blue[((threadIdx.y + j) * SHARED_SIZE) + (threadIdx.x + i)];
		     }
		}
		// Save data of convolution into devise image
		image_output[(row * width + col) * 3] = total_red / (MASK_WIDTH*MASK_WIDTH);
		image_output[(row * width + col) * 3 + 1] = total_blue / (MASK_WIDTH*MASK_WIDTH);
		image_output[(row * width + col) * 3 + 2] = total_green / (MASK_WIDTH*MASK_WIDTH);
	}
}

int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }

    //double t_start, t_end;
    //int i;
    char *filename = argv[1]; //Recebendo o arquivo!;

    PPMImage *image = readPPM(filename);
    PPMImage *image_output = readPPM(filename);

    unsigned int rows, cols, img_size;

    // Iniatialize variables
    unsigned char* image_par;
    unsigned char* image_output_par;

    // Get data
    cols = image->x;
    rows = image->y;
    img_size = cols * rows;

    unsigned char* image_vect = Image_vector(image,img_size);

    cudaMalloc((void**)&image_par, sizeof(unsigned char)*img_size*3);
    cudaMalloc((void**)&image_output_par, sizeof(unsigned char)*img_size*3);

    cudaMemcpy(image_par, image_vect, sizeof(unsigned char)*img_size*3 ,cudaMemcpyHostToDevice );
    cudaMemcpy(image_output_par, image_vect, sizeof(unsigned char)*img_size*3 ,cudaMemcpyHostToDevice );
    

    dim3 dimGrid(ceil((float)cols / TILE_WIDTH), ceil((float)rows / TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Call function
    smothing_filter_Paralelo<<<dimGrid, dimBlock>>>(image_par,image_output_par,cols,rows,img_size);

    cudaMemcpy(image_vect, image_output_par, sizeof(unsigned char)*img_size*3, cudaMemcpyDeviceToHost);

    vector_Image(image_output,image_vect ,img_size);

    //Free memory
    cudaFree(image_par);
    cudaFree(image_output_par);


    //fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);
    free(image);
    free(image_output);
}
