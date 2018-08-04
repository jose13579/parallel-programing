#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define MASK_WIDTH 3
#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255
#define TILE_WIDTH 32
#define SHARED_WIDTH TILE_WIDTH+(MASK_WIDTH-1)

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

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

__global__ void smothing_filter_Paralelo(unsigned char* image_input, unsigned char* image_output, int height, int width, int dim )
{
	__shared__ int private_red[SHARED_WIDTH*SHARED_WIDTH];
	__shared__ int private_green[SHARED_WIDTH*SHARED_WIDTH];
	__shared__ int private_blue[SHARED_WIDTH*SHARED_WIDTH];

	//int y, x;
	int total_red = 0, total_green = 0, total_blue = 0;
	int index_dst_y, index_dst_x, index_src_y, index_src_x;
	
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
	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

	// Calculate the index of the image
    int index = row * width + col;
    
    /*
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
						private_red[index_shared] = image_input[index_mask * 3];
						private_green[index_shared] = image_input[index_mask * 3 + 1];
						private_blue[index_shared] = image_input[index_mask * 3 + 2];
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
	}*/
	
	for (int i = 0; i <= TILE_WIDTH * TILE_WIDTH; i = i + TILE_WIDTH * TILE_WIDTH)
    {
        // Get indexs of dst matrix
        index_dst_y = (threadIdx.y * TILE_WIDTH + threadIdx.x + i) / SHARED_WIDTH;
        index_dst_x = (threadIdx.y * TILE_WIDTH + threadIdx.x + i) % SHARED_WIDTH;

        // Get indexs of destination matrix
        index_src_y = (blockIdx.y * TILE_WIDTH) + index_dst_y - ((MASK_WIDTH-1)/2);
        index_src_x = (blockIdx.x * TILE_WIDTH) + index_dst_x - ((MASK_WIDTH-1)/2);
        
        
        
        //Work only if dst geral index stay into shared matrix size
        if (index_dst_y * SHARED_WIDTH + index_dst_x < (SHARED_WIDTH*SHARED_WIDTH)) {
			
            // if src index stay into image save images values else save 0
            if (index_src_y >= 0 && index_src_y < height && index_src_x >= 0 && index_src_x < width){
                private_red[index_dst_y * SHARED_WIDTH + index_dst_x] = image_input[(index_src_y * width) + index_src_x];
                private_green[index_dst_y * SHARED_WIDTH + index_dst_x] = image_input[(index_src_y * width) + index_src_x];
                private_blue[index_dst_y * SHARED_WIDTH + index_dst_x] = image_input[(index_src_y * width) + index_src_x];
            }
            else{
                private_red[index_dst_y * SHARED_WIDTH + index_dst_x] = 0;
                private_green[index_dst_y * SHARED_WIDTH + index_dst_x] = 0;
                private_blue[index_dst_y * SHARED_WIDTH + index_dst_x] = 0;
            }
        }
    }
        
	
	__syncthreads();

	if(row < height && col < width)
	{
		for (int i = 0; i <= MASK_WIDTH; i++) {
			for (int j = 0; j <= MASK_WIDTH; j++) {
				
				int index_shared = (ty + j) * SHARED_WIDTH + (tx + i);
					
				total_red += private_red[((threadIdx.y + j) * SHARED_WIDTH) + (threadIdx.x + i)];
				total_blue += private_green[((threadIdx.y + j) * SHARED_WIDTH) + (threadIdx.x + i)];
				total_green += private_blue[((threadIdx.y + j) * SHARED_WIDTH) + (threadIdx.x + i)];
			} //for z
		} //for y

		//__syncthreads();
		image_output[(row * width + col) * 3] = total_red / (MASK_WIDTH*MASK_WIDTH);
		image_output[(row * width + col) *3 + 1] = total_blue / (MASK_WIDTH*MASK_WIDTH);
		image_output[(row * width + col) * 3 + 2] = total_green / (MASK_WIDTH*MASK_WIDTH);
	}
}


__global__ void test(unsigned char* in, unsigned char* out, int width, int height, int dim ){
	int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int index = row * width + col;
    if(index < dim)
    {
		out[index*3] = in[index*3];
		out[index*3+1] = in[index*3+1];
		out[index*3+2] = in[index*3+2];
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

	int x = image->x;
	int y = image->y; 
	int dim = x*y;

	unsigned char* image_vect = Image_vector(image,dim);
	unsigned char* image_output_vect = Image_vector(image_output,dim);

	unsigned char* image_par;
        unsigned char* image_output_par;
	
	// Alloc CPU memory
	//unsigned char* image_output_vect = (unsigned char*)malloc (sizeof(unsigned char)*dim*3);
	// Alloc space for device copies of a, b, c //////////////////
	cudaMalloc((void**)&image_par, sizeof(unsigned char)*dim*3);
	cudaMalloc((void**)&image_output_par, sizeof(unsigned char)*dim*3);

	// Copy inputs to device ////////////////////////////////////////////////////////////////
	cudaMemcpy(image_par, image_output_vect, sizeof(unsigned char)*dim*3 ,cudaMemcpyHostToDevice );
	cudaMemcpy(image_output_par, image_output_vect, sizeof(unsigned char)*dim*3 ,cudaMemcpyHostToDevice );

	// Initialize dimGrid and dimBlocks
	// create a grid with (32 / columns) number of columns and (32 / lines) number of rows
	// the ceiling function makes sure there are enough to cover all elements 
	dim3 dimGrid(ceil((float)x / TILE_WIDTH), ceil((float)y / TILE_WIDTH), 1);

	// create a block with 32 columns and 32 rows
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	// Launch smothing_filter_Paralelo() kernel on GPU with a grid and block as input ////
	smothing_filter_Paralelo<<<dimGrid, dimBlock>>>(image_par, image_output_par,y,x,dim);
	//cudaCheckError();
	// Copy result to local array //////////////////////////////////////////////
	cudaMemcpy(image_output_vect, image_output_par, sizeof(unsigned char)*dim*3, cudaMemcpyDeviceToHost);

	//printf("oii %d %d %d ", image_vect[0], image_vect[1], image_vect[2]);
	//printf("oi %d %d %d ", image_output_vect[0], image_output_vect[1], image_output_vect[2]);
	vector_Image(image_output,image_output_vect ,dim);

	writePPM(image_output);

	cudaFree(image_par);
	cudaFree(image_output_par);

	free(image);
	free(image_output);	
}
