#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255
#define N_THREAD 32
#define N_BINS 64

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

unsigned char*  Image_Vector ( PPMImage* image, int dim )
{
        unsigned char* Vector = ( unsigned char* ) malloc ( sizeof ( unsigned char ) * dim*3 );
		int indx = 0;
        for ( int i = 0; i < dim; i++ ) {
				// Save each value the image in a vector 
				// For the pixel 0 save theirs planes red, green and blue into the first three position in the vector
                Vector[indx++] = ( image->data[i].red *4 ) / 256;
                Vector[indx++] = ( image->data[i].green *4 ) / 256;
                Vector[indx++] = ( image->data[i].blue * 4 ) / 256;
        }
        return Vector;
}

__global__ void histogram_par(unsigned char *image, float *hist, int x, int y, int dim) {
	// Create private copies of the histogram array for each thread block
    __shared__ float private_hist[N_BINS];
        
    // Initialize the bin counters in the private copies of histogram
    // Checks if the thread position in the histogram is less than 64 bins
    if((threadIdx.x * N_THREAD + threadIdx.y) < N_BINS) 
		private_hist[threadIdx.x * N_THREAD + threadIdx.y] = 0.0;
    
    // Wait for all other threads in the block to finish
    __syncthreads();    
        
    // Calculate the row and col # of the image 
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    // Calculate the index of the image
    int index = row*y + col;
    
    // First checks if the col and row position in the image is less than x and y (image size) respectly
    if((row < x && col < y) && (index < dim)) {
	// Add one value to the private histogram
	// current position of the private histogram = 16* value plane red + 4*value plane green + value plane blue
	atomicAdd(&(private_hist[image[index*3] * 16 + image[index*3 + 1]*4 + image[index*3 + 2]]),1.0); 
    }
	
    // Wait for all other threads in the block to finish
    __syncthreads();
	
    if((threadIdx.x * N_THREAD + threadIdx.y) < N_BINS) {
	// Reads the value from the location pointed to by address in 
	// private histogram and stores the result 
	// in the global histogram
        atomicAdd(&(hist[threadIdx.x * N_THREAD + threadIdx.y]), private_hist[threadIdx.x * N_THREAD + threadIdx.y]/(float)dim);
    }
}


int main(int argc, char *argv[]) {

        if ( argc != 2 ) {
                printf ( "Too many or no one arguments supplied.\n" );
        }
        
        //Timers
        cudaEvent_t start, stop;
        cudaEventCreate ( &start );
        cudaEventCreate ( &stop );

        int i;
        char *filename = argv[1]; //get the file!;

        //Vectorize the image
        PPMImage *image = readPPM ( filename );
        int x = image->x;
        int y = image->y; 
        int dim = x*y;

        unsigned char* image_vect = Image_Vector(image,dim);
        
        // Iniatialize variables
        float *hist_par;
        unsigned char *image_par;
        
	// Alloc CPU memory
	float *hist = (float*)malloc (sizeof(float)*N_BINS);



	cudaEventRecord ( start );
	// Alloc space for device copies of a, b, c //////////////////
	cudaMalloc((void**)&image_par, sizeof(unsigned char)*dim*3);
	cudaMalloc((void**)&hist_par, sizeof(float)*N_BINS );
	//////////////////////////////////////////////////////////////
	cudaEventRecord ( stop );
        cudaEventSynchronize ( stop );
        float tempo_GPU_criar_buffer = 0;
        cudaEventElapsedTime ( &tempo_GPU_criar_buffer,start,stop );
    
    
    
	// Initialize histogram
        for ( i=0; i < N_BINS; i++ ) hist[i] = 0.0;
        
        cudaEventRecord ( start );
        // Copy inputs to device ////////////////////////////////////////////////////////////////
        cudaMemcpy(hist_par, hist, sizeof(float)*N_BINS ,cudaMemcpyHostToDevice );
	cudaMemcpy(image_par, image_vect, sizeof(unsigned char)*dim*3 ,cudaMemcpyHostToDevice );
	/////////////////////////////////////////////////////////////////////////////////////////
	cudaEventRecord ( stop );
        cudaEventSynchronize ( stop );
        float tempo_GPU_offload_enviar = 0;
        cudaEventElapsedTime ( &tempo_GPU_offload_enviar,start,stop );
        
        
	// Initialize dimGrid and dimBlocks
	// create a grid with (32 / columns) number of columns and (32 / lines) number of rows
	// the ceiling function makes sure there are enough to cover all elements 
	dim3 dimGrid(ceil((float)y / N_THREAD), ceil((float)x / N_THREAD), 1);

	// create a block with 32 columns and 32 rows
	dim3 dimBlock(N_THREAD, N_THREAD, 1);




	cudaEventRecord ( start );
	// Launch matriz_soma() kernel on GPU with a grid and block as input ////
	histogram_par<<<dimGrid, dimBlock>>>(image_par, hist_par,x,y,dim);
	/////////////////////////////////////////////////////////////////////////
	cudaEventRecord ( stop );
        cudaEventSynchronize ( stop );
        float tempo_GPU_kernel = 0;
        cudaEventElapsedTime ( &tempo_GPU_kernel,start,stop );
        
        
        
	cudaEventRecord ( start );
	// Copy result to local array //////////////////////////////////////////////
	cudaMemcpy(hist, hist_par, sizeof(float)*N_BINS, cudaMemcpyDeviceToHost);
	////////////////////////////////////////////////////////////////////////////
        cudaEventRecord ( stop );
        cudaEventSynchronize ( stop );
        float tempo_GPU_receber = 0;
        cudaEventElapsedTime ( &tempo_GPU_receber,start,stop );
        
        
        for ( i = 0; i < N_BINS; i++ ) {
                printf ( "%0.3f ", hist[i]);
        }
        printf ( "\n" );
        
        //float tempo_total = tempo_GPU_criar_buffer + tempo_GPU_offload_enviar + tempo_GPU_kernel + tempo_GPU_receber;
        //printf ( "\n cria_buffer %.2f\n offload_send %.2f\n kernel %.2f\n offload_receive %.2f\n total %.2f      %f \n",tempo_GPU_criar_buffer,tempo_GPU_offload_enviar,tempo_GPU_kernel,tempo_GPU_receber,tempo_total,tempo_total*0.001);
		//printf("\n");

        free(hist);
        cudaFree(hist_par);
	cudaFree(image_par);
}
