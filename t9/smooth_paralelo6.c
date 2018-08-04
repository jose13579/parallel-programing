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

__global__ void smothing_filter_Paralelo(PPMImage* image_input, PPMImage* image_output)
{
	// Creating variables
    int i, j, row, col;
    int total_red = 0, total_blue = 0, total_green = 0;
    int index_dst_y, index_dst_x, index_src_y, index_src_x;

    // Get Row and COl
    row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Create Shared block of data
    __shared__ PPMPixel shared_image[SHARED_SIZE*SHARED_SIZE];

    /*
    for (i = 0; i <= TILE_WIDTH * TILE_WIDTH; i = i + TILE_WIDTH * TILE_WIDTH)
    {
        // Get indexs of dst matrix
        index_dst_y = (threadIdx.y * TILE_WIDTH + threadIdx.x + i) / SHARED_SIZE;
        index_dst_x = (threadIdx.y * TILE_WIDTH + threadIdx.x + i) % SHARED_SIZE;

        // Get indexs of destination matrix
        index_src_y = (blockIdx.y * TILE_WIDTH) + index_dst_y - ((MASK_WIDTH-1)/2);
        index_src_x = (blockIdx.x * TILE_WIDTH) + index_dst_x - ((MASK_WIDTH-1)/2);

        //Work only if dst geral index stay into shared matrix size
        if (index_dst_y * SHARED_SIZE + index_dst_x < (SHARED_SIZE*SHARED_SIZE)) {
            // if src index stay into image save images values else save 0
            if (index_src_y >= 0 && index_src_y < image_output->y && index_src_x >= 0 && index_src_x < image_output->x){
                shared_image_data[index_dst_y * SHARED_SIZE + index_dst_x].red = image_input->data[(index_src_y * image_output->x) + index_src_x].red;
                shared_image_data[index_dst_y * SHARED_SIZE + index_dst_x].blue = image_input->data[(index_src_y * image_output->x) + index_src_x].blue;
                shared_image_data[index_dst_y * SHARED_SIZE + index_dst_x].green = image_input->data[(index_src_y * image_output->x) + index_src_x].green;
            }
            else{
                shared_image_data[index_dst_y * SHARED_SIZE + index_dst_x].red = 0;
                shared_image_data[index_dst_y * SHARED_SIZE + index_dst_x].blue = 0;
                shared_image_data[index_dst_y * SHARED_SIZE + index_dst_x].green = 0;
            }
        }
    }*/


    	if(row < image_output->y && col < image_output->x)
	{
		for(int i = -(MASK_WIDTH-1)/2; i<=(MASK_WIDTH-1)/2; i++)
		{
			for(int j = -(MASK_WIDTH-1)/2; j<=(MASK_WIDTH-1)/2; j++)
			{
				int new_col= col + i; 
				int new_row = row + j;
				
				int index_mask = new_row * image_output->x  + new_col;
					
				int shared_col = threadIdx.x + ((MASK_WIDTH-1)/2) + i;
				int shared_row = threadIdx.y + ((MASK_WIDTH-1)/2) + j;
				
				int index_shared = shared_row * SHARED_SIZE + shared_col;
					
				if(index_shared < SHARED_SIZE*SHARED_SIZE)
				{	
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

    // sync threads
    __syncthreads();

    // if row and col stay into image proceed with convolution
    if (row < image_output->y && col < image_output->x){
        for (i = 0; i < MASK_WIDTH; i++){
             for (j = 0; j < MASK_WIDTH; j++) {
                total_red += shared_image[((threadIdx.y + j) * SHARED_SIZE) + (threadIdx.x + i)].red;
                total_blue += shared_image[((threadIdx.y + j) * SHARED_SIZE) + (threadIdx.x + i)].blue;
                total_green += shared_image[((threadIdx.y + j) * SHARED_SIZE) + (threadIdx.x + i)].green;
             }
        }
        // Save data of convolution into devise image
        image_output->data[(row * image_output->x) + col].red = total_red / (MASK_WIDTH*MASK_WIDTH);
        image_output->data[(row * image_output->x) + col].blue = total_blue / (MASK_WIDTH*MASK_WIDTH);
        image_output->data[(row * image_output->x) + col].green = total_green / (MASK_WIDTH*MASK_WIDTH);
    }
}


/*
__global__ void smothing_filter_Paralelo(PPMImage* image_input, PPMImage* image_output)
{
	// Creating variables
    int i, j, row, col;
    int total_red = 0, total_blue = 0, total_green = 0;
    int index_dst_y, index_dst_x, index_src_y, index_src_x;

    // Get Row and COl
    row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Create Shared block of data
    __shared__ PPMPixel shared_image_data[SHARED_SIZE*SHARED_SIZE];

    for (i = 0; i <= TILE_WIDTH * TILE_WIDTH; i = i + TILE_WIDTH * TILE_WIDTH)
    {
        // Get indexs of dst matrix
        index_dst_y = (threadIdx.y * TILE_WIDTH + threadIdx.x + i) / SHARED_SIZE;
        index_dst_x = (threadIdx.y * TILE_WIDTH + threadIdx.x + i) % SHARED_SIZE;

        // Get indexs of destination matrix
        index_src_y = (blockIdx.y * TILE_WIDTH) + index_dst_y - ((MASK_WIDTH-1)/2);
        index_src_x = (blockIdx.x * TILE_WIDTH) + index_dst_x - ((MASK_WIDTH-1)/2);

        //Work only if dst geral index stay into shared matrix size
        if (index_dst_y * SHARED_SIZE + index_dst_x < (SHARED_SIZE*SHARED_SIZE)) {
            // if src index stay into image save images values else save 0
            if (index_src_y >= 0 && index_src_y < image_output->y && index_src_x >= 0 && index_src_x < image_output->x){
                shared_image_data[index_dst_y * SHARED_SIZE + index_dst_x].red = image_input->data[(index_src_y * image_output->x) + index_src_x].red;
                shared_image_data[index_dst_y * SHARED_SIZE + index_dst_x].blue = image_input->data[(index_src_y * image_output->x) + index_src_x].blue;
                shared_image_data[index_dst_y * SHARED_SIZE + index_dst_x].green = image_input->data[(index_src_y * image_output->x) + index_src_x].green;
            }
            else{
                shared_image_data[index_dst_y * SHARED_SIZE + index_dst_x].red = 0;
                shared_image_data[index_dst_y * SHARED_SIZE + index_dst_x].blue = 0;
                shared_image_data[index_dst_y * SHARED_SIZE + index_dst_x].green = 0;
            }
        }
    }

    // sync threads
    __syncthreads();

    // if row and col stay into image proceed with convolution
    if (row < image_output->y && col < image_output->x){
        for (i = 0; i < MASK_WIDTH; i++){
             for (j = 0; j < MASK_WIDTH; j++) {
                total_red += shared_image_data[((threadIdx.y + j) * SHARED_SIZE) + (threadIdx.x + i)].red;
                total_blue += shared_image_data[((threadIdx.y + j) * SHARED_SIZE) + (threadIdx.x + i)].blue;
                total_green += shared_image_data[((threadIdx.y + j) * SHARED_SIZE) + (threadIdx.x + i)].green;
             }
        }
        // Save data of convolution into devise image
        image_output->data[(row * image_output->x) + col].red = total_red / (MASK_WIDTH*MASK_WIDTH);
        image_output->data[(row * image_output->x) + col].blue = total_blue / (MASK_WIDTH*MASK_WIDTH);
        image_output->data[(row * image_output->x) + col].green = total_green / (MASK_WIDTH*MASK_WIDTH);
    }
}*/

int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }

    //double t_start, t_end;
    //int i;
    char *filename = argv[1]; //Recebendo o arquivo!;

    PPMImage *image_copy = readPPM(filename);
    PPMImage *image = readPPM(filename);

    unsigned int rows, cols, img_size;
    PPMImage *d_image, *d_image_copy;
    PPMPixel *d_pixels, *d_pixels_copy, *new_pixels;

    // Get data
    cols = image->x;
    rows = image->y;
    img_size = cols * rows;

    // Alloc structure to devise
    cudaMalloc((void **)&d_image, sizeof(PPMImage));
    cudaMalloc((void **)&d_image_copy, sizeof(PPMImage));

    // Alloc image to devise
    cudaMalloc((void **)&d_pixels, sizeof(PPMPixel) * img_size);
    cudaMalloc((void **)&d_pixels_copy, sizeof(PPMPixel) * img_size);

    // cpy stucture to devise
    cudaMemcpy(d_image, image, sizeof(PPMImage), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pixels, image->data, sizeof(PPMPixel) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_image->data), &d_pixels, sizeof(PPMPixel *), cudaMemcpyHostToDevice);

    cudaMemcpy(d_image_copy, image, sizeof(PPMImage), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pixels_copy, image->data, sizeof(PPMPixel) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_image_copy->data), &d_pixels_copy, sizeof(PPMPixel *), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil((float)cols / TILE_WIDTH), ceil((float)rows / TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Call function
    smothing_filter_Paralelo<<<dimGrid, dimBlock>>>(d_image_copy,d_image);


    new_pixels = (PPMPixel *) malloc(img_size * sizeof(PPMPixel));
    // Copy result to local array
    cudaMemcpy(image, d_image, sizeof(PPMImage), cudaMemcpyDeviceToHost);
    cudaMemcpy(new_pixels, d_pixels, sizeof(PPMPixel) * img_size, cudaMemcpyDeviceToHost);
    image->data = new_pixels;

    //Free memory
    cudaFree(d_image);
    cudaFree(d_image_copy);
    cudaFree(d_pixels);
    cudaFree(d_pixels_copy);

    writePPM(image);

    //fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);
    free(image);
    free(image_copy);
}
