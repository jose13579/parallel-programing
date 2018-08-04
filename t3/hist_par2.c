#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

pthread_mutex_t mutex;


/* function that calculates the minimun value in a vector */
double min_val(double * vet,int nval) {
	int i;
	double min;

	min = FLT_MAX;

	for(i=0;i<nval;i++) {
		if(vet[i] < min)
			min =  vet[i];
	}
	
	return min;
}

/* function that calculates the maximum value in a vector */
double max_val(double * vet, int nval) {
	int i;
	double max;

	max = FLT_MIN;

	for(i=0;i<nval;i++) {
		if(vet[i] > max)
			max =  vet[i];
	}
	
	return max;
}

/* Create a structure to pass save the parameters of each thread */
struct histogram {
	double min;
	double max;
	int *vet;
	int nbins;
	double h;
	double *val;
	int nval;
	int id, size;
};

/* Create a thread */
void *count_hist(void *struct_hist)
{
	int j, count;
	double min_t, max_t;
	//int indx;
	
	// Create a new structure using the input parameter struct_hist
	struct histogram *struct_histogram = struct_hist;

	// Set the new interval
	int n = struct_histogram->nval/struct_histogram->size;
	int i = n*struct_histogram->id;
	int last_i = n*(struct_histogram->id+1);

	// Declare  a array to save the histogram values
	//int *l_hist = (int*)malloc(struct_histogram->nval * sizeof(int));
	
	// Run all the bins
	for(j=0;j<struct_histogram->nbins;j++) {
		count = 0;
		min_t = struct_histogram->min + j*struct_histogram->h;
		max_t = struct_histogram->min + (j+1)*struct_histogram->h;

		// For each data belongs to interval
		for(;i<last_i;i++) {
			if( (struct_histogram->val[i] <= max_t && struct_histogram->val[i] > min_t) || (j == 0 && struct_histogram->val[i] <= min_t) ) {
				pthread_mutex_lock(&mutex);
				struct_histogram->vet[j] += 1;
				pthread_mutex_unlock(&mutex);
			}
		}
		i = n*struct_histogram->id; // Set the first value
	}

	// Free memory of dinamic arrays
	//free(l_hist);
}

/* Create a function that performs thread functions */
void histogram_parallel(double min, double max, int * vet, int nbins, double h, double * val, int nval, int size) {
 
	/* Get number of threads from command line */
	int thread_count = size;
	int thread;

	pthread_t* thread_handles = malloc(thread_count*sizeof(pthread_t));

	// create a structure to store the necessary histogram parameters
	struct histogram *struct_hist = malloc(thread_count*sizeof(struct histogram));

	// Store the parameters and create the thread
	for (thread = 0; thread < thread_count; thread++)
	{
		struct_hist[thread].min = min;
		struct_hist[thread].max = max;
		struct_hist[thread].vet = vet;
		struct_hist[thread].nbins = nbins;
		struct_hist[thread].h = h;
		struct_hist[thread].val = val;
		struct_hist[thread].nval = nval;
		struct_hist[thread].id = thread;
		struct_hist[thread].size = size;

		pthread_create(&thread_handles[thread], NULL,count_hist, &struct_hist[thread]);
	}

	// Join to each thread for continue
	for (thread = 0; thread < thread_count; ++thread)
	{
		pthread_join(thread_handles[thread], NULL);
	}

	// Free memory of dinamic arrays
	free(struct_hist);
	free(thread_handles);
}

int main(int argc, char * argv[]) {
	double h, *val, max, min;
	int n, nval, i, *vet, size;
	long unsigned int duracao, all_duracao;
	struct timeval start, end;
	struct timeval all_start, all_end;

	gettimeofday(&all_start, NULL);
	scanf("%d",&size);

	/* entrada do numero de dados */
	scanf("%d",&nval);
	/* numero de barras do histograma a serem calculadas */
	scanf("%d",&n);

	/* vetor com os dados */
	val = (double *)malloc(nval*sizeof(double));
	vet = (int *)malloc(n*sizeof(int));

	/* entrada dos dados */
	for(i=0;i<nval;i++) {
		scanf("%lf",&val[i]);
	}

	/* calcula o minimo e o maximo valores inteiros */
	min = floor(min_val(val,nval));
	max = ceil(max_val(val,nval));

	/* calcula o tamanho de cada barra */
	h = (max - min)/n;

	/***********************************************/
	gettimeofday(&start, NULL);

	/* chama a funcao */
	histogram_parallel(min, max, vet, n, h, val, nval,size);

	gettimeofday(&end, NULL);
	/***********************************************/

	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - \
	(start.tv_sec * 1000000 + start.tv_usec));

	printf("%.2lf",min);	
	for(i=1;i<=n;i++) {
		printf(" %.2lf",min + h*i);
	}
	printf("\n");

	/* imprime o histograma calculado */	
	printf("%d",vet[0]);
	for(i=1;i<n;i++) {
		printf(" %d",vet[i]);
	}
	printf("\n");
	
	
	/* imprime o tempo de duracao do calculo */
	printf("%lu\n",duracao);

	free(vet);
	free(val);
	gettimeofday(&all_end, NULL);

	all_duracao = ((all_end.tv_sec * 1000000 + all_end.tv_usec) - \
	(all_start.tv_sec * 1000000 + all_start.tv_usec));

	/* imprime o tempo de duracao do programa */
	printf("%lu\n",all_duracao);
	return 0;
}


/*

************************8
Serial
*************************
Arq1 = 954
Arq2 = 54287
Arq3 = 414993



*************************
Parallel
*************************
Tabela Speedup e eficiencia

S : Speedup
E : Eficiencia
p : Número de núcleos

S = TSerial / TParallel

Eficiencia = S / p
--------------------------------------------------------------------
|      | Threads   |     1   |    2   |     4   |     8   |    16  |
--------------------------------------------------------------------
| arq1 | tempo     |   1176  |   939  |   656   |   700   |   666  |
|      -------------------------------------------------------------
|      | Speedup   |  0.811  |  1.016 |  1.454  |  1.363  |  1.432 |
|      -------------------------------------------------------------
|      | Eficência |  0.811  |  0.508 |  0.379  |  0.364  |  0.090 |
--------------------------------------------------------------------
| arq2 | tempo     |  63730  |  34380 |  16147  |  17374  |  14235 |
|      -------------------------------------------------------------
|      | Speedup   |  0.851  |  1.580 |  3.362  |  3.125  |  3.814 |
|      -------------------------------------------------------------
|      | Eficência |  0.851  |  0.790 |  0.841  |  0.391  |  0.238 |
--------------------------------------------------------------------
| arq3 | tempo     |  452794 | 235936 | 120740  |  130355 | 117236 |
|      -------------------------------------------------------------
|      | Speedup   |  0.917  |  1.759 |  3.437  |  3.183  |  3.540 |
|      -------------------------------------------------------------
|      | Eficência |  0.917  | 0.879  |  0.859  |  0.398  |  0.221 |
--------------------------------------------------------------------

For
arq3 with 1 thread
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
100.33      0.43     0.43                             count_hist
  0.00      0.43     0.00        1     0.00     0.00  histogram_parallel
  0.00      0.43     0.00        1     0.00     0.00  max_val
  0.00      0.43     0.00        1     0.00     0.00  min_val




arq3 with 2 threads
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
100.33      0.44     0.44                             count_hist
  0.00      0.44     0.00        1     0.00     0.00  histogram_parallel
  0.00      0.44     0.00        1     0.00     0.00  max_val
  0.00      0.44     0.00        1     0.00     0.00  min_val




arq3 with 4 threads
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
100.33      0.33     0.33                             count_hist
  0.00      0.33     0.00        1     0.00     0.00  histogram_parallel
  0.00      0.33     0.00        1     0.00     0.00  max_val
  0.00      0.33     0.00        1     0.00     0.00  min_val



arq3 with 8 threads
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
100.33      0.21     0.21                             count_hist
  0.00      0.21     0.00        1     0.00     0.00  histogram_parallel
  0.00      0.21     0.00        1     0.00     0.00  max_val
  0.00      0.21     0.00        1     0.00     0.00  min_val



arq3 with 16 threads
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
100.33      0.12     0.12                             count_hist
  0.00      0.12     0.00        1     0.00     0.00  histogram_parallel
  0.00      0.12     0.00        1     0.00     0.00  max_val
  0.00      0.12     0.00        1     0.00     0.00  min_val


Se pode observar quev quando o numero de threads ou threads aumenta, a eficiencia baixa e o speedup cresce, chegando a ter uma eficiencia fraca, mas um speedup ótimo.
Quando mais threads existe mais comunicaçao entre estas existirá, eso quer dizer que existira armazanamento em exceso (tempo de computaçao, de memoria, da largura de banda ou qualquer outro recurso), producindo um overhead ou disminuçao da eficiencia por o numero de threads usados.

Usando 1 thread se pode observar que existe uma eficiencia ótima e um baixo speed up, posto que o tempo em serial é menor ao tempo paralelizavel. Não existe overhead.

Usando 2 threads se pode observar que a eficiencia vai descrecendo, e o speedup vai crescendo, o programa usa dois thread para executar o algoritmo, mas se esta producindo mais comunicaçao, entre os threads.

Usando 4, 8 ou 16 threads se pode observar que a eficiencia baixa consideravelmente, mas o speedup cresce rápidamente, producindo overhead, muitos threads sao usados, e memoria e recursos.

Além disso usando mais de 4 threads, o speedup deixa de subir muito rápido e a eficiencia baixa, porque se tem somente 4 cores (cpu) na computadora, exigindo mais quando mais threads se faz uso, producindo overhead rápidamente.


Para obter o porcentual do programa que é paralelizavel












*/

