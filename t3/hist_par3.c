#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

pthread_mutex_t mutex;
pthread_t* thread_handles;
double h, *val, max, min;
int n, nval, i, *vet, size;
long unsigned int duracao;
struct timeval start, end;
int thread;

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

/* Create a thread */
void *count_hist(void *id_rank)
{	int id = (int) id_rank;
	int j, count;
	double min_t, max_t;

	// Set the new interval
	int n_thread = nval/size;
	int i = n_thread*id;
	int last_i = n_thread*(id+1);

	// Run all the bins
	for(j=0;j<n;j++) {
		count = 0;
		min_t = min + j*h;
		max_t = min + (j+1)*h;

		// For each data belongs to interval
		for(;i<last_i;i++) {
			if( (val[i] <= max_t && val[i] > min_t) || (j == 0 && val[i] <= min_t) ) {
				// Create the mutex to lock the count part
				pthread_mutex_lock(&mutex);
				vet[j] += 1;
				pthread_mutex_unlock(&mutex);
			}
		}
		i = n_thread*id; // Set the first value
	}
}

int main(int argc, char * argv[]) {
	scanf("%d",&size);

	/* entrada do numero de dados */
	scanf("%d",&nval);
	/* numero de barras do histograma a serem calculadas */
	scanf("%d",&n);

	/* vetor com os dados */
	val = (double *)malloc(nval*sizeof(double));
	vet = (int *)malloc(n*sizeof(int));
	thread_handles = malloc(size*sizeof(pthread_t));

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

	// Store the parameters and create the thread
	for (thread = 0; thread < size; thread++)
	{
		pthread_create(&thread_handles[thread], NULL,count_hist, (void*) thread);
	}

	// Join to each thread for continue
	for (thread = 0; thread < size; ++thread)
	{
		pthread_join(thread_handles[thread], NULL);
	}

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
	free(thread_handles);

	
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
| arq2 | tempo     |  63730  |  34380 |  16147  |  20956  |  19194 |
|      -------------------------------------------------------------
|      | Speedup   |  0.851  |  1.580 |  3.362  |  2.591  |  2.828 |
|      -------------------------------------------------------------
|      | Eficência |  0.851  |  0.790 |  0.841  |  0.324  |  0.177 |
--------------------------------------------------------------------
| arq3 | tempo     |  452794 | 235936 |  158697 |  155167 | 147067 |
|      -------------------------------------------------------------
|      | Speedup   |  0.917  |  1.759 |  2.615  |  2.674  |  2.822 |
|      -------------------------------------------------------------
|      | Eficência |  0.917  | 0.879  |  0.653  |  0.334  |  0.176 |
--------------------------------------------------------------------

For
arq1
Flat profile:

Each sample counts as 0.01 seconds.
 no time accumulated

  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
  0.00      0.00     0.00        1     0.00     0.00  count
  0.00      0.00     0.00        1     0.00     0.00  max_val
  0.00      0.00     0.00        1     0.00     0.00  min_val

864
1433



arq2
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ms/call  ms/call  name    
100.41      0.05     0.05        1    50.20    50.20  count
  0.00      0.05     0.00        1     0.00     0.00  max_val
  0.00      0.05     0.00        1     0.00     0.00  min_val

52863
550067



arq3
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ms/call  ms/call  name    
100.41      0.42     0.42        1   421.71   421.71  count
  0.00      0.42     0.00        1     0.00     0.00  max_val
  0.00      0.42     0.00        1     0.00     0.00  min_val


417877
467116


Se pode observar na tabela para o arq1, arq2, arq3 que usando até 4 threads o speedup sube, mas 8 e 16 threads o speedup cresce, segue ou baixa um pouco, não melhorando.
Além a eficiencia baixa consideravelmente enquanto mais threads é usado producindo overhead e page fault, chegando a ter uma eficiencia fraca, mas um speedup bom.

Quando mais threads existe mais comunicaçao entre estas existirá, eso quer dizer que o haberá armazanamento em exceso (tempo de computaçao, de memoria, da largura de banda ou qualquer outro recurso), producindo um overhead.

Usando 1 thread se pode observar que o tempo paralelizavel é pior que o tempo serial, tendo um speedup muito fraco.

Usando 2 threads se pode observar que a eficiencia vai descrecendo, e o speedup vai crescendo.

Usando 4, 8 ou 16 threads se pode observar que a eficiencia baixa consideravelmente, mas o speedup cresce rápidamente, producindo overhead, muitos threads sao usados, e memoria e recursos.

Além disso usando mais de 4 threads, o speedup deixa de subir muito rápido e a eficiencia baixa consideravelmente, a computadora tem 4 cores (cpu), exigindo mais quando mais threads se faz uso, producindo overhead rápidamente.


Para obter o porcentual do programa que é paralelizavel
For 
arq1

T(par) = 864
T(total) = 1433
O porcentual da parte paralelizável é 60.29%


arq2
T(par) = 52863
T(total) = 550067
O porcentual da parte paralelizável é 9.61%


arq3
T(par) = 417877
T(total) = 467116
O porcentual da parte paralelizável é 8.95%

*/

