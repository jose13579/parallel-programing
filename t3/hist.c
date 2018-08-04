#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* funcao que calcula o minimo valor em um vetor */
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

/* funcao que calcula o maximo valor em um vetor */
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

/* conta quantos valores no vetor estao entre o minimo e o maximo passados como parametros */
int * count(double min, double max, int * vet, int nbins, double h, double * val, int nval) {
	int i, j, count;
	double min_t, max_t;

	for(j=0;j<nbins;j++) {
		count = 0;
		min_t = min + j*h;
		max_t = min + (j+1)*h;
		for(i=0;i<nval;i++) {
			//if(val[i]>99.40 && val[i]<=99.50 && j==0)
					//printf(" val[i]: %.2lf",val[i]);

			if( (val[i] <= max_t && val[i] > min_t) || (j == 0 && val[i] <= min_t) ) {
				count++;
			}
		}
		//printf(" min_t: %.2lf max_t: %.2lf count: %d\n",min_t,max_t,count);
		vet[j] = count;
	}

	return vet;
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

	gettimeofday(&start, NULL);

	/* chama a funcao */
	vet = count(min, max, vet, n, h, val, nval);

	gettimeofday(&end, NULL);

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
