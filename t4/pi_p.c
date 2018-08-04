#include<stdio.h>
#include<stdlib.h>
#include <sys/time.h>
#include <pthread.h>

pthread_mutex_t mutex;
long long unsigned int sum = 0;
int n_threads;
unsigned int n_val, i, thread_val;
double d, pi, x, y;
long unsigned int duracao;
struct timeval start, end;

void *monte_carlo_pi(void *thread_val) {
	unsigned int n_thread_val = (unsigned int)thread_val;
	long long unsigned int vet = 0;
	unsigned int seed = time(NULL);//returns the current calendar time

	// For each thread rand_r return a pseudo-random integer  
	// rand_r makes the sequences independent of each other
	// and with the seed null probably get the different sequences in both threads
	for (int j = 0; j < n_thread_val;j++) {
		x = ((rand_r(&seed) % 1000000)/500000.0)-1;
		y = ((rand_r(&seed) % 1000000)/500000.0)-1;
		d = ((x*x) + (y*y));
		if (d <= 1) vet += 1;
	}

	// Create the mutex to lock the global variavel
	pthread_mutex_lock(&mutex);
	sum += vet;
	pthread_mutex_unlock(&mutex);
}

int main(void) {
	scanf("%d %u",&n_threads, &n_val);

	pthread_t *thread_handles = malloc(n_threads* sizeof(pthread_t));

	//Define the number values by thread
	thread_val = n_val/n_threads;

	gettimeofday(&start, NULL);

	//Call monte carlo method per thread
	for (i = 0; i < n_threads; ++i)
	{
		pthread_create(&thread_handles[i], NULL, monte_carlo_pi, (void*)thread_val);
	}

	//Join to threads
	for (i = 0; i < n_threads; ++i)
	{
		pthread_join(thread_handles[i], NULL);
	}

	gettimeofday(&end, NULL);

	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));

	//Free memory used by threads
	free(thread_handles);


	//Compute the pi value
	pi = 4*sum/((double)n_val);
	printf("%lf\n%lu\n",pi,duracao);

	

	return 0;
}
