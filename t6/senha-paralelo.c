#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>

FILE *popen(const char *command, const char *type);

char finalcmd[300] = "unzip -P%d -t %s 2>&1";
int thread_count, thread;

char filename[100];
double t_start, t_end;
int n = 500000;
pthread_mutex_t mutex;
int flag = 0;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void *find_senha(void *id_rank)
{
	// Create private variables 
	int id = (int) id_rank;
	char ret[200];
	char cmd[400];
	FILE * fp;

	// Block variables - Set the new range
	pthread_mutex_lock(&mutex);
	int n_thread = n/thread_count;
	int i = n_thread*id;
	int last_i = n_thread*(id+1);
	pthread_mutex_unlock(&mutex);

	// Loop 500000/thread_count by thread
	for(;i<last_i;i++){
		//Create cmd string
		sprintf((char*)&cmd, finalcmd, i, filename);

		// Open File
		fp = popen(cmd, "r");	

		// While this open read data
		while (!feof(fp)) {
			
			//Get data from file
			fgets((char*)&ret, 200, fp);

			//File data from file is equal to ok, password works
			if (strcasestr(ret, "ok") != NULL) {
				printf("Senha:%d\n", i);

				// Change the flag variable to true
				flag = 1;
			}
		}
		pclose(fp);

		// If flag is true leave the loop and return NULL
		if(flag==1) return(NULL);
	}
}

int main ()
{
   scanf("%d", &thread_count);
   scanf("%s", filename);

   // Create threads
   pthread_t* thread_handles = malloc(thread_count*sizeof(pthread_t));

   t_start = rtclock();

  /***********************************************/
  // Store the parameters and create the thread
  for (thread = 0; thread < thread_count; thread++)
  {
   	pthread_create(&thread_handles[thread], NULL,find_senha, (void*) thread);
  }

  // Join to each thread for continue
  for (thread = 0; thread < thread_count; ++thread)
  {
     	pthread_join(thread_handles[thread], NULL);
  }
  /***********************************************/

  t_end = rtclock();
 
  fprintf(stdout, "%0.6lf\n", t_end - t_start); 
  
  // Free memory
  free(thread_handles);  
}
