#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>

FILE *popen(const char *command, const char *type);

char finalcmd[300] = "unzip -P%d -t %s 2>&1";
int thread_count;
	   char filename[100];
double t_start, t_end;

int terminated = 0;
int n =500000;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void run()
{
	int id = omp_get_thread_num();
	FILE * fp;
	char ret[200];
	char cmd[400];
	int n_thread;
	int i;
	//int last_i; 

	#pragma omp parallel num_threads(thread_count)
	{
		#pragma omp single
		{
			for(;i<n; i++){
			sprintf((char*)&cmd, finalcmd, i, filename);

				fp = popen(cmd, "r");	
				while (!feof(fp)) {
					fgets((char*)&ret, 200, fp);
					#pragma omp task
					if (strcasestr(ret, "ok") != NULL) {
						printf("Senha:%d\n", i);
						terminated = 1;
					}
				}
				pclose(fp);

				if(terminated==1) i=n;
			}
		}
	}
}

int main ()
{
   scanf("%d", &thread_count);
   scanf("%s", filename);

   t_start = rtclock();
   run();
   t_end = rtclock();
	 
   fprintf(stdout, "%0.6lf\n", t_end - t_start);  
}
