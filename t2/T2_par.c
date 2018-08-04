#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void producer_consumer_ser(int *buffer, int size, int *vec, int n, int thread_count) {
	int i, j;
	long long unsigned int sum = 0;
	//long long unsigned int k = 0;

	for(i=0;i<n;i++) {
		if(i % 2 == 0) {	// PRODUTOR
			for(j=0;j<size;j++) {
				buffer[j] = vec[i] + j*vec[i+1];
			}
		}
		else {	// CONSUMIDOR
			for(j=0;j<size;j++) {
				sum += buffer[j];
			}
		}
	}
	printf("%llu\n",sum);
}

void producer_consumer_par(int size, int *vec, int n, int thread_count) {
	int i, j;
	long long unsigned int sum = 0;
	int *buffer;
	//Criar um novo buffer para cada loop, isto garante que um thread nao escriba sobre o buffer do outro thread, as variaveis como size, vec, n e i foram definido como variaveis compartilhadas, j e buffer sao definidos como variaveis privadas porque estas variaveis só podem ser usadas por o proprio thread, provocando que o codigo poda ser executado de forma paralela sem que outro thread afeite em seu valor. Foi feito o uso de reduction porque isto vai permitir achar uma unica soma de forma paralelizada e salvar na mesma variavel, como tambem fazer uma proteçao critical à variavel sum.
	#pragma omp parallel for num_threads(thread_count) \
		default(none) shared(size, vec, n, i) private(j, buffer) reduction(+:sum) 
	for(i=0;i<n;i++) {
		buffer = (int *)malloc(size*sizeof(int)); //Se define un novo buffer para cada loop for garantindo que o consumidor consuma só o que o anterior produtor ha produzido
		if(i % 2 == 0) {// PRODUTOR
			for(j=0;j<size;j++) {
				buffer[j] = vec[i] + j*vec[i+1];
				
			}
		}
		else {	// CONSUMIDOR
			for(j=0;j<size;j++) {
				sum += buffer[j];/*Funçao critica controlada por reduction*/
			}
		}

		free(buffer); //liberar memoria de buffer
	}
	printf("%llu\n",sum);
}

int main(int argc, char * argv[]) {
	double start, end;
	//float d;
	int i, n, size, nt;
	int *buff;
	int *vec;

	scanf("%d %d %d",&nt,&n,&size);
	
	buff = (int *)malloc(size*sizeof(int)); /* Inicializa buff com um tamanho de memoria tamanho do buffer*tamnho de um inteiro  */
	vec = (int *)malloc(n*sizeof(int)); /* Inicializa buff com um tamanho de memoria numero de valores*tamnho de um inteiro  */

	for(i=0;i<n;i++)
		scanf("%d",&vec[i]); /* ler todos os valores de entrada  */
	
	start = omp_get_wtime();
	producer_consumer_par(size, vec, n, nt);
	end = omp_get_wtime();

	printf("%lf\n",end-start);

	free(buff); /* Libera a memoria buff  */
	free(vec); /* Libera a memoria vec  */

	return 0;
}


/*

cat /proc/cpuinfo

processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 42
model name	: Intel(R) Core(TM) i5-2310 CPU @ 2.90GHz
stepping	: 7
microcode	: 0x14
cpu MHz		: 2893.569
cache size	: 6144 KB
physical id	: 0
siblings	: 4
core id		: 0
cpu cores	: 4
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 popcnt tsc_deadline_timer aes xsave avx lahf_lm epb pti retpoline tpr_shadow vnmi flexpriority ept vpid xsaveopt dtherm ida arat pln pts
bugs		: cpu_meltdown spectre_v1 spectre_v2
bogomips	: 5787.13
clflush size	: 64
cache_alignment	: 64
address sizes	: 36 bits physical, 48 bits virtual
power management:

processor	: 1
vendor_id	: GenuineIntel
cpu family	: 6
model		: 42
model name	: Intel(R) Core(TM) i5-2310 CPU @ 2.90GHz
stepping	: 7
microcode	: 0x14
cpu MHz		: 2893.569
cache size	: 6144 KB
physical id	: 0
siblings	: 4
core id		: 1
cpu cores	: 4
apicid		: 2
initial apicid	: 2
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 popcnt tsc_deadline_timer aes xsave avx lahf_lm epb pti retpoline tpr_shadow vnmi flexpriority ept vpid xsaveopt dtherm ida arat pln pts
bugs		: cpu_meltdown spectre_v1 spectre_v2
bogomips	: 5787.13
clflush size	: 64
cache_alignment	: 64
address sizes	: 36 bits physical, 48 bits virtual
power management:

processor	: 2
vendor_id	: GenuineIntel
cpu family	: 6
model		: 42
model name	: Intel(R) Core(TM) i5-2310 CPU @ 2.90GHz
stepping	: 7
microcode	: 0x14
cpu MHz		: 2893.569
cache size	: 6144 KB
physical id	: 0
siblings	: 4
core id		: 2
cpu cores	: 4
apicid		: 4
initial apicid	: 4
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 popcnt tsc_deadline_timer aes xsave avx lahf_lm epb pti retpoline tpr_shadow vnmi flexpriority ept vpid xsaveopt dtherm ida arat pln pts
bugs		: cpu_meltdown spectre_v1 spectre_v2
bogomips	: 5787.13
clflush size	: 64
cache_alignment	: 64
address sizes	: 36 bits physical, 48 bits virtual
power management:

processor	: 3
vendor_id	: GenuineIntel
cpu family	: 6
model		: 42
model name	: Intel(R) Core(TM) i5-2310 CPU @ 2.90GHz
stepping	: 7
microcode	: 0x14
cpu MHz		: 2893.569
cache size	: 6144 KB
physical id	: 0
siblings	: 4
core id		: 3
cpu cores	: 4
apicid		: 6
initial apicid	: 6
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 popcnt tsc_deadline_timer aes xsave avx lahf_lm epb pti retpoline tpr_shadow vnmi flexpriority ept vpid xsaveopt dtherm ida arat pln pts
bugs		: cpu_meltdown spectre_v1 spectre_v2
bogomips	: 5787.13
clflush size	: 64
cache_alignment	: 64
address sizes	: 36 bits physical, 48 bits virtual
power management:


*******************************************************************************************
*******************************************************************************************
*******************************************************************************************
gprof

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				Paralelo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Para arq1.in
Flat profile:

Each sample counts as 0.01 seconds.
 no time accumulated

  %   cumulative   self              self     total
 time   seconds   seconds    calls  Ts/call  Ts/call  name
  0.00      0.00     0.00        1     0.00     0.00  producer_consumer_par


Para arq2.in
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total
 time   seconds   seconds    calls  Ts/call  Ts/call  name
100.50      0.31     0.31                             main
  0.00      0.31     0.00        1     0.00     0.00  producer_consumer_par


Para arq3.in
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total
 time   seconds   seconds    calls  Ts/call  Ts/call  name
100.50      3.30     3.30                             main
  0.00      3.30     0.00        1     0.00     0.00  producer_consumer_par

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				Serial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Para arq1.in
Flat profile:

Each sample counts as 0.01 seconds.
 no time accumulated

  %   cumulative   self              self     total
 time   seconds   seconds    calls  Ts/call  Ts/call  name
  0.00      0.00     0.00        1     0.00     0.00  producer_consumer


Para arq2.in
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total
 time   seconds   seconds    calls  ms/call  ms/call  name
100.76      0.30     0.30        1   302.29   302.29  producer_consumer


Para arq3.in
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total
 time   seconds   seconds    calls   s/call   s/call  name
100.76      3.04     3.04        1     3.04     3.04  producer_consumer


Se pode ver que rodando em paralelo gasta o mesmo tempo em comparacao do serial, o programa serial gasta tempo na funcao local producer_consumer, no programa paralelo deixa todo o trabalho ao main

*******************************************************************************************
*******************************************************************************************
*******************************************************************************************
flags -O0, -O1, -O2 e -O3

Speedup = (TempoSemFlag/TempoComFlag)

Para arquivo arq1.in

SpeedUp = 2,061388796

Para -O0
1,007872724 = (0.006145/0.006097)

Para -O1
4,191678035 = (0.006145/0.001466)

Para -O2
5,338835795 = (0.006145/0.001151)

Para -O3
8,791130186 = (0.006145/0.000699)


Para arquivo arq2.in

SpeedUp = 1,840503152

Para -O0
1,010130858 = (0.307999/0.304910)

Para -O1
3,914727303 = (0.307999/0.078677)

Para -O2
5,154018642 = (0.307999/0.059759)

Para -O3
7,931372801 = (0.307999/0.038833)


Para arquivo arq3.in

SpeedUp = 1,796709552

Para -O0
1,001870809 = (3.016095/3.010463)

Para -O1
4,195377996 = (3.016095/0.718909)

Para -O2
5,478828338 = (3.016095/0.550500)

Para -O3
8,935042274 = (3.016095/0.337558)


Se pode observar que usar os flag -O0, O1, O2, O3 optimizan muito o programa serial
obtindo como resultado melhores tempos em comparaçao usando o programa paralelo
*/



