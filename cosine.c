#include <stdio.h>
#include <math.h>     /* for sqrt() */
#include <stdlib.h>  /* for rand() */
#include <sys/time.h>    /* for gettimeofday */
#include <time.h>

const int N=4096;   /* Number of Vectors */


double rtclock()
{
   struct timezone Tzp;
   struct timeval Tp;

   int stat;
   stat = gettimeofday (&Tp, &Tzp);

   if (stat != 0) printf("Error return from gettimeofday: %d",stat);

   return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

int main (int argc, char* argv[]) 
{
  double x[N];   /* X coordinates of N points (vectors) */
  double y[N];   /* Y coordinates of N points (vectors) */
  double** sim;  /* pairwise cosine similarity of N points (vectors) sim[N][N] */
  /* Use malloc to avoid stack overflow */
  double vlen[N]; /* vector length of the N points (vectors) */
  int i,j; 

  sim= malloc(sizeof(double*) *N);
  for (i=0; i<N; i++)
    sim[i]=malloc(sizeof(double)*N); 

  /* initialize random vectors */
  /* Do not parallelize this loop */
  for (i=0; i<N; i++) 
  {
      x[i] = (double) rand() / (double) RAND_MAX;
      y[i] = (double) rand() / (double) RAND_MAX;
  }

  /* calculate the vector length */
  for (i=0; i<N; i++)
  {
      vlen[i] = sqrt(x[i]*x[i]+y[i]*y[i]);
  }
  
  double start_cpu = rtclock();
  /* calculate pairwise similarity */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++) 
    {
      sim[i][j] = (x[i]*x[j]+y[i]*y[j])/(vlen[i]*vlen[j]);      
    }

  double end_cpu = rtclock();
  printf("total time is %lf\n",(double)(end_cpu-start_cpu));  
/*
  for (i=0; i<N; i++)
  {
      printf("<x,y>=<%f,%f>, vlen=%f\n", x[i],y[i],vlen[i]);
  }
*/
/*  for (i=0; i<N; i++)
  {
    for (j=0; j<N; j++) 
    {
      printf("%.2f ", sim[i][j]);
    }
    printf("\n");
  }
*/
}

