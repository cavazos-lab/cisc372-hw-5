#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

const int N=4096;   /* Number of Vectors */

/* do not change the following function */
double rtclock()
{
   struct timezone Tzp;
   struct timeval Tp;

   int stat;
   stat = gettimeofday (&Tp, &Tzp);

   if (stat != 0) printf("Error return from gettimeofday: %d",stat);

   return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

/* Your JOB to implement the cosine GPU kernel */
__global__ void cosine(YOURJOB)
{
    /* YOUR JOB  */ 
}

int main(int argc, char*argv[])
{
    double x[N], y[N], vlen[N];  /* data structure on CPU */
    double ** sim;
	double *d_x, *d_y, *d_sim, *d_vlen;     /* data structure for GPU */
    double * gpu_sim;                       /* data structure to be dynamically allocated */
                                            /* It holds the results copied back from GPU */
	int size = N * sizeof( double );        /* N is the number of vectors (points). */ 
    int size2 = N * N * sizeof (double);    /* N*N is the 2D similarity matrix */

    /* The following 4 variables is for configuration grid size and block size */
    /* If you let THREAD_DIMY (BLOCK_DIMY) be 1, then the grid size and block size
       are 1D, otherwise the grid size is 2D (assuming THREAD_DIMX BLOCK_DIMX are not 1*/
    int NUM_OF_THREADS_IN_X_DIM_OF_A_BLOCK; 
    int NUM_OF_THREADS_IN_Y_DIM_OF_A_BLOCK;
    int NUM_OF_BLOCKS_IN_X_DIM_OF_A_GRID; 
    int NUM_OF_BLOCKS_IN_Y_DIM_OF_A_GRID;
    int i;

    /* Dynamically allocate memory for sim[N][N] on the CPU */
    sim= (double**)malloc(sizeof(double*) *N);
    for (i=0; i<N; i++)
      sim[i]=(double*)malloc(sizeof(double)*N); 
   
    /* allocation for holding gpu results */ 
    /* convert the 2D to 1D:   sim[i][j] <--> gpu_sim[i*N+j] */
    gpu_sim= (double*) malloc (size2);

	/* allocate space for device copies */
	cudaMalloc( (void **) &d_x, size );
	cudaMalloc( (void **) &d_y, size );
	cudaMalloc( (void **) &d_vlen, size );
    cudaMalloc( (void **) &d_sim, size2);

    /* initialize with random numbers */
	for( int i = 0; i < N; i++ )
	{
      x[i] = (double) rand() / (double) RAND_MAX;
      y[i] = (double) rand() / (double) RAND_MAX;
      /* The following is for calculating |a| (|b|) term */
      vlen[i] = sqrt(x[i]*x[i]+y[i]*y[i]);
	}

	/* copy inputs to device */
	cudaMemcpy( d_x, x, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_y, y, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_vlen, vlen, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_sim, sim, size2, cudaMemcpyHostToDevice );

	/* launch the kernel on the GPU */
    NUM_OF_THREADS_IN_X_DIM_OF_A_BLOCK = YOURJOB;    /* You decide how many threads in X direction of a block*/
    NUM_OF_THREADS_IN_Y_DIM_OF_A_BLOCK = YOURJOB;    /* You decide how many threads in Y direction of a block*/
    NUM_OF_BLOCKS_IN_X_DIM_OF_A_GRID = YOURJOB;      /* You decide how many blocks in X direction of a grid*/
    NUM_OF_BLOCKS_IN_Y_DIM_OF_A_GRID = YOURJOB;    /* You decide how many blocks in Y direction of a grid*/
    dim3 dimBlock(NUM_OF_THREADS_IN_X_DIM_OF_A_BLOCK, NUM_OF_THREADS_IN_Y_DIM_OF_A_BLOCK, 1);
    dim3 dimGrid(NUM_OF_BLOCKS_IN_X_DIM_OF_A_GRID, NUM_OF_BLOCKS_IN_Y_DIM_OF_A_GRID, 1);
   
    /* start the timer */ 
    double start_cpu = rtclock();

	/* your job is to implement the cosine GPU kernel */
	cosine<<< dimGrid, dimBlock>>>(YOURJOB);

    /* error checking */    
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err){
        printf( "Error! %s \n", cudaGetErrorString(err));
        exit(-1);
    } 

    /* Ensure that the CPU codes after this line wait until GPU job finishes execution  */
    cudaThreadSynchronize();

    /* end the timer */
    double end_cpu = rtclock();
    printf("total time is %lf\n",(double)(end_cpu-start_cpu));  

	/* copy result back to host */
	cudaMemcpy( gpu_sim, d_sim, size2, cudaMemcpyDeviceToHost );
    /* do not change the following lines */
    for (int i=0; i<N; i++)
      for (int j=0; j<N; j++)
      {
        /* calculate results on the CPU */
        sim[i][j] = (x[i]*x[j]+y[i]*y[j])/(vlen[i]*vlen[j]);      
        /* if your GPU calculation is correct, you should NOT see any printf printout */
        /* if you do, you made a mistake in the cosine GPU kernel */
        if ( (sim[i][j] - gpu_sim[i*N+j]) > 1e-5 || (gpu_sim[i*N+j] - sim[i][j] > 1e-5))
        {
            printf("CPU %f and GPU %f results do not match!\n", sim[i][j], gpu_sim[i*N+j]);
            exit(-1);
        } 
    }


	/* clean up */
	cudaFree( d_x );
	cudaFree( d_y );
	cudaFree( d_vlen );
	cudaFree( d_sim );
	
	return 0;
} /* end main */
