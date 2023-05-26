#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*
Solution of the acoustic wave equation 
	d^2u/dx^2 + d^2u/dy^2 + d^2u/dz^2 = 1/c^2 dot d^2u/dt^2
in 3D using finite difference method.
	where u = u(x,y,z,t) - displacement vector (or acoustic pressure)
	and c - speed of sound

This program outputs .vtk file that can be opened in ParaView

compile it with
$nvcc main_proj.cu -o main_proj
then run with
$./main_proj
*/


typedef double my_type;


// Boundary conditions on the displacement vector
__global__ void define_u(my_type *d_u, int N)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;
    
    d_u[x+y*N+z*N*N] = 0;
}


// Velocity model initialization
__global__ void define_c(my_type *d_c, my_type c_max, int N)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;
    
    if (x < 105 && x > 95 && y < 105 && y > 95 && z < 105 && z > 95)
    {
        d_c[x+y*N+z*N*N] = c_max/2;
    }
    else
    {
        d_c[x+y*N+z*N*N] = c_max;
    }
}   


// 3D gaussian (initial conditions)
__global__ void define_f(my_type *f, int N, int x0, int y0, int z0)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    my_type a = 0.2;
    f[x+y*N+z*N*N]=exp(-a*( (x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0) ));
}


// Finite difference implementation
__global__ void calculate_u(my_type *dun, my_type *duc, my_type *dup, my_type *c, my_type *f, int N, my_type dt, my_type dh, my_type s)
{
	/*
    dun <==> d_u next
    duc <==> d_u current
    dup <==> d_u previous
	*/
	
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    double alpha = c[x+y*N+z*N*N]*c[x+y*N+z*N*N]*dt*dt/(dh*dh);
    if (x == blockDim.x*gridDim.x-1 || x == 0)
        return;
    if (y == blockDim.y*gridDim.y-1 || y == 0)
        return;
    if (z == blockDim.z*gridDim.z-1 || z == 0)
        return;
    dun[x+y*N+z*N*N]=s*dt*dt*f[x+y*N+z*N*N]+2*duc[x+y*N+z*N*N]-dup[x+y*N+z*N*N]+
                        alpha*(duc[x+1+y*N+z*N*N]-2*duc[x+y*N+z*N*N]+duc[x-1+y*N+z*N*N]+ 
                               duc[x+(y+1)*N+z*N*N]-2*duc[x+y*N+z*N*N]+duc[x+(y-1)*N+z*N*N]+ 
                               duc[x+y*N+(z+1)*N*N]-2*duc[x+y*N+z*N*N]+duc[x+y*N+(z-1)*N*N]);
    dup[x+y*N+z*N*N]=duc[x+y*N+z*N*N];
    duc[x+y*N+z*N*N]=dun[x+y*N+z*N*N];
}


// Source function
void ricker(my_type f0, my_type t0, my_type* time_arr, my_type* s_arr, int N_time)
{
    my_type arg;
    for (int i=0; i<N_time; i++)
    {
        arg = M_PI*f0*(time_arr[i]-t0);
        s_arr[i] = (2*arg*arg-1)*exp(-arg*arg);
    }
}


// Write solution result to .vtk file
void save_VTK(my_type* arr_3D, int N, char filename[], int i)
{
    //**************************************open_file**************************************
    FILE *writefile;
    writefile = fopen (filename, "w");
    if (writefile == NULL)
    {
        printf("error opening file\n");
        exit(-1);
    }
    //****************wrtie_VTK****************
    fprintf(writefile, "# vtk DataFile Version 2.0\n");
    fprintf(writefile, "acoustic_wave_equation_sulotion\n");
    fprintf(writefile, "ASCII\n");
    fprintf(writefile, "DATASET STRUCTURED_POINTS\n");
    fprintf(writefile, "DIMENSIONS %d %d %d\n", N, N, N);
    fprintf(writefile, "ORIGIN 0 0 0\n");
    fprintf(writefile, "SPACING 1 1 1\n");
    fprintf(writefile, "POINT_DATA %d\n", N*N*N);
    if (i == -1)
    {
        fprintf(writefile, "SCALARS vel_model float 1\n");
    }
    else
    {
        if (sizeof(my_type) == sizeof(double))
            fprintf(writefile, "SCALARS u_next double 1\n");
        else if (sizeof(my_type) == sizeof(float))
            fprintf(writefile, "SCALARS u_next float 1\n");
        else{
            fprintf(writefile, "ERROR");
            printf("ERROR! please choose float or double");}
    }
    fprintf(writefile, "LOOKUP_TABLE default\n");
    for (int i=0; i<N*N*N; i++){
        fprintf(writefile, "%.12lf ", arr_3D[i]);
    }
    //**************************************close_file**************************************
    if (fclose(writefile) == 0)
    {
        printf("writing done; file %d closed\n", i);
    }
}


int main()
{
    printf("initializing numerical model...\n");
    printf("***********************spatial_parameters***********************\n");
    const int N = 128;//[number of spatial elements along one axis]
    const my_type dh = 1; //[m]
    my_type c_max = 1000.0;//[m/s]
    printf("dh = %f [m]\n", dh);
    printf("x, y and z range = %.3f [m]\n", (my_type) N*dh);
    printf("N = %d [elements]\n", N);
    printf("****************************************************************\n");
    
    printf("*************************time_parameters*************************\n");
    my_type dt = (dh / c_max)*0.1;//[sec]
    my_type time_duration = 0.085; //[sec]
    int N_time = time_duration/dt;//[number of time elements]
    printf("dt = %f [sec]\n", dt);
    printf("time_duration = %.3f [sec]\n", time_duration);
    printf("N_time = %d [elements]\n", N_time);    
    printf("*****************************************************************\n");
    
    my_type * time_arr = (my_type*) malloc (sizeof(my_type)*N_time);
    my_type * s_arr = (my_type*) malloc (sizeof(my_type)*N_time);  
    printf("**************************initial_force**************************\n");
    int x0 = N/2;
    int y0 = N/2;
    int z0 = N/2;
    printf("source location: x=%f[m], y=%f[m], z=%f[m]\n", x0*dh, y0*dh, z0*dh);
    time_arr[0] = 0;
    for (int i=1; i<N_time; i++)
    {
        time_arr[i] = time_arr[i-1] + dt;
    }
    my_type f0 = 100; // [Hz] - ricker frequency
    my_type t0 = 0.01; // [sec] - ricker shift
    ricker(f0, t0, time_arr, s_arr, N_time);
    //**********************write_ricker_to_file**********************
    FILE* writefile;
    writefile = fopen ("ricker.txt", "w");
    if (writefile == NULL)
    {
        printf("error opening file\n");
        exit(-1);
    }
    for (int i=0; i<N_time; i++)
    {
        fprintf(writefile, "%f ", s_arr[i]);
    }
    if (fclose(writefile) == 0)
    {
        printf("writing ricker.txt done; file closed\n");
    }
    printf("*****************************************************************\n");
    printf("initializing done!\n");
    
    printf("Press ENTER to Continue\n");
    getchar();  
    
    my_type *d_u_next; // field in time t+1
    my_type *d_u_cur; // field in time t
    my_type *d_u_prev; // field in time t-1
    my_type *d_c; // velocity 
    my_type *d_f; // initial_force
    cudaMalloc( (void**) &d_u_next, sizeof(my_type)*N*N*N );
    cudaMalloc( (void**) &d_u_cur, sizeof(my_type)*N*N*N );
    cudaMalloc( (void**) &d_u_prev, sizeof(my_type)*N*N*N );
    cudaMalloc( (void**) &d_c, sizeof(my_type)*N*N*N );
    cudaMalloc( (void**) &d_f, sizeof(my_type)*N*N*N );
    
    my_type *u = (my_type *) malloc (sizeof(my_type)*N*N*N); // field_host(to write it to file)
    
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(N/8, N/8, N/8);
    
    define_f<<<numBlocks, threadsPerBlock>>>(d_f, N, x0, y0, z0);
    define_u<<<numBlocks, threadsPerBlock>>>(d_u_next, N);
    define_u<<<numBlocks, threadsPerBlock>>>(d_u_cur, N);
    define_u<<<numBlocks, threadsPerBlock>>>(d_u_prev, N);
    define_c<<<numBlocks, threadsPerBlock>>>(d_c, c_max, N);
        
    char filename[30];
    for (int i=0; i<N_time; i++)
    {
        calculate_u<<<numBlocks, threadsPerBlock>>>(d_u_next, d_u_cur, d_u_prev, d_c, d_f, N, dt, dh, s_arr[i]);
        if (i % 10 == 0)
        {
            cudaMemcpy(u, d_u_next, N*N*N*sizeof(my_type), cudaMemcpyDeviceToHost);
            snprintf(filename, sizeof(filename), "./result/u_next%d.vtk", i);
            save_VTK(u, N, filename, i);
        }
    }
    printf("cuda calculating done!\n");
    
    //save velocity model to velocity_model.vtk
    my_type *c = (my_type *) malloc (sizeof(my_type)*N*N*N); // velocity_host(to write it to file)
    cudaMemcpy(c, d_c, N*N*N*sizeof(my_type), cudaMemcpyDeviceToHost);
    snprintf(filename, sizeof(filename), "velocity_model.vtk");
    save_VTK(c, N, filename, -1);
    free(c);
    
    free(u);
    free(time_arr);
    free(s_arr);
    cudaFree(d_u_next);
    cudaFree(d_u_cur);
    cudaFree(d_u_prev);
    cudaFree(d_c);
    cudaFree(d_f);
    return 0;
}