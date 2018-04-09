/*
Sparse matrix multiplication on heterogeneous system
*/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <omp.h>
#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <curand.h> 
#include <time.h>

#define largeNumber 5000000
#define TRb 256
#define blocksize 64
#define gridsize 256
#define smpBloxx 64
using namespace std;

__global__ void selectPoints(bool *selected,int numRows,int sampleRows,unsigned int *randomVals){
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	//int rid=tid
	while(tid<sampleRows){
		selected[int(randomVals[tid]%numRows)]=true;
		tid+=blockDim.x*gridDim.x;
	}
}

void sampleGenerate(int *c,int *edges,int *r,int &numRows,int *out_c,int *out_edges,int *out_r,int totalRows, int &sampleSize) {
	size_t n = numRows;
	curandGenerator_t gen;
	unsigned int *devData;
	/* Allocate n floats on device */
	cudaMalloc((void **)&devData, n*sizeof(int));
	/* Create pseudo-random number generator */
	time_t ti;
	int seed=time(&ti);
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
 	/* Generate n floats on device */
 	curandGenerate(gen, devData, n);
 	bool *selected,*hselected;
 	hselected=new bool[totalRows];
 	cudaMalloc((void**)&selected,sizeof(bool)*totalRows);
 	cudaMemset(selected,0,totalRows);
 	selectPoints<<<256,64>>>(selected,totalRows,numRows,devData);
 	cudaDeviceSynchronize();
	cudaMemcpy(hselected, selected, totalRows * sizeof(bool), cudaMemcpyDeviceToHost);

 	int *indices=new int[totalRows];
 	thrust::inclusive_scan(hselected,hselected+totalRows,indices);
 	numRows=0;
 	for(int i=0;i<totalRows;i++){
 		if(hselected[i]){	
 			int indx=indices[i];
 			out_r[indx]=sampleSize;
 			numRows++;
 			int start=r[i];
 			int end=r[i+1];
 			for(int j=start;j<end;j++){
 				if(hselected[c[j]]){
 					out_c[sampleSize]=indices[c[j]];
 					out_edges[sampleSize]=edges[c[j]];
 					
 					sampleSize++;
 				}
 			}
 		}
 	}
 	out_r[numRows]=sampleSize;
	curandDestroyGenerator(gen);
	cudaFree(devData);
	cudaFree(selected);
}
/*
* CSR format struct
*/
struct csr{
	int *columns, *rows, *edges;
};
/*
* COO format struct
*/
struct coo{
	int column, row, val;
};
__device__ int entryPt = 0;

__global__ void entryPoint(int* x){
	x[0]=entryPt;
}

__device__ bool exit_flag=false;
__global__ void cost_population(int* csr_r, int *population, int size){
	/*Calculate the number of entries in a row in the sparse matrix*/
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index<size)
		population[index] = csr_r[index + 1] - csr_r[index];
}
__device__ int processed=0;

__global__ void getProgress(float *prg,int rows){
	prg[0]=processed;
	processed=0;
}

/*
* GPU code for calculating cost of multiplication
* Works similar to spmv kernel
*/
__global__ void cost_spmv(const int *population, const int *row, const int *col, int *out,int numRows){
	/*Calculate the cost vector for multiplication of the matrices*/
	int lane = threadIdx.x;
	extern __shared__ volatile int val[];
	int r = blockIdx.x;
	while(r<numRows){
		int rowStart = row[r];
		int rowEnd = row[r + 1];
		val[threadIdx.x] = 0;
		for (int i = rowStart + lane; i<rowEnd; i += 32)
			val[threadIdx.x] += population[col[i]];
		__syncthreads();
		if (lane<16)
			val[threadIdx.x] += val[threadIdx.x + 16];
		if (lane<8)
			val[threadIdx.x] += val[threadIdx.x + 8];
		if (lane<4)
			val[threadIdx.x] += val[threadIdx.x + 4];
		if (lane<2)
			val[threadIdx.x] += val[threadIdx.x + 2];
		if (lane<1)
			val[threadIdx.x] += val[threadIdx.x + 1];
		if (threadIdx.x == 0)
			out[r] = val[threadIdx.x];
		__syncthreads();
		r+=gridDim.x;
	}
}

/*
 * Binary search cpu function
 */
int b_search(int start,int ender,long long *arr,long long val){
    if(start==ender-1)
        return start;
    else{
        int mid=(start+ender)/2;
        if(arr[mid]>val)
            return b_search(start,mid,arr,val);
        else
            return b_search(mid,ender,arr,val);
    }
}

/*
* CPU code for multiplication of sparse matrices
* mat1 and mat2 are in csr notation and out in coo notation
* Improvement(s) possible- OpenMP for parallelism
*/

void cpuMultiply(csr mat1, csr mat2, vector<vector<coo> > &out, int numCols, int numRows, int numElem,int part){
	int **temp;
	int threads = omp_get_max_threads();
	temp = new int*[threads];
	for (int i = 0; i<threads; i++)
		temp[i] = new int[numCols];
#pragma omp parallel for schedule(dynamic,100)
	for (int i = 0; i<part; i++){
		int id = omp_get_thread_num();
		//Set the entries to all zero
		for (int j = 0; j<numCols; j++)
			temp[id][j] = 0;
		int start = mat1.rows[i];
		int end = mat1.rows[i + 1];
		//Main multiplication part
		for (int j = start; j<end; j++){
			int elem = mat1.edges[j];
			for (int k = mat2.rows[mat1.columns[j]]; k<mat2.rows[mat1.columns[j] + 1]; k++){
				if (mat2.columns[k] == numCols)//Check for invalid memory access
					cout << "leak" << endl;
				temp[id][mat2.columns[k]] += mat2.edges[k] * elem;
			}
		}
		//Write results to out matrix
		for (int j = 0; j<numCols; j++){
			if (temp[id][j] != 0){
				//cout<<j<<" "<<i<<" "<<temp[id][j]<<endl;
				coo tmp;
				tmp.column = j;
				tmp.row = i;
				tmp.val = temp[id][j];
				out[id].push_back(tmp);				
			}
		}
	}
	for (int i = 1; i<threads; i++){
		out[0].insert(out[0].end(), out[i].begin(), out[i].end());
		delete[] temp[i];
		out[i].clear();
	}
	delete[] temp[0];
	delete[] temp;
}
__global__ void aborter(){
	exit_flag=true;
}

__global__ void gpuMultiply(int *mat1_c, int *mat1_r, int *mat1_edges,
			    int *mat2_c, int *mat2_r, int *mat2_edges, coo *out, int numRows, int *partOut,int part){
	
	//identify the rows to be processed
	int bid = blockIdx.x;
	int tidx = threadIdx.x;
	int rid = bid+part;//row being processed
	//shared memory arrays
	__shared__ int tmat1[blocksize];//Decide the size of array
	__shared__ int tmat2[blocksize];
	while (rid<numRows){
		int start = mat1_r[rid];
		int end = mat1_r[rid + 1];
		int ttid = tidx;
		int stepStart = 0, stepEnd = TRb;
		while (stepStart<numRows){
			tmat1[tidx] = 0;
			tmat2[tidx] = 0;
			for (int i = tidx; i<TRb; i += blocksize){
				partOut[bid*TRb + i] = 0;
			}
			__syncthreads();
			ttid = tidx;
			while (ttid<(end - start)){//partrow=32 threads are launched per kernel
				//load part row elements into shared memory
				tmat1[tidx] = mat1_c[start + ttid];
				tmat2[tidx] = mat1_edges[start + ttid];
				int str2 = mat2_r[tmat1[tidx]];
				int in = str2;
				int end2 = mat2_r[tmat1[tidx] + 1];
				//Try binary search instead of linear search
				while (in<end2 && mat2_c[in]<stepStart)
					in++;
				while (in<end2 && mat2_c[in]<stepEnd){
					atomicAdd(&partOut[bid*TRb + mat2_c[in] - stepStart], tmat2[tidx] * mat2_edges[in]);//Check the address in which atomic update is taking place
					in++;
				}
				ttid += blockDim.x;
			}
			__syncthreads();
			//Write to output buffer
			tmat1[tidx] = 0;
			for (int i = tidx; i<TRb; i += blocksize)
				if (partOut[bid*TRb + i] != 0)
					tmat1[tidx]++;
			
			__syncthreads();
			//Exclusive scan to get the index where we need to write
			int offset = 1;
			int mx;
			while (offset <= blocksize / 2){
				if (tidx<blocksize / 2)
					if ((tidx&(offset - 1)) == offset - 1)//micro optimization bitwise & instead of %
						tmat1[2 * tidx + 1] += tmat1[2 * tidx + 1 - offset];
				offset = offset << 1;
				__syncthreads();
			}
			
			__syncthreads();
			mx = tmat1[blockDim.x - 1];
			__syncthreads();
			if (tidx == 0)
				tmat1[blockDim.x - 1] = 0;
			offset = offset >> 1;
			__syncthreads();
			while (offset >= 1){
				if (tidx<blocksize / 2){
					if ((tidx&(offset - 1)) == offset - 1){
						int temp = tmat1[2 * tidx + 1 - offset];
						tmat1[2 * tidx + 1 - offset] = tmat1[2 * tidx + 1];
						tmat1[2 * tidx + 1] += temp;
					}
				}
				__syncthreads();
				offset = offset >> 1;
			}
			if (tidx == 0)
				tmat2[0] = atomicAdd(&entryPt, mx);
			
			__syncthreads();
			int threadPlace = tmat2[0] + tmat1[tidx];

			for (int i = tidx; i<TRb; i += blocksize){//8=TRb/32
				if (partOut[bid*TRb + i] != 0){
					out[threadPlace].row = rid;
					out[threadPlace].val = partOut[bid*TRb + i];
					out[threadPlace].column = stepStart + i;
					threadPlace++;
				}
			}
			__syncthreads();
			stepStart += TRb;
			stepEnd += TRb;
			if (stepEnd>numRows)
				stepEnd = numRows;
		}
		rid += gridDim.x;
		__syncthreads();
	}
}
__global__ void gpuSample(int *mat1_c, int *mat1_r, int *mat1_edges,
			    int *mat2_c, int *mat2_r, int *mat2_edges, int numRows,int *partOut,int part,bool *flg){

	//identify the rows to be processed
	int bid = blockIdx.x;
	int tidx = threadIdx.x;
	int rid = bid+part;//row being processed
	//shared memory arrays
	__shared__ int tmat1[smpBloxx];//Decide the size of array
	__shared__ int tmat2[smpBloxx];
	while (rid<numRows){
		int start = mat1_r[rid];
		int end = mat1_r[rid + 1];
		int ttid = tidx;
		int stepStart = 0, stepEnd = TRb;
		if(flg[0]==true){
			asm("exit;");
		}		
		while (stepStart<numRows){
			tmat1[tidx] = 0;
			tmat2[tidx] = 0;
			for (int i = tidx; i<TRb; i += smpBloxx){
				partOut[bid*TRb + i] = 0;
			}
			__syncthreads();
			ttid = tidx;
			while (ttid<(end - start)){//partrow=32 threads are launched per kernel
				//load part row elements into shared memory
				tmat1[tidx] = mat1_c[start + ttid];
				tmat2[tidx] = mat1_edges[start + ttid];
				int str2 = mat2_r[tmat1[tidx]];
				int in = str2;
				int end2 = mat2_r[tmat1[tidx] + 1];
				//Try binary search instead of linear search
				while (in<end2 && mat2_c[in]<stepStart)
					in++;
				while (in<end2 && mat2_c[in]<stepEnd){
					atomicAdd(&partOut[bid*TRb + mat2_c[in] - stepStart], tmat2[tidx] * mat2_edges[in]);//Check the address in which atomic update is taking place
					in++;
				}
				ttid += blockDim.x;
			}
			__syncthreads();
			//Write to output buffer
			tmat1[tidx] = 0;
			for (int i = tidx; i<TRb; i +=smpBloxx)
				if (partOut[bid*TRb + i] != 0)
					tmat1[tidx]++;
			__syncthreads();
			//Exclusive scan to get the index where we need to write
			int offset = 1;
			while (offset <= smpBloxx / 2){
				if (tidx<smpBloxx / 2)
					if ((tidx&(offset - 1)) == offset - 1)
						tmat1[2 * tidx + 1] += tmat1[2 * tidx + 1 - offset];
				offset = offset << 1;
				__syncthreads();
			}
			//__syncthreads();
			if (tidx == 0)
				tmat1[blockDim.x - 1] = 0;
			offset = offset >> 1;
			__syncthreads();
			while (offset >= 1){
				if (tidx<smpBloxx / 2){
					if ((tidx&(offset - 1)) == offset - 1){
						int temp = tmat1[2 * tidx + 1 - offset];
						tmat1[2 * tidx + 1 - offset] = tmat1[2 * tidx + 1];
						tmat1[2 * tidx + 1] += temp;
					}
				}
				__syncthreads();
				offset = offset >> 1;
			}
			__syncthreads();
			stepStart += TRb;
			stepEnd += TRb;
			if (stepEnd>numRows)
				stepEnd = numRows;
		}
		
		if(tidx==0)
			atomicAdd(&processed,1);
		rid += gridDim.x;
		__syncthreads();
	}
}


int cpuSample(csr mat1, csr mat2, int numCols, int numRows,int part,cudaStream_t &stream1,bool bypass){
	int **temp;
	int threads = omp_get_max_threads();
	temp = new int*[threads];
	int *progress=new int[threads];
	for (int i = 0; i<threads; i++)
		temp[i] = new int[numCols];
	volatile bool flag=false;
	vector <vector<coo> > out(omp_get_max_threads());
	#pragma omp parallel
	//int id = omp_get_thread_num();
	for (int i = omp_get_thread_num(); i<part && !flag; i+=threads){
		if(cudaStreamQuery(stream1)==cudaSuccess && bypass){
			flag=true;
		}
		int id=omp_get_thread_num();
		progress[id]=i;
		//Set the entries to all zero
		
		for (int j = 0; j<numCols; j++)
			temp[id][j] = 0;

		int start = mat1.rows[i];
		int end = mat1.rows[i + 1];
		//Main multiplication part
		for (int j = start; j<end; j++){
			int elem = mat1.edges[j];
			for (int k = mat2.rows[mat1.columns[j]]; k<mat2.rows[mat1.columns[j] + 1]; k++){
				if (mat2.columns[k]< numCols)//Check for invalid memory access
				  temp[id][mat2.columns[k]] += mat2.edges[k] * elem;
			}
		}
		for (int j = 0; j<numCols/8; j++){
			if (temp[id][j] != 0){
				coo tmp;
				tmp.column = j;
				tmp.row = i;
				tmp.val = temp[id][j];
				out[id].push_back(tmp);				
			}
		}
		
	}
	int pgrs=0;
	for(int i=0;i<threads;i++){
		pgrs=max(progress[i],pgrs);
		delete[] temp[i];
	}
	delete[] temp;
	return pgrs;
}
float  timeEstimate(double t,double cpuDone,double gpuDone){
	float left=min(cpuDone,gpuDone);
	if(left==0)
		return double(largeNumber);
	else
		return double(t/left);
}
double sample(csr mat1,int *d_mat1_c,int *d_mat1_r,int *d_mat1_edges,int *partOut,int numRows,int numCols,cudaStream_t &stream1,bool *hflg,bool *dflg,float *prg,float percent,long long *load){
	double t1,t2;
	int workDiv=percent*load[numRows-1];
	int part=b_search(0,numRows,load,workDiv);
	
	part=(percent*numRows*3+part)/4.0;
	cout<<"sample part is "<<part<<" percent is "<<percent<< endl;
	if(workDiv-load[part-1]<load[part]-workDiv)
		part--;
	t1=omp_get_wtime();
	gpuSample<<<gridsize,smpBloxx,0,stream1>>>(d_mat1_c,d_mat1_r,d_mat1_edges,d_mat1_c, d_mat1_r, d_mat1_edges,numRows, partOut,part,dflg);
	int cpudone=cpuSample(mat1,mat1,numCols,numRows,part,stream1,false);
	//hflg[0]=true;
	cudaDeviceSynchronize();
	//getProgress<<<1,1,0,stream1>>>(prg,numRows-part);
	t2=omp_get_wtime();
	cudaDeviceSynchronize();
	hflg[0]=false;
	return t2-t1;
}

float sampleSearch(csr mat1,int *d_mat1_c,int *d_mat1_r,int *d_mat1_edges,int *partOut,int numRows,int numCols,long long *h_load){
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	float *prg;
	cudaMallocManaged(&prg,sizeof(float));
	bool *hflg,*dflg;
	cudaHostAlloc((void**)&hflg, 1*sizeof(bool), cudaHostAllocMapped);
	cudaHostGetDevicePointer(&dflg, hflg, 0);
	bool flag=false;
	double estimate;
	gpuSample<<<gridsize,smpBloxx,0,stream1>>>(d_mat1_c,d_mat1_r,d_mat1_edges,d_mat1_c, d_mat1_r, d_mat1_edges,numRows, partOut,0,dflg);
	int cpudone=cpuSample(mat1,mat1,numCols,numRows,numRows,stream1,true);
	hflg[0]=true;
	cudaDeviceSynchronize();
	getProgress<<<1,1,0,stream1>>>(prg,numRows);
	//t2=omp_get_wtime();
	cudaDeviceSynchronize();
	estimate=(cpudone*1.0)/(cpudone+prg[0]);
	hflg[0]=false;
	//cout<<"estimate is "<<estimate<<endl;
	int dir=0;
	double startSample=sample(mat1, d_mat1_c, d_mat1_r, d_mat1_edges, partOut, numRows, numCols, stream1, hflg, dflg, prg, estimate,h_load);
	double psample=sample(mat1,d_mat1_c,d_mat1_r,d_mat1_edges,partOut,numRows,numCols,stream1,hflg,dflg,prg,estimate+.1,h_load);
	double msample=sample(mat1, d_mat1_c,d_mat1_r,d_mat1_edges,partOut,numRows,numCols, stream1,hflg,dflg,prg,estimate-.1, h_load);
	if(psample<startSample){
		dir++;
		startSample=psample;
	}
	else if(msample<startSample){
		dir--;
		startSample=msample;
	}
	//cout<<estimate<<endl;
	if(dir!=0){
	estimate+=dir*.01;
	estimate=min(max(0.0,estimate),1.0);
	while(!flag && estimate>.1 && estimate<.9){	
		double dSample=sample(mat1,d_mat1_c,d_mat1_r,d_mat1_edges,partOut,numRows,numCols,stream1,hflg,dflg,prg,estimate+(dir*0.1),h_load);
		if(dSample<startSample){
			startSample=dSample;
			estimate+=.1*dir;
			estimate=min(max(0.0,estimate),1.0);
			
		}
		else
			flag=true;
		//cout<<dSample<<endl;
	}
	}/*
	psample=sample(mat1,d_mat1_c,d_mat1_r,d_mat1_edges,partOut,numRows,numCols,stream1,hflg,dflg,prg,estimate+.01,h_load);
	msample=sample(mat1, d_mat1_c,d_mat1_r,d_mat1_edges,partOut,numRows,numCols, stream1,hflg,dflg,prg,estimate-.01, h_load);
	
	if(psample<startSample){
		dir=1;
		startSample=psample;
	}
	else if(msample<startSample){
		dir=-1;
		startSample=msample;
	}
	else{
		dir=0;	
	}
	estimate+=dir*.01;
	 */
	flag=false;
	if(dir!=0){
		while(!flag && estimate>0 &&  estimate<1 ){
			double dSample=sample(mat1,d_mat1_c,d_mat1_r,d_mat1_edges,partOut,numRows,numCols,stream1,hflg,dflg,prg,estimate+(dir*0.01),h_load);
			//cout<<estimate<<" "<<startSample<<endl; 		
			//cout<<estimate+dir*0.01<<" "<<dSample<<endl;
			if(dSample<startSample){
				startSample=dSample;
				estimate+=.01*dir;
			}
			else
				flag=true;
		}
	}
	cout<<estimate<<" is the first estimate"<<endl;
	return estimate;
}

/*
 * Main function
 */
 
int main(){
	double startTime=omp_get_wtime();
	std::ios::sync_with_stdio(false);
	freopen("../soc.mtx", "r", stdin);

	int numCols, numRows, numEdges;
	double start1, finish2;//, finish1;

	cin >> numCols >> numRows >> numEdges;
	/*Matrix on host*/
	//Generate random sample using curand
	int smpRows=int(numRows/8);
	int *h_csr_r = new int[numRows + 1];
	int *h_csr_c = new int[numEdges];
	int *h_csr_edges = new int[numEdges];
	int *h_sample_r=new int[smpRows+1];
	int *h_sample_c = new int[numEdges];
	int *h_sample_edges = new int[numEdges];
	/*Format conversion*/
	int prevRow = 0;
	h_csr_r[0] = 0;
	h_sample_r[0]=0;
	int sampleSize=0;
	
	for (int i = 0; i<numEdges; i++){
		cin >> h_csr_c[i];
		h_csr_c[i]--;
		int curr_row;
		cin >> curr_row;
		curr_row--;
		cin >> h_csr_edges[i];
		//h_csr_edges[i]=1;
		while (prevRow<curr_row){
			prevRow++;
			h_csr_r[prevRow] = i;
		}
	}
	while (prevRow<numRows){
		prevRow++;
		h_csr_r[prevRow] = numEdges;
	}
	freopen("/dev/tty","r",stdin);
	sampleGenerate(h_csr_c,h_csr_edges,h_csr_r,smpRows,h_sample_c,h_sample_edges,h_sample_r,numRows,sampleSize);
	cout<<sampleSize<<" "<<smpRows<<endl;
	//Sample generated 
	/*Matrix converted to csr format*/
	int   *d_csr_r, *d_csr_c,*load,*d_population, *h_load;
	int *d_csr_edges,*d_sample_r,*d_sample_c,*d_sample_edges;
	cudaMalloc(&load, numRows*sizeof(int));
	cudaMalloc(&d_population, numRows*sizeof(int));

	/*Matrix on device*/
	cudaMalloc(&d_csr_r, (numRows + 1)*sizeof(int));
	cudaMalloc(&d_csr_c, numEdges*sizeof(int));
	cudaMalloc(&d_csr_edges, numEdges*sizeof(int));
	
	/*Sample on device*/
	cudaMalloc(&d_sample_r, (smpRows + 1)*sizeof(int));
	cudaMalloc(&d_sample_c, sampleSize*sizeof(int));
	cudaMalloc(&d_sample_edges, sampleSize*sizeof(int));

	cout<<numEdges<<" "<<numRows<<" are the number of edges and rows"<<endl;
	// cudaMemcpy(destination,source,size, cudaMemcpy* )
	h_load = new int[numRows];
	for (int i = 0; i<numRows; i++){
		h_load[i] = h_csr_r[i + 1] - h_csr_r[i];
	}
	cudaStream_t stream1, stream2, stream3;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	cudaMemcpyAsync(d_csr_r, h_csr_r, (numRows + 1)*sizeof(int), cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_csr_c, h_csr_c, numEdges*sizeof(int), cudaMemcpyHostToDevice, stream2);
	cudaMemcpyAsync(d_population, h_load, numRows*sizeof(int), cudaMemcpyHostToDevice, stream3);

	cudaDeviceSynchronize();
	cost_spmv << <256, 32, 32 * sizeof(int) >> >(d_population, d_csr_r, d_csr_c, load,numRows);
	cudaMemcpyAsync(d_sample_r, h_sample_r, (smpRows + 1)*sizeof(int), cudaMemcpyHostToDevice, stream1);

	cudaMemcpyAsync(d_sample_c, h_sample_c, sampleSize*sizeof(int), cudaMemcpyHostToDevice, stream2);
	cudaMemcpyAsync(d_sample_edges, h_sample_edges, sampleSize*sizeof(int), cudaMemcpyHostToDevice, stream3);


	cudaDeviceSynchronize();

	cout << cudaGetErrorString(cudaGetLastError()) << endl;
	cudaMemcpyAsync(d_csr_edges, h_csr_edges, numEdges*sizeof(int), cudaMemcpyHostToDevice, stream2);

	cudaMemcpyAsync(h_load, load, numRows*sizeof(int), cudaMemcpyDeviceToHost, stream1);
	cudaDeviceSynchronize();
	cout << "Done" << endl;
	long long *total_load;
	total_load=new long long[numRows];
	for(int i=0;i<numRows;i++)
		total_load[i]=h_load[i];
	delete[] h_load;
	
	thrust::inclusive_scan(total_load, total_load + numRows, total_load);
        
	int *partOut;
	cudaMalloc(&partOut, TRb*sizeof(int) * gridsize);//TRb columns and 256  rows dealt with at the same time	
	cudaDeviceSynchronize();
	cout<<endl;
	csr mat2;
	mat2.columns=h_sample_c;
	mat2.rows=h_sample_r;
	mat2.edges=h_sample_edges;
	float percent=sampleSearch(mat2,d_sample_c,d_sample_r,d_sample_edges,partOut,smpRows,smpRows,total_load);

	percent=min(percent,1.0);
	cout<<percent<<" is the split"<<endl;
	cudaFree(d_sample_c);
	cudaFree(d_sample_r);
	cudaFree(d_sample_edges);
	
	long long workDiv=((percent)*total_load[numRows-1]);
	cout<<workDiv<<"is the division of work"<<endl;

	int part=b_search(0,numRows,total_load,workDiv);
	part=(part+3*numRows*percent)/4;
	if(workDiv-total_load[part-1]<total_load[part]-workDiv)
		part--;
	cout<<part<<" is the part"<<endl;
	csr mat1;
	mat1.columns = h_csr_c;
	mat1.rows = h_csr_r;
	mat1.edges = h_csr_edges;
	vector<vector<coo> > out(omp_get_max_threads());
	coo *h_out, *d_out;
	//Allocate pinned memory which can be written from the gpu/device
	//float cpu_flops=2*2.7*16;
	//float gpu_flops=768*.8;
	cudaHostAlloc((void**)&h_out, 50 * numEdges*sizeof(coo), cudaHostAllocMapped);
	cudaHostGetDevicePointer(&d_out, h_out, 0);
	start1 = omp_get_wtime();
	
	gpuMultiply <<< gridsize, blocksize >>>(d_csr_c, d_csr_r, d_csr_edges, d_csr_c, d_csr_r, d_csr_edges, d_out, numRows, partOut,part);
	cpuMultiply(mat1, mat1, out, numCols, numRows, numEdges,part);
	cudaDeviceSynchronize();
	finish2 = omp_get_wtime();
        
	cout << "Done with time "<<finish2 - start1 << endl;
	//entryPrint << <1, 1 >> >();
	cudaDeviceSynchronize();
        
	int *gpuPart;
	cudaMallocManaged(&gpuPart,1*sizeof(int));
	entryPoint<<<1,1>>>(gpuPart);
	cudaDeviceSynchronize();
	//cout << gpuPart[0] << " " << out[0].size()<<"\n"<< gpuPart[0]+out[0].size() << " is the size of output" << endl;
	
	/*Call cudaFree */
	
	///freopen("out.mtx", "w", stdout);
	//for(int j=0;j< gpuPart[0];j++)
	//	cout<<h_out[j].column<<" "<<h_out[j].row<<" "<<h_out[j].val<<endl;
	/*
	  for(int j=0;j<out[0].size();j++){
		cout<<out[0][j].column<<" "<<out[0][j].row<<" "<<out[0][j].val<<endl;
		}*/
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
	cudaDeviceReset();
	double endTime=omp_get_wtime();
	cout<<"Total time is "<<endTime-startTime<<endl;
	return 0;
}
