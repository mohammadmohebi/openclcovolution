#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Les nombres en double précision ne sont pas supportés par cette version d'OpenCL."
#endif

__kernel void convolution_kernel(__global uchar4 *input, 
                                 __constant double *filter, 
                                 __global uchar4 *output,
                                 const unsigned int lHalfK) {
                                    
 
    int i = get_global_id(0);
    int j = get_global_id(1);

    int lWidth = get_global_size(0); 
    int lHeight = get_global_size(1); 
    
    int idx = i * lHeight + j;
    
    uchar4 lInputTargetPixel = input[idx];
       
    int lFilterIndex = 0;
    double4 lSum = (double4) 0.0; 
    
    int lOffset = 0;
    int lCurrentRow = 0;
    
    for (int y = -lHalfK; y <= (int)lHalfK; y++)
    {
        lCurrentRow = idx + y * lWidth;
        for (int x = -lHalfK; x <= (int)lHalfK; x++)
        {
                lOffset = x;
                uchar4 lInputPixel = input[lCurrentRow + lOffset];
                double4 lInputPixelDouble = (double4)(lInputPixel.x, lInputPixel.y, lInputPixel.z, lInputPixel.w);
                lSum += lInputPixelDouble * filter[lFilterIndex]; 
                lFilterIndex++;
        }
    }

    output[idx] = (uchar4)(lSum.x, lSum.y, lSum.z, lInputTargetPixel.w);     
}

