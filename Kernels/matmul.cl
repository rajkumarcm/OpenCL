//
//  matmul.cl
//  SampleOpenCL
//
//  Created by Rajkumar Conjeevaram Mohan on 17/03/2018.
//  Copyright © 2018 Rajkumar Conjeevaram Mohan. All rights reserved.
//

typedef unsigned short int u_sint;

float* get_row(__constant float * A,
               u_sint batch,
               u_sint r_index,
               u_sint height,
               u_sint width)
{
    float array[width];
    u_sint s_i = batch*(height*width);
    u_sint e_i = s_i + (height*width);
    for(u_sint i = s_i; i < e_i; i++)
    {
        array[i] = A[i];
    }
    return array;
}

float* get_column(__constant float * A,
                  u_sint batch,
                  u_sint c_index,
                  u_sint height,
                  u_sint width)
{
    float array[height];
    u_sint s_i = batch*(height*width)+c_index;
    u_sint skip = width;
    u_sint index = s_i;
    u_sint count = 0;
    
    while(count < height)
    {
        array[count] = A[index];
        index += skip;
        count++; // Keeps a count of height added to column array
    }
    return array;
}

float inner_product(float* vec1,
                    float* vec2,
                    u_sint size)
{
    float result = 0;
    for(u_sint i = 0; i < size; i++)
    {
        result += vec1[i] * vec2[i];
    }
    return result;
}

__kernel
void matmul(__constant float * A,
            __constant float * B,
            __constant u_sint * size_A,
            __constant u_sint * size_B,
            __global float * result) {
    
    size_t w_id = get_group_id(0);
    size_t l_id = get_local_id(0);
    
    u_sint local_id = w_id * size_A[2] * size_B[0] + l_id;

    u_sint m = size_A[1];
    u_sint n = size_B[1];
    
    u_sint row = (l_id/m);
    u_sint col = (l_id-((row-1)*n));
    
    float * vec1 = get_row(A,w_id,row,size_A[1],size_A[2]);
    float * vec2 = get_column(B,w_id,col,size_B[0],size_B[1]);
    
    
    result[local_id] = inner_product(vec1,vec2,size_A[2]);
}
