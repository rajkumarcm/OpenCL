//
//  matmul.cl
//  SampleOpenCL
//
//  Created by Rajkumar Conjeevaram Mohan on 17/03/2018.
//  Copyright © 2018 Rajkumar Conjeevaram Mohan. All rights reserved.
//

typedef unsigned short int u_sint;

void get_row(__constant float * A,
               u_sint batch,
               u_sint r_index,
               u_sint height,
               u_sint width,
               __private float ** array)
{
    float temp[width];
    u_sint s_i = (batch*(height*width))+(r_index*width);
    u_sint e_i = s_i + width;
    for(u_sint i = s_i; i < e_i; i++)
    {
        temp[i] = A[i];
        printf("%.2f, ",A[i]);
    }
    printf("\n");
    *array = temp;
    //return array;
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

__kernel //__attribute__((reqd_work_group_size(8,0,0)))
void matmul(__constant float * A,
            __constant float * B,
            __constant u_sint * size_A,
            __constant u_sint * size_B,
            __global float * result) {
    
    u_sint w_id = (u_sint)get_group_id(0);
    u_sint l_id = (u_sint)get_local_id(0);
    u_sint global_id = (u_sint)get_global_id(0);
   
//    u_sint local_id = w_id * size_A[1] * size_B[1] + l_id;
    
    
    u_sint m = size_A[1];
    u_sint n = size_B[1];
    
    u_sint row;
    u_sint col;
    
    // Do not type cast as I have deliberately
    // left this to achieve non-decimal value.
    row = l_id/n;
    col = (l_id-(row*n));
    
    
    float * vec1;
    get_row(A,w_id,row,size_A[1],size_A[2],&vec1);
    float * vec2 = get_column(B,w_id,col,size_B[0],size_B[1]);
    
    printf("group_id: %d, local_id: %d, value: %.2f\n",
           w_id,
           l_id,
           vec1[2]);
    
    result[global_id] = inner_product(vec1,vec2,size_A[2]);
    
}
