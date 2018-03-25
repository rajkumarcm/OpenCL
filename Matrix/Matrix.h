/*--------------------------------
 Author: Rajkumar Conjeevaram Mohan
 Email: rajkumarcm@yahoo.com
 Date: 03.03.2018
 Program: Matrix
 Copyright © 2018 Rajkumar Conjeevaram Mohan. All rights reserved.
 --------------------------------*/

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <math.h>
#define PI 3.14159265

using namespace std;
typedef unsigned short int u_sint;
class Matrix
{
  private:
    void print_vector(int* vec,
                      unsigned short int size);
    
    void get_column(int** mat,
                    int col_index,
                    int rows,
                    int** columnVec);
    
    float _truncated_normal(float mean,
                            float std);
    
  public:
    
    // Empty constructor for now...
    Matrix() {}
    
    float random_normal(float mean,
                       float std);
    
    void truncated_normal(float mean,
                          float std,
                          u_sint size,
                          float ** array);
    
    int** matmul_2d(int** mat1,
                    int** mat2,
                    unsigned short int mat1_size[],
                    unsigned short int mat2_size[]);
};

float Matrix::random_normal(float mean,
                            float std)
{
    float u[2]{(float(rand()%RAND_MAX+1)/RAND_MAX),(float(rand()%RAND_MAX+1)/RAND_MAX)};
//    cout << "Printing rand u: " << u[0] << ", " << u[1] << endl;
    float r = sqrt(-2*log(u[0]));
    float theta = 2*PI*u[1];
    float x = r*cos(theta);
//    cout << "ran: " << std*x+mean << endl;
    return (std*x)+mean;
}

float Matrix::_truncated_normal(float mean,
                                float std)
{
    float x = random_normal(mean,std);
    if( (x >= mean-2*std) && (x <= mean+2*std) )
        return x;
    else
        return _truncated_normal(mean,std);
}

void Matrix::truncated_normal(float mean,
                              float std,
                              u_sint size,
                              float** array)
{
    float * temp = new float[size];
    for(u_sint i = 0; i < size; i++)
    {
        temp[i] = _truncated_normal(mean,std);
    }
    *array = temp;
}


void Matrix::print_vector(int* vec,
                          unsigned short int size)
{
    for (unsigned short int i = 0; i < size; i++)
    {
        cout << vec[i] << " ";
    }
    cout << endl;
}

void Matrix::get_column(int** mat,
                        int col_index,
                        int rows,
                        int** columnVec)
{
    int* temp = new int[rows] {};
    for (unsigned short int i = 0; i < rows; i++)
        temp[i] = mat[i][col_index];
    *columnVec = temp;
}

int** Matrix::matmul_2d(int** mat1,
                        int** mat2,
                        unsigned short int mat1_size[],
                        unsigned short int mat2_size[])
{
    int* new_size = new int[2]{ mat1_size[0],mat2_size[1] };
    int** mat = new int*[new_size[0]];
    for (unsigned short int i = 0; i < new_size[0]; i++)
        mat[i] = new int[new_size[1]];
    
    cout << "New size: [" << new_size[0] << "," << new_size[1] << "]" << endl;
    
    for (unsigned short int i = 0; i < new_size[0]; i++)
    {
        for (unsigned short int j = 0; j < new_size[1]; j++)
        {
            //Get the row vector
            int* rowVec = mat1[i];
            
            // Get the column vector
            int* columnVec;
            get_column(mat2, j, mat2_size[0], &columnVec);
            
            // Printing the row vector
            cout << "For row " << endl;
            print_vector(rowVec, mat1_size[1]);
            
            // Printing the column vector
            cout << ", printing column: " << endl;
            print_vector(columnVec, mat2_size[0]);
            
            // Inner product of the two vectors
            mat[i][j] = inner_product(rowVec, rowVec + mat1_size[1], columnVec, 0);
            
        }
    }
    return mat;
}

