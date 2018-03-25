
/* OpenCL_h */
//OpenCL kernel which is run for every work item created.
//const char *saxpy_kernel =
//
//"static float mul(float alpha, float A){\n"
//"  return alpha*A; }\n"
//
//"__kernel \n"
//"void saxpy_kernel(  \n"
//"float alpha, \n"
//" __global float *A, \n"
//" __global float *B, \n"
//" __global float *C) \n"
//"{ \n"
//" //Get the index of the work-item \n"
//" int index = get_global_id(0); \n"
//" C[index] = mul(alpha,A[index]) + B[index]; \n"
//"} \n";
