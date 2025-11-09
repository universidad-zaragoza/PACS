__kernel void pow_of_two(
  __global float *in,
  __global float *out,
  const unsigned int count){

  int i = get_global_id(0);

  if(i < count){
    out[i] = in[i] * in[i];
  }
}
