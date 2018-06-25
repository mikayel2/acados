/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CODEGEN_PREFIX
  #define NAMESPACE_CONCAT(NS, ID) _NAMESPACE_CONCAT(NS, ID)
  #define _NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) inv_pendulum_impl_ode_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s2[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s3[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

/* inv_pendulum_impl_ode_fun:(i0[6],i1[6],i2,i3[5])->(o0[11]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[1] ? arg[1][0] : 0;
  a1=arg[0] ? arg[0][2] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[1] ? arg[1][1] : 0;
  a2=arg[0] ? arg[0][3] : 0;
  a0=(a0-a2);
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[1] ? arg[1][2] : 0;
  a3=arg[3] ? arg[3][0] : 0;
  a0=(a0-a3);
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[1] ? arg[1][3] : 0;
  a4=arg[3] ? arg[3][1] : 0;
  a0=(a0-a4);
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[1] ? arg[1][4] : 0;
  a5=arg[3] ? arg[3][2] : 0;
  a0=(a0-a5);
  if (res[0]!=0) res[0][4]=a0;
  a0=2.;
  a6=(a0*a3);
  a7=arg[3] ? arg[3][3] : 0;
  a6=(a6-a7);
  a8=arg[2] ? arg[2][0] : 0;
  a6=(a6-a8);
  if (res[0]!=0) res[0][5]=a6;
  a0=(a0*a4);
  a6=1.9620000000000001e+01;
  a0=(a0+a6);
  a6=arg[3] ? arg[3][4] : 0;
  a0=(a0-a6);
  if (res[0]!=0) res[0][6]=a0;
  a0=1.0000000000000001e-01;
  a0=(a0*a5);
  a9=-3.5000000000000000e+00;
  a0=(a0+a9);
  a7=(a7+a8);
  a8=arg[0] ? arg[0][1] : 0;
  a7=(a7*a8);
  a0=(a0-a7);
  a7=arg[0] ? arg[0][0] : 0;
  a6=(a6*a7);
  a0=(a0+a6);
  if (res[0]!=0) res[0][7]=a0;
  a0=arg[0] ? arg[0][4] : 0;
  a2=(a2*a0);
  a3=(a3+a2);
  a8=(a8*a5);
  a3=(a3+a8);
  if (res[0]!=0) res[0][8]=a3;
  a1=(a1*a0);
  a4=(a4-a1);
  a7=(a7*a5);
  a4=(a4-a7);
  if (res[0]!=0) res[0][9]=a4;
  a4=arg[1] ? arg[1][5] : 0;
  a4=(a4-a0);
  a4=(-a4);
  if (res[0]!=0) res[0][10]=a4;
  return 0;
}

CASADI_SYMBOL_EXPORT int inv_pendulum_impl_ode_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT void inv_pendulum_impl_ode_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void inv_pendulum_impl_ode_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int inv_pendulum_impl_ode_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int inv_pendulum_impl_ode_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT const char* inv_pendulum_impl_ode_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* inv_pendulum_impl_ode_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* inv_pendulum_impl_ode_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* inv_pendulum_impl_ode_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int inv_pendulum_impl_ode_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
