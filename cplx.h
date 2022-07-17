#ifndef COMPLEX
#define COMPLEX

//#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct _C_double_complex
{
	double _Val[2];
} _C_double_complex;

typedef _C_double_complex  _Dcomplex;

typedef _Dcomplex ComplexType;   // Complex Variables
typedef double DoubleType;       // Double Variables



//数的初始化
/* Initiation of Complex Number */
void InitComplex(ComplexType* Complex)  // Transfer of Pointer
{
	Complex->_Val[0] = 0.0;
	Complex->_Val[1] = 0.0;

}



//复数获取实部
double creal(const ComplexType a)
{
	return a._Val[0];
}

//复数获取虚部
double cimag(const ComplexType a)
{
	return a._Val[1];
}


/* Add Complex: Complex_C = Complex_A + Complex_B */
ComplexType AddComplex(const ComplexType Complex_A, const ComplexType Complex_B)
{
	ComplexType Complex_C;
	Complex_C._Val[0] = creal(Complex_A) + creal(Complex_B);
	Complex_C._Val[1] = cimag(Complex_A) + cimag(Complex_B);
	return Complex_C;
}



/* Subvision Complex: Complex_C = Complex_A - Complex_B */
ComplexType SubComplex(const ComplexType Complex_A, const ComplexType Complex_B)
{
	ComplexType Complex_C;   // Change Pointer to a Variable
	Complex_C._Val[0] = creal(Complex_A) - creal(Complex_B);
	Complex_C._Val[1] = cimag(Complex_A) - cimag(Complex_B);
	return Complex_C;
}

//复数乘复数
ComplexType _Cmulcc(ComplexType a, ComplexType b)
{
	ComplexType x;
	x._Val[0] = a._Val[0] * b._Val[0] - a._Val[1] * b._Val[1];
	x._Val[1] = a._Val[0] * b._Val[1] + a._Val[1] * b._Val[0];
	return x;
}

//复数乘实数
ComplexType _Cmulcr(ComplexType a, double b)
{
	ComplexType x;
	x._Val[0] = a._Val[0] * b;
	x._Val[1] = a._Val[1] * b;
	return x;
}

//复数开平方
ComplexType csqrt(ComplexType a)
{
	ComplexType res;
	double t = sqrt(creal(a)*creal(a) + cimag(a)*cimag(a));
	//printf("t of csqrt = %lf\n", t);
	res._Val[0] = sqrt((t + creal(a)) / 2.0);
	res._Val[1] = sqrt((t - creal(a)) / 2.0);
	//printf("real of csqrt = %lf\n", (t + creal(a) / 2.0));
	//printf("imag of csqrt = %lf\n", res._Val[1]);
	if (cimag(a) < 0) {
		res._Val[1] *= -1;
	}
	return res;
}


//打印复数
void PrintComplexType(const ComplexType* src)
{
	if (src->_Val[1] >= 0) {
		printf("%.4lf + %.4lfi", src->_Val[0], src->_Val[1]);
	}
	else {
		printf("%.4lf - %.4lfi", src->_Val[0], -1 * src->_Val[1]);
	}
	return;
}

#endif
