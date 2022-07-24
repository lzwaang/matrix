#ifndef MATRIX_COMPLEX
#define MATRIX_COMPLEX

#include "cplx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Complex Cell */
typedef struct matrix 
{
	int row, column;  
	ComplexType* arrayComplex;  
	
	//void (*print)(void);  //成员函数print
	
	
}Matrix;


/* Double Cell */
typedef struct
{
	int row, column;
	DoubleType* arrayDouble;
}Matrix2Double;

/* bool */
typedef enum
{
	False = 0, True = 1
}Bool;

//构造矩阵
/* Initiation of Complex Matrix */
Matrix* InitComplexMatrix(int row, int column)  // Transmiss of Pointer
{
	int size = row * column * sizeof(ComplexType);
	if (size <= 0)
	{
		printf("InitComplexMatrix ERROE: An invalid matrix!\n");
		return NULL;
	}
	
	Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
	matrix->arrayComplex = (ComplexType*)malloc(size); 			// initiate pointer
	if (matrix->arrayComplex)
	{
		matrix->row = row;                           			 //  initiate row and size
		matrix->column = column;
		for (int index = 0; index < row * column; index++)       //  initiate cell
		{
			InitComplex(matrix->arrayComplex + index);  // call InitComplex() function
		}
	}
	return matrix;
}

/* Initiation of Double Matrix */
Matrix2Double* InitDoubleMatrix(int row, int column)
{
	int size = row * column * sizeof(DoubleType);
	if (size <= 0)
	{
		printf("InitDoubleMatrix ERROE: An invalid matrix!\n");
		return NULL;
	}
	Matrix2Double* matrix = (Matrix2Double*)malloc(sizeof(Matrix2Double));
	matrix->arrayDouble = (DoubleType*)malloc(size);
	if (matrix->arrayDouble)
	{
		matrix->row = row;
		matrix->column = column;
		for (int row_i = 0; row_i < row; ++row_i)
		{
			for (int column_j = 0; column_j < column; ++column_j)
			{
				matrix->arrayDouble[row_i * matrix->column + column_j] = 0.0;
			}
		}

	}
	return matrix;
}



//free单独矩阵：
/* Validity of Complex Matrix */
Bool IsNullComplexMatrix(const Matrix* matrix)
{
	int size = matrix->row * matrix->column;

	if (size <= 0 || matrix->arrayComplex == NULL)
	{
		return True;
	}
	return False;
}
/* Free Memory of Complex Matrix */
void DestroyComplexMatrix(Matrix* matrix)
{
	if (!IsNullComplexMatrix(matrix))    // Nested Call of IsNullComplexMatrix()
	{
		free(matrix->arrayComplex);      // Pair: malloc--free
		matrix->arrayComplex = NULL;
	}
	matrix->row = matrix->column = 0;
	free(matrix);
}


/* Validity of Double Matrix */
Bool IsNullDoubleMatrix(const Matrix2Double* matrix)
{
	int size = matrix->row * matrix->column;

	if (size <= 0 || matrix->arrayDouble == NULL)
	{
		return True;
	}
	return False;
}
/* Free Memory of Double Matrix */
void DestroyDoubleMatrix(Matrix2Double* matrix)
{
	if (!IsNullDoubleMatrix(matrix))
	{
		free(matrix->arrayDouble);
		matrix->arrayDouble = NULL;
	}
	matrix->row = matrix->column = 0;
	free(matrix);
}


//free矩阵数组：
	/* Free Memory of Complex Matrice Array */
void DestroyComplexMatrixArray(Matrix matrixArray[], int num)  // Array Transfer--->Pointer Transfer
{
	if (num)      // if no matrix
	{
		for (int i = 0; i < num; i++)
		{
			DestroyComplexMatrix(&matrixArray[i]);  // Nested Call of DestroyComplexMatrix()
		}
	}
}


/* Free Memory of Double Matrix Array */
void DestroyDoubleMatrixArray(Matrix2Double* matrixArray, int num)
{
	if (num)  // if no cell
	{
		for (int i = 0; i < num; i++)
		{
			DestroyDoubleMatrix(&matrixArray[i]);
		}
	}
}


//获取实数矩阵行、列、元胞数目
int DoubleMatrixRow(const Matrix2Double* matrix)
{
	return matrix->row;
}

int DoubleMatrixColumn(const Matrix2Double* matrix)
{
	return matrix->column;
}

int DoubleMatrixSize(const Matrix2Double* matrix)
{
	return DoubleMatrixRow(matrix) * DoubleMatrixColumn(matrix);   // Size Refers to Numbers of Cell of Matrix
}



//获取复数矩阵行、列、元胞数目
/* Return Matrix Row Size */
int MatrixRow(const Matrix* matrix)
{
	return matrix->row;
}
/* Return Matrix Column Size */
int MatrixColumn(const Matrix* matrix)
{
	return matrix->column;
}
/* Return Complex Matrix Size */
int MatrixSize(const Matrix* matrix)
{
	return MatrixRow(matrix) * MatrixColumn(matrix);   // Size Refers to Numbers of Cell of Matrix
}


//矩阵加法
/* Complex Matrix Add: matrixC = matrixA + matrixB */
Matrix* AddMatrix(const Matrix* matrix_A, const Matrix* matrix_B)
{
	// Size Incompatability
	if (matrix_A->row != matrix_B->row || matrix_A->column != matrix_B->column)
	{
		printf("ERROE: An incompatable matrix!\n");
		return NULL;
	}
	// Validity of Known Matrice
	else if (IsNullComplexMatrix(matrix_A) || IsNullComplexMatrix(matrix_B))
	{
		printf("ERROE: An invalid matrix!\n");
		return NULL;
	}
	else
	{
		Matrix* matrix_C = InitComplexMatrix(matrix_A->row, matrix_B->column);
		ComplexType tempCell_C;
		int indexC;
		for (int row_i = 0; row_i < matrix_C->row; row_i++)
		{
			for (int column_j = 0; column_j < matrix_C->column; column_j++)
			{
				tempCell_C = AddComplex(matrix_A->arrayComplex[row_i * matrix_A->column + column_j], matrix_B->arrayComplex[row_i * matrix_B->column + column_j]);
				indexC = row_i * matrix_C->column + column_j;
				matrix_C->arrayComplex[indexC] = tempCell_C;
				// matrix_C->arrayComplex[i * matrix_C->row + j]._Val[0] = \
				// creal(matrix_A->arrayComplex[i * matrix_A->row + j]) + creal(matrix_B->arrayComplex[i * matrix_A->row + j]);
				// matrix_C->arrayComplex[i * matrix_C->row + j]._Val[1] = \
				// cimag(matrix_A->arrayComplex[i * matrix_A->row + j]) + cimag(matrix_B->arrayComplex[i * matrix_A->row + j]);
			}
		}
		return matrix_C;
	}
	
}




//实数矩阵转化成复数矩阵
Matrix* DtoCMatrix( const Matrix2Double* matrix_B)
{
	
	int row = matrix_B->row;
	int column = matrix_B->column;
	Matrix* matrix_A = InitComplexMatrix(row, column);
	if (matrix_A->arrayComplex) {
		matrix_A->row = row;
		matrix_A->column = column;
		int i = 0;
		for (; i < row*column; i++) {
			matrix_A->arrayComplex[i]._Val[0] = matrix_B->arrayDouble[i];
		}
	}
	return matrix_A;

}





//实矩阵cos
Matrix2Double *cos_D_M(Matrix2Double* matrix)
{
	int row = matrix->row;
	int col = matrix->column;
	//Matrix2Double* res = (Matrix2Double*)malloc(sizeof(matrix));
	Matrix2Double *res = InitDoubleMatrix(row, col);
	int i = 0;
	for (i = 0; i < row*col; i++) {
		res->arrayDouble[i] = cos(matrix->arrayDouble[i]);
	}
	return res;
}

//实矩阵sin
Matrix2Double *sin_D_M(Matrix2Double* matrix)
{
	int row = matrix->row;
	int col = matrix->column;
	//Matrix2Double *res = (Matrix2Double*)malloc(sizeof(matrix));
	Matrix2Double *res = InitDoubleMatrix(row, col);
	int i = 0;
	for (i = 0; i < row*col; i++) {
		res->arrayDouble[i] = sin(matrix->arrayDouble[i]);
	}
	return res;
}




//复矩阵点乘复数
/* Point Multiple: Complex matrixB = Complex matrixA .* Complex_c */
Matrix* MatrixPointMulComplex(const Matrix* matrixA, ComplexType c)
{
	
	if (IsNullComplexMatrix(matrixA))
	{
		printf("MatrixPointMulComplex ERROE: An invalid matrix!\n");
		return NULL;
	}
	else
	{
		Matrix* matrixB = InitComplexMatrix(matrixA->row, matrixA->column);
		int row_i, column_j;
		for (row_i = 0; row_i < matrixB->row; ++row_i)
			for (column_j = 0; column_j < matrixB->column; ++column_j)
			{
				matrixB->arrayComplex[row_i * matrixB->column + column_j] =
					_Cmulcc(matrixA->arrayComplex[row_i * matrixA->column + column_j], c);
			}
		return matrixB;
	}
}

//复矩阵点乘实数
/* Point Multiple: Complex matrixB = Complex matrixA .* Double_c */
Matrix* MatrixPointMulDouble(const Matrix* matrixA, DoubleType c)
{
	if (IsNullComplexMatrix(matrixA))
	{
		printf("MatrixPointMulDouble ERROE: An invalid matrix!\n");
		return NULL;
	}
	Matrix* matrixB = InitComplexMatrix(matrixA->row, matrixA->column);
	int row_i, column_j;
	for (row_i = 0; row_i < matrixB->row; ++row_i)
		for (column_j = 0; column_j < matrixB->column; ++column_j)
		{
			matrixB->arrayComplex[row_i * matrixB->column + column_j] = \
				_Cmulcr(matrixA->arrayComplex[row_i * matrixA->column + column_j], c);
		}
	return matrixB;
}



//实矩阵点乘实数
/* Point Multiple: Double matrixB = Double matrixA .* Double_c */
Matrix2Double* DoubleMatrixPointMulDouble(const Matrix2Double* matrixA, DoubleType c)
{
	if (IsNullDoubleMatrix(matrixA))
	{
		printf("DoubleMatrixPointMulDouble ERROE: An invalid matrix!\n");
		return NULL;
	}
	Matrix2Double* matrixB = InitDoubleMatrix(matrixA->row, matrixA->column);
	int row_i, column_j;
	for (row_i = 0; row_i < matrixB->row; ++row_i)
		for (column_j = 0; column_j < matrixB->column; ++column_j)
		{
			matrixB->arrayDouble[row_i * matrixB->column + column_j] = \
				matrixA->arrayDouble[row_i * matrixA->column + column_j] * c;
		}
	return matrixB;
}


//复数矩阵乘法
/*
  Complex Matrix Multiple: matrixC = matrixA * matrixB
*/
Matrix* MatrixMulMatrix(const Matrix* matrixA, const Matrix* matrixB)
{
	if (IsNullComplexMatrix(matrixA) || IsNullComplexMatrix(matrixB))  // Validity of Known Matrice
	{
		puts("MatrixMulMatrix ERROE: An invalid matrix!\n");
		return NULL;
	}

	else if (matrixA->column != matrixB->row)  // Incompatability of Size
	{
		puts("MatrixMulMatrix ERROE: An incompatable matrix!\n");
		return NULL;
	}
	else
	{
		Matrix *matrixC = InitComplexMatrix(matrixA->row, matrixB->column);
		int row_i, column_j, ij;
		int indexA, indexB, indexC;
		ComplexType tempCell_C;   // Save Temp Value of Cell of Matrix C

		for (row_i = 0; row_i < matrixC->row; ++row_i)
		{
			for (column_j = 0; column_j < matrixC->column; ++column_j)
			{
				tempCell_C._Val[0] = 0;
				tempCell_C._Val[1] = 0;
				for (ij = 0; ij < matrixA->column; ++ij)
				{
					indexA = row_i * matrixA->column + ij;
					indexB = ij * matrixB->column + column_j;
					tempCell_C = AddComplex(tempCell_C, \
						_Cmulcc(matrixA->arrayComplex[indexA], matrixB->arrayComplex[indexB]));
					/*					temp._Val[0] = creal(temp) + creal(_Cmulcc(matrixA->arrayComplex[indexA], matrixB->arrayComplex[indexB]));
										temp._Val[1] = cimag(temp) + cimag(_Cmulcc(matrixA->arrayComplex[indexA], matrixB->arrayComplex[indexB]));
					*/
				}
				indexC = row_i * matrixC->column + column_j;
				matrixC->arrayComplex[indexC] = tempCell_C;
			}
		}
		return matrixC;
	}
}


//复数矩阵普通转置
/* Change Postion: [i, j] -- > [j, i] */
Matrix* TransPoseMatrix(const Matrix* matrixA)			// Tip:if A==U * U', then A'==A
{																		// So this function is not a necessity
	if (IsNullComplexMatrix(matrixA))
	{
		puts("TransPoseMatrix ERROE: An invalid matrix!\n");
		return NULL;
	}

	else
	{
		Matrix* matrixB = InitComplexMatrix(matrixA->column, matrixA->row);
		for (int row_i = 0; row_i < matrixA->row; row_i++)
		{
			for (int column_j = 0; column_j < matrixA->column; column_j++)
			{
				matrixB->arrayComplex[column_j * matrixB->column + row_i] = \
					matrixA->arrayComplex[row_i * matrixA->column + column_j];   // Attention!!!
				/*matrixB->arrayComplex[column_j * matrixB->column + row_i]._Val[0] = \
					matrixA->arrayComplex[row_i * matrixA->column + column_j]._Val[0];   // Attention!!!
				matrixB->arrayComplex[column_j * matrixB->column + row_i]._Val[1] = \
					matrixA->arrayComplex[row_i * matrixA->column + column_j]._Val[1]; */
			}
		}
		return matrixB;
	}
}



//共轭转置
/* Conjugate Complex: Complex_B = creal(Complex_A) - cimag(Complpex_A) */
ComplexType ConjugateComplex(const ComplexType Complex_A)
{
	ComplexType Complex_B;
	Complex_B._Val[0] = creal(Complex_A);
	Complex_B._Val[1] = cimag(Complex_A) * (-1.0);
	return Complex_B;
}

/* Transposition: Complex matrixB = Complex matrixA' */
void TransMatrix(const Matrix* matrixA, Matrix* matrixB)			// matrixB is transposal matrix
{
	if (IsNullComplexMatrix(matrixA))
	{
		puts("TransMatrix ERROE: An invalid matrix!\n");
		return;
	}
	else if ((matrixA->row != matrixB->column) || (matrixA->column != matrixB->row))
	{
		puts("TransMatrix ERROE: An incompatable matrix!\n");
		return;
	}
	else
	{
		for (int row_i = 0; row_i < matrixA->row; row_i++)
		{
			for (int column_j = 0; column_j < matrixA->column; column_j++)
			{
				// Transpose: position and sign w.r.t a Complex Number; only position w.r.t a Double Number
				matrixB->arrayComplex[column_j * matrixB->column + row_i] = \
					ConjugateComplex(matrixA->arrayComplex[row_i * matrixA->column + column_j]); // Attention!!!
				// OR
/*
				matrixB->arrayComplex[column_j * matrixB->column + row_i]._Val[0] = \
					creal(matrixA->arrayComplex[row_i * matrixA->column + column_j]);
				matrixB->arrayComplex[column_j * matrixB->column + row_i]._Val[1] = \
					cimag(matrixA->arrayComplex[row_i * matrixA->column + column_j]) * (-1.0);
*/
			}
		}
	}
}



//实数矩阵普通转置
/* Change Postion: [i, j] -- > [j, i] */
Matrix2Double* TransPoseDoubleMatrix(const Matrix2Double* matrixA)			// Tip:if A==U * U', then A'==A
{																		// So this function is not a necessity
	if (IsNullDoubleMatrix(matrixA))
	{
		puts("TransPoseDoubleMatrix ERROE: An invalid matrix!\n");
		return NULL;
	}
	else
	{
		Matrix2Double* matrixB = InitDoubleMatrix(matrixA->column, matrixA->row);
		for (int row_i = 0; row_i < matrixA->row; row_i++)
		{
			for (int column_j = 0; column_j < matrixA->column; column_j++)
			{
				matrixB->arrayDouble[column_j * matrixB->column + row_i] = \
					matrixA->arrayDouble[row_i * matrixA->column + column_j];   // Attention!!!
			}
		}
		return matrixB;
	}
}



//行列式（仅限三阶及以下）
/* Determinant Value */
ComplexType Comput_D(const Matrix* matrix /*float *p, short int n*/)
{
	int i, j, m;          // i--row; j--column
	int loop = 0;         // Loop Number
	ComplexType result;
	result._Val[0] = 0; result._Val[1] = 0;

	if (MatrixRow(matrix) != MatrixColumn(matrix))
	{
		puts("Comput_D ERROE: Not a square matrix!\n");
		return result;
	}
	if (IsNullComplexMatrix(matrix))
	{
		printf("Comput_D ERROE: An invalid matrix!\n");
		return result;
	}

	if (matrix->row != 1)   // More Than 1D
	{
		loop = (matrix->row == 2) ? 1 : matrix->row;     // 2nd order: 1; >2nd order: row

		for (m = 0; m < loop; m++)
		{
			ComplexType mid;
			mid._Val[0] = 1; mid._Val[1] = 0;        // Subsequent--Add
		   // mid = 1;          
			for (i = 0, j = m; i < matrix->row; i++, j++)
				mid = _Cmulcc(*(matrix->arrayComplex + i * matrix->row + j % matrix->row), mid);
			result = AddComplex(result, mid);
			//          OR

			//		    result._Val[0] = creal(result) + creal(mid);    // Real Part
			//			result._Val[1] = cimag(result) + cimag(mid);    // Imaginary Part
		}

		for (m = 0; m < loop; m++)
		{
			ComplexType mid;
			mid._Val[0] = 1; mid._Val[1] = 0;        // Reverse--Sub
			for (i = 0, j = matrix->row - 1 - m + matrix->row; i < matrix->row; i++, j--)
				mid = _Cmulcc(*(matrix->arrayComplex + i * matrix->row + j % matrix->row), mid);
			result = SubComplex(result, mid);
			//          OR
			//			result._Val[0] = creal(result) - creal(mid);    // Real Part
			//			result._Val[1] = cimag(result) - cimag(mid);    // Imaginary Part
		}
		return result;
	}
	else
	{
		result = *(matrix->arrayComplex);
		return (result);
	}

}



//代数余子式(仅限三阶及以下)
/* Algebraic Cofactor Cell: Used for Inverting A Matrix----Square Matrix */
ComplexType CofactorMatrix(const Matrix* matrix, int m, int n)
{
	int row_i, column_j;
	ComplexType Cofactor_result;
	Cofactor_result._Val[0] = 0; Cofactor_result._Val[1] = 0;
	if (IsNullComplexMatrix(matrix))
	{
		puts("CofactorMatrix ERROE: An invalid matrix!\n");
		return Cofactor_result;
	}

	DoubleType prefixSign = 1;
	Matrix *matrix_Cofactor = InitComplexMatrix(matrix->row - 1, matrix->column - 1);  // Cofactir Matrix;
	

	// Tip: matrix->row = MatrixRow(matrix); matrix->row = MatrixRow(matrix)
	if (MatrixRow(matrix) != MatrixColumn(matrix) || MatrixRow(matrix) < 2)   // Square Matrix && Dimension>2
	{
		puts("ERROE: Not a square 2D matrix!\n");
		return Cofactor_result;
	}

	int i = 0;
	for (row_i = 0; row_i < matrix->row; row_i++)
	{
		if (row_i == m)
			continue;    // unwanted row
		for (column_j = 0; column_j < matrix->column; column_j++)
		{

			if (column_j == n)   // unwanted column
				continue;

			matrix_Cofactor->arrayComplex[i++] = matrix->arrayComplex[row_i * matrix->column + column_j];   // Pointer to the tail of momory

		}
	}
	/*
	// Attention: this will result in the incompatability of matrix size: Overflow of Stack
	for (row_i = 0; row_i < matrix->row; row_i++)
	{
		for (column_j = 0; column_j < matrix->column; column_j++)
		{
			if (row_i != m || column_j != n)   // Correction: ||---> &&
			matrix_Cofactor.arrayComplex[i++] = matrix->arrayComplex[row_i * matrix->column + column_j];   // Pointer to the tail of momory
		}
	}
	*/

	prefixSign = ((m + n) % 2 == 0) ? 1 : -1;
	Cofactor_result = _Cmulcr(Comput_D(matrix_Cofactor), prefixSign);
	// Free Memory
	DestroyComplexMatrix(matrix_Cofactor);
	return (Cofactor_result);
}



//复数矩阵求逆
/* Euqal Complex */
Bool IsEqual(const ComplexType Complex_A, const ComplexType Complex_B)
{
	// Approximate Equality
	if ((creal(Complex_A) == creal(Complex_B)) && (cimag(Complex_A) == cimag(Complex_B)))
		return True;
	else
	{
		return False;
	}
}

/* Division Complex : Complex_C = Complex_A / Complex_B */
ComplexType DivComplex(const ComplexType Complex_A, const ComplexType Complex_B)   // Tip: Return Value Is NOT A Pointer
{
	ComplexType Complex_C;
	Complex_C._Val[0] = (creal(Complex_A) * creal(Complex_B) + cimag(Complex_A)
		* cimag(Complex_B)) / (pow(creal(Complex_B), 2) + pow(cimag(Complex_B), 2));
	Complex_C._Val[1] = (cimag(Complex_A) * creal(Complex_B) - creal(Complex_A)
		* cimag(Complex_B)) / (pow(creal(Complex_B), 2) + pow(cimag(Complex_B), 2));
	/*
		Complex_C._Val[0] = (Complex_A._Val[0] * Complex_B._Val[0] + Complex_A._Val[1]
			* Complex_B._Val[1]) / (pow(Complex_B._Val[0], 2) + pow(Complex_B._Val[1], 2));
		Complex_C._Val[1] = (Complex_A._Val[1] * Complex_B._Val[0] - Complex_A._Val[0]
			* Complex_B._Val[1]) / (pow(Complex_B._Val[0], 2) + pow(Complex_B._Val[1], 2));
	*/
	return Complex_C;
	//  Div = (ac + bd） / （c ^ 2 + d ^ 2））＋（bc－ad） / （c ^ 2 + d ^ 2）i
}

/* Invert: matrix_Invert = 1/matrix_Initial */
Matrix* InvMat(const Matrix* matrix_Initial)
{
	Matrix* matrix_Invert = InitComplexMatrix(matrix_Initial->row, matrix_Initial->column);
	ComplexType Determ;
	Determ._Val[0] = creal(Comput_D(matrix_Initial));
	Determ._Val[1] = cimag(Comput_D(matrix_Initial));
	ComplexType ZeroDeterm;
	ZeroDeterm._Val[0] = 0; ZeroDeterm._Val[1] = 0;
	if (!IsEqual(Determ, ZeroDeterm))
	{
		for (int row_i = 0; row_i < matrix_Initial->row; row_i++)
		{
			for (int column_j = 0; column_j < matrix_Initial->column; column_j++)
			{
				// Tip: *(matrix_C->arrayComplex + j * matrix_A->row + i) = matrix_Invert->arrayComplex[j * matrix_Initial->row + i]
				*(matrix_Invert->arrayComplex + column_j * matrix_Initial->column + row_i) = \
					DivComplex(CofactorMatrix(matrix_Initial, row_i, column_j), Determ);
			}

		}
	}
	return matrix_Invert;
}



//求实数的协方差阵
/* Covariance Matrix Cell */
DoubleType covDoubleMatrixCell(const DoubleType x[], const DoubleType y[], const int size)  // size refers to row number of matrix
{
	DoubleType cov; // = NULL;
//	cov = (DoubleType *)malloc(sizeof(DoubleType ));
	int i;
	DoubleType averX = 0.0, averY = 0.0, sumX = 0.0, sumY = 0.0, sumXY = 0.0, subx = 0.0, suby = 0.0;

	// Sum Values of Row/Column
	for (i = 0; i < size; i++)
	{
		sumX += x[i];   // double + double
		sumY += y[i];   // double + double
	}
	// Average Values of Row/Column
	averX = sumX / size;
	averY = sumY / size;

	// Covariance Values
	for (i = 0; i < size; i++)
	{
		subx -= averX;
		suby -= averY;

		sumXY = sumXY + subx * suby;
	}
	cov = sumXY / (size - 1.0);
	return cov;
}

/* Covarianxe Matrix of Double Matrix */
void CovarianceDoubleMatrix(const Matrix2Double* matrix, const Matrix2Double* TransMat, Matrix2Double * CovMat)
{
	if (IsNullDoubleMatrix(matrix))
		return;
	else
		if (CovMat->row != DoubleMatrixColumn(matrix) || CovMat->column != DoubleMatrixColumn(matrix))
			return;
		else
		{
			for (int i = 0; i < CovMat->row; i++)
			{
				for (int j = 0; j < CovMat->column; ++j)
				{
					CovMat->arrayDouble[CovMat->column * i + j] = \
						covDoubleMatrixCell(TransMat->arrayDouble + i * TransMat->column, \
							TransMat->arrayDouble + j * TransMat->column, TransMat->column);
				}
			}
		}
}


//复数矩阵协方差阵
/* Covariance Matrix Cell: void */
ComplexType covMatrixCell(const ComplexType x[], const ComplexType y[], const int size)  // size refers to row number of matrix
{
	ComplexType cov; // = NULL;
//	cov = (ComplexType*)malloc(sizeof(ComplexType));
	int i;
	ComplexType averX, averY, sumX, sumY, sumXY, subx, suby;

	// Initiate Complex Variables
	InitComplex(&averX);
	InitComplex(&averY);
	InitComplex(&sumX);
	InitComplex(&sumY);
	InitComplex(&sumXY);
	InitComplex(&subx);
	InitComplex(&suby);

	// Sum Values of Row/Column
	for (i = 0; i < size; i++)
	{
		// Solution_1
		sumX = AddComplex(x[i], sumX);   // Complex + Complex
		sumY = AddComplex(y[i], sumY);   // Complex + Complex
		// Solution_2
//		sumX = AddComplex(x[i * size], sumX);   // Complex + Complex
//		sumY = AddComplex(y[i * size], sumY);   // Complex + Complex
		// Solution_3
//		sumX._Val[0] = creal(x[i]) + sumX._Val[0];
//		sumX._Val[1] = cimag(x[i]) + sumX._Val[1];
//		sumY._Val[0] = creal(y[i]) + sumY._Val[0];
//		sumY._Val[1] = cimag(y[i]) + sumY._Val[1];
	}
	// Average Values of Row/Column
	averX = _Cmulcr(sumX, pow(size * 1.0, -1.0));   // Tip: 1/size = pow(size, -1)
	averY = _Cmulcr(sumY, pow(size * 1.0, -1.0));

	// Covariance Values
	for (i = 0; i < size; i++)
	{
		// Solution_1
		subx = SubComplex(x[i], averX);
		suby = SubComplex(y[i], averY);
		// Solution_2
//		subx = SubComplex(x[i * size], averX);
//		suby = SubComplex(y[i * size], averY);
		// Solution_3
//		subx._Val[0] = creal(x[i]) - creal(averX);
//		subx._Val[1] = cimag(x[i]) - cimag(averX);
//		suby._Val[0] = creal(y[i]) - creal(averY);
//		suby._Val[1] = cimag(y[i]) - cimag(averY);
		sumXY = AddComplex(sumXY, _Cmulcc(subx, suby));
		//		sumXY._Val[0] = sumXY._Val[0] + _Cmulcc(subx, suby)._Val[0];
		//		sumXY._Val[1] = sumXY._Val[1] + _Cmulcc(subx, suby)._Val[1];
	}
	//  *cov = _Cmulcr(sumXY, pow(size - 1, -1));  // OR:
	cov = _Cmulcr(sumXY, pow(size - 1.0, -1.0));
	return cov;
}

/* Covarianxe Matrix of Complex Matrix */
void CovarianceComplexMatrix(const Matrix* matrix, const Matrix* TransMat, const Matrix *TransPoseMat, Matrix* CovMat)
{
	if (IsNullComplexMatrix(matrix))
		return;
	else
		if (CovMat->row != MatrixColumn(matrix) || CovMat->column != MatrixColumn(matrix))
			return;
		else
		{
			for (int i = 0; i < CovMat->row; i++)
			{
				for (int j = 0; j < CovMat->column; ++j)
				{
					CovMat->arrayComplex[CovMat->column * i + j] = \
						covMatrixCell(TransMat->arrayComplex + i * TransMat->column, \
							TransPoseMat->arrayComplex + j * TransPoseMat->column, TransMat->column);
				}
			}
		}
}



//矩阵的模
/* 2-norm of a Matrix */
ComplexType MatrixNorm2(const Matrix* matrix)
{
	//	OR
	//	int size = matrix->row * matrix->column;
	ComplexType norm;
	norm._Val[0] = 0; norm._Val[1] = 0;
	if (IsNullComplexMatrix(matrix))
	{
		puts("MatrixNorm2 ERROE: An invalid matrix!\n");
		return (csqrt(norm));
	}
	else
	{
		for (int i = 0; i < matrix->row; ++i)
		{
			for (int j = 0; j < matrix->column; ++j)
				norm = AddComplex(norm, _Cmulcc(matrix->arrayComplex[i * matrix->column + j], matrix->arrayComplex[i * matrix->column + j]));
		}

		// OR
//		for (row_i = 0; row_i < MatrixSize(matrix); ++row_i)
//		{
//			for (column_j = 0; column_j < MatrixSize(matrix); ++column_j)
//				norm = *(AddComplex(norm, _Cmulcc(matrix->arrayComplex[row_i * matrix->row + column_j], matrix->arrayComplex[row_i * matrix->row + column_j])));
//		}

		return (csqrt(norm));   // csqrt w.r.t a Complex is different frm sqrt w.r.t a Double
	}
}



//QR分解
/* QR Decompose */
void QR(const Matrix* A, Matrix* Q, Matrix* R)
{
	if (IsNullComplexMatrix(A))
	{
		puts("QR ERROE: An invalid matrix!\n");
		return;
	}
	// Not A Square Matrix
	if (A->row != A->column)
	{
		printf("ERROE: Not a square matrix!\n");
		return;
	}

	int i, j, k, m;
	int size;
	const int N = MatrixRow(A);
	ComplexType temp;

	// Column Vector Saving Column Vectors of A
	Matrix *a = InitComplexMatrix(N, 1);
	Matrix *b = InitComplexMatrix(N, 1);
	size = MatrixSize(A);
	if (MatrixSize(Q) != size)
	{
		// free(Q->arrayComplex);
		DestroyComplexMatrix(Q);                   // Free The Initial Matrix
		Q = InitComplexMatrix(A->row, A->column);   // Reset Size and Initialize
	}

	if (MatrixSize(R) != size)
	{
		// free(R->arrayComplex);
		DestroyComplexMatrix(R);
		R = InitComplexMatrix(A->row, A->column);
	}

	for (j = 0; j < N; ++j)
	{
		for (i = 0; i < N; ++i)
		{
			a->arrayComplex[i] = b->arrayComplex[i] = A->arrayComplex[i * A->column + j];   // Cols Vector of A
		}

		for (k = 0; k < j; ++k)
		{
			R->arrayComplex[k * R->column + j]._Val[0] = 0;
			R->arrayComplex[k * R->column + j]._Val[1] = 0;
			//			InitComplex(R->arrayComplex + k * R->column + j);  // reset
			for (m = 0; m < N; ++m)
			{
				R->arrayComplex[k * R->column + j] = AddComplex(R->arrayComplex[k * R->column + j], \
					_Cmulcc(a->arrayComplex[m], Q->arrayComplex[m * Q->column + k]));
			}

			for (m = 0; m < N; ++m)
			{
				b->arrayComplex[m] = SubComplex(b->arrayComplex[m], _Cmulcc(R->arrayComplex[k * R->column + j], \
					Q->arrayComplex[m * Q->column + k]));
			}
		}

		temp = MatrixNorm2(b);
		R->arrayComplex[j * R->column + j] = temp;

		for (i = 0; i < N; ++i)
		{
			Q->arrayComplex[i * Q->column + j] = DivComplex(b->arrayComplex[i], temp);
		}
	}
	// Free Local Memory
	DestroyComplexMatrix(a);
	DestroyComplexMatrix(b);
}

//复数矩阵复制
/* Copy: Complex matrixA = Complex matrixB */
void CopyMatrix(const Matrix* matrix_A, Matrix* matrix_B)
{
	if (IsNullComplexMatrix(matrix_A))
	{
		puts("CopyMatrix ERROE: An invalid matrix!\n");
		return;
	}
	//matrix_B->row = MatrixRow(matrix_A);
	//matrix_B->column = MatrixColumn(matrix_A);

	else
	{
		int index = 0;
		for (int row_i = 0; row_i < matrix_A->row; row_i++)
		{
			for (int column_j = 0; column_j < matrix_A->column; column_j++)
			{
				index = matrix_B->column * row_i + column_j;
				matrix_B->arrayComplex[index] = matrix_A->arrayComplex[index];
				
			}
		}
		// OR:
		// memcpy(matrix_B->arrayComplex, matrix_A->arrayComplex, MatrixSize(matrix_A) * sizeof(ComplexType));
	}
}


//特征值
/* eigen values */
void EigenValue(const Matrix* matrix, Matrix* eigenvalue)
{
	const int NUM = 100;   // Iteration Times

	// Local Matrice
	Matrix *Q, *R;
	Matrix *temp;
	// Initiate
	Q = InitComplexMatrix(matrix->row, matrix->column);
	R = InitComplexMatrix(matrix->row, matrix->column);
	temp = InitComplexMatrix(matrix->row, matrix->column);

	if (IsNullComplexMatrix(matrix))
	{
		puts("EigenValue ERROE: An invalid matrix!\n");
		return;
	}
	// Copy matrix to temp in order not to change matrix
	CopyMatrix(matrix, temp);
	// QR Decompose and 
	for (int k = 0; k < NUM; ++k)
	{
		QR(temp, Q, R);
		DestroyComplexMatrix(temp);
		temp = MatrixMulMatrix(R, Q);
	}
	// Abstract Eigen Values from the Diagonal Elements of temp =  Q * R
	for (int k = 0; k < temp->row; ++k)
	{
		eigenvalue->arrayComplex[k] = temp->arrayComplex[k * temp->column + k];
	}
	// Free Local Memory
	Matrix ComplexMatrixArray[] = { *R, *Q, *temp };
	int numComplexMatrixArray = sizeof(ComplexMatrixArray) / sizeof(Matrix);
	DestroyComplexMatrixArray(ComplexMatrixArray, numComplexMatrixArray);      // Complpex Matrix
	
	//DestroyComplexMatrix(Q);
	//DestroyComplexMatrix(R);
	//DestroyComplexMatrix(temp);
//	return eigenvalue;
}



//特征向量
/* Negative Complex : Complex_B = -Complex_A = -creal(Complex_A) - cimag(Complpex_A) */
ComplexType NegativeComplex(const ComplexType Complex_A)
{
	ComplexType Complex_B;
	Complex_B = _Cmulcr(Complex_A, -1.0);
	// OR
/*
	Complex_B._Val[0] = -creal(Complex_A);
	Complex_B._Val[1] = cimag(Complex_A) * (-1.0);
*/
	return Complex_B;
}

/* eigen vectors */
void EigenVector(const Matrix* matrix, const Matrix* eigenvalue, Matrix* eigenvector)
{
	if (IsNullComplexMatrix(matrix) || IsNullComplexMatrix(eigenvalue))
	{
		puts("EigenVector ERROE: An invalid matrix!\n");
		return;
	}
	int i, j, q;
	int m;

	// Access to Eigen Values
	int count;
	int num = MatrixRow(matrix);   // = matrix->row: Numbers of Eigen Values or Cols
	ComplexType evalue;

	// Access to temp
	ComplexType sum, midsum, mid;
	Matrix *temp;   // temp = A - λI: (A - λI) * x = 0
	temp = InitComplexMatrix(matrix->row, matrix->column);

	for (count = 0; count < num; ++count)
	{
		// Calculate x: Ax = λ * x
		evalue = eigenvalue->arrayComplex[count];
		CopyMatrix(matrix, temp);
		for (i = 0; i < temp->column; ++i)
		{
			temp->arrayComplex[i * temp->column + i] = SubComplex(temp->arrayComplex[i * temp->column + i], evalue);
			//			temp->arrayComplex[i * temp->column + i]._Val[0] -= creal(evalue);
			//			temp->arrayComplex[i * temp->column + i]._Val[1] -= cimag(evalue);
		}

		// Transform temp to Ladder Matrix
		for (i = 0; i < temp->row - 1; ++i)
		{
			mid._Val[0] = creal(temp->arrayComplex[i * temp->column + i]);   // Diagonal Element
			mid._Val[1] = cimag(temp->arrayComplex[i * temp->column + i]);
			for (j = i; j < temp->column; ++j)
			{
				temp->arrayComplex[i * temp->column + j] = DivComplex(temp->arrayComplex[i * temp->column + j], mid);
			}

			for (j = i + 1; j < temp->row; ++j)
			{
				mid = temp->arrayComplex[j * temp->column + i];
				for (q = i; q < temp->column; ++q)
				{
					temp->arrayComplex[j * temp->column + q] = SubComplex(temp->arrayComplex[j * temp->column + q], \
						_Cmulcc(temp->arrayComplex[i * temp->column + q], mid));

				}
			}
		}
		midsum._Val[0] = 1;
		midsum._Val[1] = 0;
		eigenvector->arrayComplex[(eigenvector->row - 1) * eigenvector->column + count]._Val[0] = 1;
		eigenvector->arrayComplex[(eigenvector->row - 1) * eigenvector->column + count]._Val[1] = 0;
		for (m = temp->row - 2; m >= 0; --m)
		{
			//			InitComplex(&sum);
			sum._Val[0] = 0; sum._Val[1] = 0;   // Zero Complex
			for (j = m + 1; j < temp->column; ++j)
			{
				sum = AddComplex(sum, _Cmulcc(temp->arrayComplex[m * temp->column + j],
					eigenvector->arrayComplex[j * eigenvector->column + count]));
			}
			sum = DivComplex(NegativeComplex(sum), temp->arrayComplex[m * temp->column + m]);  // Warning: Parameters' Type
			//sum = -sum / *(temp.arrayComplex[m * temp.column + m]);
			midsum = AddComplex(midsum, _Cmulcc(sum, sum));
			eigenvector->arrayComplex[m * eigenvector->column + count] = sum;
		}

		midsum = csqrt(midsum);
		for (i = 0; i < eigenvector->row; ++i)  // One Column Vector--Eigen Vector
		{
			eigenvector->arrayComplex[i * eigenvector->column + count] =
				DivComplex(eigenvector->arrayComplex[i * eigenvector->column + count], midsum);
		}
	}
	DestroyComplexMatrix(temp);
	//	return eigenvector;
}


void qsort_double(ComplexType* array, int left, int right)
{
	if (left > right)  return;
	double midnum = array[left]._Val[0];
	int begin_idx = left;
	int end_idx = right;
	double tmp = 0;
	while (left < right) {
		while (array[right]._Val[0] >= midnum && left < right) {
			right--;
		}
		while (array[left]._Val[0] <= midnum && left < right) {
			left++;
		}
		if (left < right) {
			tmp = array[left]._Val[0];
			array[left]._Val[0] = array[right]._Val[0];
			array[right]._Val[0] = tmp;
		}
	}
	//此时left==right
	tmp = array[left]._Val[0];
	array[left]._Val[0] = array[begin_idx]._Val[0];
	array[begin_idx]._Val[0] = tmp;
	qsort_double(array, begin_idx, left - 1);
	qsort_double(array, left + 1, end_idx);
	return;
}


//将特征向量按照升序排列，快排
void eig_sort(Matrix* EigenValue)
{
	int size = EigenValue->column * EigenValue->row;
	//double *num = (double*)malloc(size * sizeof(double));
	//int i = 0;
	int left = 0, right = size - 1;
	qsort_double(EigenValue->arrayComplex, left, right);
	return;
}


void qresort_double(ComplexType* array, int left, int right)
{
	if (left > right)  return;
	double midnum = array[left]._Val[0];
	int begin_idx = left;
	int end_idx = right;
	double tmp = 0;
	while (left < right) {
		while (array[right]._Val[0] <= midnum && left < right) {
			right--;
		}
		while (array[left]._Val[0] >= midnum && left < right) {
			left++;
		}
		if (left < right) {
			tmp = array[left]._Val[0];
			array[left]._Val[0] = array[right]._Val[0];
			array[right]._Val[0] = tmp;
		}
	}
	//此时left==right
	tmp = array[left]._Val[0];
	array[left]._Val[0] = array[begin_idx]._Val[0];
	array[begin_idx]._Val[0] = tmp;
	qresort_double(array, begin_idx, left - 1);
	qresort_double(array, left + 1, end_idx);
	return;
}

//将特征向量按照降序排列，快排
void eig_resort(Matrix* EigenValue)
{
	int size = EigenValue->column * EigenValue->row;
	//double *num = (double*)malloc(size * sizeof(double));
	//int i = 0;
	int left = 0, right = size - 1;
	qresort_double(EigenValue->arrayComplex, left, right);
	return;
}




//复矩阵切片函数, begin:从第begin行/列开始切片（含第begin行）；end: 到第begin行/列结束（含第end行）；op=0：按行切片，op=1：按列切片
Matrix *MatrixSlice(Matrix *matrix, int begin, int end, int op)
{
	if (begin < 1) {
		printf("MatrixSlice 'begin' para error!\n");
	}
	else if (begin > end) {
		printf("MatrixSlice 'begin' larger than 'end'!\n");
	}
	
	if (op == 1) {
		if (end > matrix->column) {
			printf("MatrixSlice 'end' para error!\n");
		}
		/*Matrix target;
		target.arrayComplex = (ComplexType*)malloc((begin - end + 1)*matrix->row * sizeof(ComplexType));
		target.row = matrix->row;
		target.column = begin - end + 1;*/
		Matrix *target = InitComplexMatrix(matrix->row, begin - end + 1);
		
		Matrix *res = target;
		//Matrix res;
		//DestroyComplexMatrix(&target);
		
		int i = 0, idx = 0;
		while (i < matrix->row) {
			int j = begin - 1;
			for (j = begin - 1; j < end; j++) {
				res->arrayComplex[idx++] = matrix->arrayComplex[i * matrix->column + j];
			}
			i++;
		}
		return res;
	}
	if (op == 0) {
		if (end > matrix->row) {
			printf("MatrixSlice 'end' para error!\n");
		}
		/*Matrix* target;
		target->arrayComplex = (ComplexType*)malloc((begin - end + 1)*matrix->column * sizeof(ComplexType));
		target->row = begin - end + 1;
		target->column = matrix->column;
		Matrix res = *target;
		free(target);*/
		/*Matrix target;
		target.arrayComplex = (ComplexType*)malloc((begin - end + 1)*matrix->column * sizeof(ComplexType));
		target.row = begin - end + 1;
		target.column = matrix->column;*/
		Matrix *target = InitComplexMatrix(begin-end+1, matrix->column);
		Matrix *res = target;
		//DestroyComplexMatrix(&target);
		int idx = 0;
		int j = 0;
		int i = begin - 1;
		for (i = begin - 1; i < end; i++) {
			while (j < matrix->column){
				res->arrayComplex[idx++] = matrix->arrayComplex[i * matrix->column + j];
				j++;
			}
		}
		return res;
	}
	printf("----MatrixSlice error!----\n");
	return NULL;
}



















//打印复矩阵
void PrintMatrix(const Matrix* matrix)
{
	int row = matrix->row;
	int column = matrix->column;
	printf(">>Matrix_%x:\n", matrix);
	int i, j;
	for (i = 0; i < row; i++) {
		for (j = 0; j < column; j++) {
			PrintComplexType(&matrix->arrayComplex[i*column + j]);
			printf("  ");
		}
		printf("\n");
	}
	return;

}

//打印实矩阵
void PrintMatrix2Double(const Matrix2Double* matrix)
{
	int row = matrix->row;
	int column = matrix->column;
	printf(">>Matrix_%x:\n", matrix);
	int i, j;
	for (i = 0; i < row; i++) {
		for (j = 0; j < column; j++) {
			printf("%.4lf  ", matrix->arrayDouble[i*column + j]);
		}
		printf("\n");
	}

}



#endif
