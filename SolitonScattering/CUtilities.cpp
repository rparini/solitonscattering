#include <iostream>
#include <complex>

typedef std::complex<double> cx;

// indexing: A[m,i,j] = A[m * uSize * uSize + i * uSize + j]
int index3D(int uSize, int m, int i, int j) {
	return (m * uSize + i ) * uSize + j;
}

int index2D(int uSize, int m, int i) {
	return m * uSize + i;
}

void RungeKutta(int M, int ySize, double h, cx * y, cx * A, cx * B) {
    // Solve y'(t) = A(t).y(t) + B(t)
    // These input arrays will be assumed to be flattened to 1D so that
    // A = [[a00 a01    = [a00, a01, a10, a11]
    //       a10 a11]]
    // h is the distance in t between consecutive A and B
    // Since Runge Kutta uses a midpoint step the 'step size' is 2h
    // y is the initial value and the final value will be stored here
    // ySize is the size of the field vector y (also size of y0)

    // M is the number of steps to be taken by Runge Kutta
    // h is the size of these steps

    cx k1[ySize];
    cx k2[ySize];
    cx k3[ySize];
    cx k4[ySize];

    cx kprod[ySize];

    int indexi, indexj, indexij, nextIndexij;

    for (int m = 0; m < M; ++m) {
        // k1_i = h * (A[2m]_ij * y_j + B[2m]_i)
        for (int i = 0; i < ySize; ++i) {
            kprod[i] = 0;
            for (int j = 0; j < ySize; ++j) {
                // Calculate A[m]_ij * y_j
                indexij = index3D(ySize,2*m,i,j);
                kprod[i] += A[indexij] * y[j];
            }            
            indexi = index2D(ySize,2*m,i);
            k1[i] = h*(kprod[i] + B[indexi]);
        }

        // k2_i = h * (A[2m+1]_ij * (y_j + k1_j/2) + B[2m+1]_i)
        for (int i = 0; i < ySize; ++i) {
            kprod[i] = 0;
            for (int j = 0; j < ySize; ++j) {
                // Calculate A[2m+1]_ij * (y_j + k1_j/2)
                indexij = index3D(ySize,2*m+1,i,j);
                kprod[i] += A[indexij] * (y[j] + k1[j]/2.);
            }
            indexi = index2D(ySize,2*m+1,i);
            k2[i] = h*(kprod[i] + B[indexi]);
        }

        // k3_i = h * (AInterp[2m+1]_ij * (y_j + k2_j/2) + B[2m+1]_i)
        for (int i = 0; i < ySize; ++i) {
            kprod[i] = 0;
            for (int j = 0; j < ySize; ++j) {
                // Calculate AInterp[m]_ij * (y_j + k2_j/2)
                indexij = index3D(ySize,2*m+1,i,j);
                kprod[i] += A[indexij] * (y[j] + k2[j]*0.5);
            }
            indexi = index2D(ySize,2*m+1,i);
            k3[i] = h*(kprod[i] + B[indexi]);
        }

        // k4_i = h * (A[2m+2]_ij * (y_j + k3_j) + B[2m+2]_i)
        for (int i = 0; i < ySize; ++i) {
            kprod[i] = 0;
            for (int j = 0; j < ySize; ++j) {
                // Calculate A[m+1]_ij * (y_j + k3_j)
                indexij = index3D(ySize,2*m+2,i,j);
                kprod[i] += A[indexij] * (y[j] + k3[j]);
            }
            indexi = index2D(ySize,2*m+2,i);
            k4[i] = h*(kprod[i] + B[indexi]);
        }

        // y[m+1] = y[m] + (k1 + 2*k2 + 2*k3 + k4)/6
        for (int i = 0; i < ySize; ++i)
            y[i] += (k1[i] + 2.*k2[i] + 2.*k3[i] + k4[i])/6.;

	    // printf("--------\n");
	    // for(int i = 0; i < ySize; i++) {
	    //     printf("%f + %f i\n", real(y[i]), imag(y[i]));
	    // }
    }
    // printf("--------\n");
}

int main() {
    // --- Set up the data to run the Runge Kutta function ---
    int M = 6;      // The number of steps in the domain t

    // Give the initial value of the field u(t = t0) = u0
    cx u[2] = {cx(0., 0.), cx(0., 0.)};

    // Define the size of the arrays during runtime with 'new' command
    // A = new cx [M * 2 * 2];
    // B = new cx [M * 2];
    cx A[M * 2 * 2];
    cx B[M * 2];

    // Create the multidimensional array as a dynamic linear array
    // indexing: A[m,i,j] = A[m * 2 * 2 + i * 2 + j]
    // m runs from 0 to M and indexes the time A[m,i,j] = A[i,j](t = m * h)
    for (int m = 0; m < M; ++m) {
        A[index3D(2, m, 0, 0)] = cx(-4, 0);
        A[index3D(2, m, 0, 1)] = cx(3, 0);
        A[index3D(2, m, 1, 0)] = cx(-2.4, 0);
        A[index3D(2, m, 1, 1)] = cx(1.6, 0);

        B[index2D(2, m, 0)] = cx(6, 0);
        B[index2D(2, m, 1)] = cx(3.6, 0);
    }

    double h = 0.1;  // The time step size

    RungeKutta(2, 2, h, &u[0], &A[0,0,0], &B[0,0]);

    printf("%f + %f i\n", real(u[0]), imag(u[0]));
    printf("%f + %f i\n", real(u[1]), imag(u[1]));

    return 0;
}



