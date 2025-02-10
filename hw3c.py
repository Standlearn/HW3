import numpy as np


def is_symmetric(matrix):
    """Check if a matrix is symmetric."""
    return np.allclose(matrix, matrix.T)


def is_positive_definite(matrix):
    """Check if a matrix is positive definite."""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def cholesky_solve(A, b):
    """Solve Ax = b using Cholesky decomposition."""
    L = np.linalg.cholesky(A)

    # Solve L * y = b
    y = np.linalg.solve(L, b)

    # Solve L.T * x = y
    x = np.linalg.solve(L.T, y)
    return x


def doolittle_solve(A, b):
    """Solve Ax = b using Doolittle's LU decomposition."""
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Upper triangular matrix U
        for k in range(i, n):
            sum1 = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum1

        # Lower triangular matrix L
        for k in range(i, n):
            if i == k:
                L[i][i] = 1  # Diagonal as 1
            else:
                sum2 = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - sum2) / U[i][i]

    # Solve L * y = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i)))

    # Solve U * x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return x


def get_matrices_and_vectors():
    """Prompt user to input two matrices and vectors."""
    matrices = []
    vectors = []
    for i in range(2):
        print(f"Enter matrix A{i + 1}:")
        n = int(input("Enter the size of the matrix: "))
        print("Enter the matrix row by row, with space-separated values: example: 1 1 1 0 for row 1")
        A = np.array([list(map(float, input().split())) for _ in range(n)])
        print(f"Enter vector b{i + 1} (space-separated values):")
        b = np.array(list(map(float, input().split())))
        matrices.append(A)
        vectors.append(b)
    return matrices, vectors


def solve_system(A, b, problem_number):
    print(f"Solving problem {problem_number}:")
    if is_symmetric(A) and is_positive_definite(A):
        print("Matrix is symmetric and positive definite. Using Cholesky method.")
        x = cholesky_solve(A, b)
    else:
        print("Matrix is not symmetric positive definite. Using Doolittle method.")
        x = doolittle_solve(A, b)

    print("Solution x:", x)


def main():
    """Main function to check matrix properties and solve Ax = b for two problems."""
    matrices, vectors = get_matrices_and_vectors()
    for i in range(2):
        solve_system(matrices[i], vectors[i], i + 1)
        print("-")


if __name__ == "__main__":
    main()
