"Homework 3 attempt"
import math


def t_distribution_pdf(x, m):
    """
    Probability density function of the t-distribution.

    :param x: The value at which to evaluate the PDF.
    :param m: Degrees of freedom.
    :return: The value of the PDF at x.
    """
    if m <= 0:
        raise ValueError("Degrees of freedom must be positive.")

    return (math.gamma((m + 1) / 2) /
            (math.sqrt(m * math.pi) * math.gamma(m / 2))) * (1 + (x ** 2) / m) ** (-(m + 1) / 2)


def compute_K_m(m):
    """
    Compute the normalization constant K_m for the t-distribution.

    Args:
        m (int): Degrees of freedom.

    Returns:
        float: The value of K_m.
    """
    if m <= 0:
        raise ValueError("Degrees of freedom must be positive.")

    return math.gamma((m + 1) / 2) / (math.sqrt(m * math.pi) * math.gamma(m / 2))


def trapezoidal_integration(func, a, b, *args, n=1000):
    """
    Perform numerical integration using the trapezoidal rule.

    :param func: The function to integrate.
    :param a: The lower limit of integration.
    :param b: The upper limit of integration.
    :param args: Additional arguments to pass to the function.
    :param n: The number of trapezoids to use (default is 1000).
    :return: The approximate integral of the function from a to b.
    """
    h = (b - a) / n
    integral = 0.5 * (func(a, *args) + func(b, *args))

    for i in range(1, n):
        integral += func(a + i * h, *args)

    return integral * h


def t_distribution_cdf(z, m):
    """
    Compute the cumulative distribution function (CDF) of the t-distribution.

    :param z: The value at which to evaluate the CDF.
    :param m: Degrees of freedom.
    :return: The CDF value at z.
    """
    if m <= 0:
        raise ValueError("Degrees of freedom must be positive.")

    lower_limit = -10  # A more reasonable limit instead of -1000
    return trapezoidal_integration(t_distribution_pdf, lower_limit, z, m)


def user_input_t_distribution():
    """
    Allow user to input degrees of freedom and z values for the t-distribution CDF.
    """
    print("This program computes the CDF of the t-distribution for given degrees of freedom and z values.")

    while True:
        try:
            # Get degrees of freedom from user
            m = int(input("\nEnter degrees of freedom (positive integer): ").strip())
            if m <= 0:
                print("Degrees of freedom must be a positive integer. Try again.")
                continue

            # Get multiple z values as comma-separated input
            z_values_input = input("Enter z values separated by commas (e.g., -1.5, 0, 1.5): ").strip()
            z_values = [float(z.strip()) for z in z_values_input.split(",")]

            print(f"\nDegrees of freedom (m): {m}")
            for z in z_values:
                cdf_value = t_distribution_cdf(z, m)
                print(f"F(z={z:0.3f} | m={m}) = {cdf_value:0.6f}")

            break  # Exit loop after successful computation

        except ValueError:
            print("Invalid input. Please enter numeric values correctly.")


if __name__ == "__main__":
    user_input_t_distribution()
