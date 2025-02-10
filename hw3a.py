"""Homework 3 attempt"""

from numericalMethods import GPDF, Probability


def secant_method_v2(func, target, x0, x1, tol=1e-6, max_iter=100):
    """
    Secant method to find the root of the equation func(x) - target = 0.

    This version introduces small variations while achieving the same result.

    Args:
        func (function): The function for which the root is to be found.
        target (float): The target value (desired probability).
        x0 (float): First initial guess for the root.
        x1 (float): Second initial guess for the root.
        tol (float): Tolerance for convergence (default: 1e-6).
        max_iter (int): Maximum number of iterations (default: 100).

    Returns:
        float: The value of x that satisfies func(x) = target within tolerance.
    """
    iteration = 0  # Track iteration count
    prev_x0, prev_x1 = x0, x1  # Store previous values

    while iteration < max_iter:
        # Compute function values at x0 and x1
        fx0 = func(prev_x0) - target
        fx1 = func(prev_x1) - target

        # Check for convergence
        if abs(fx1) < tol:
            return prev_x1

        # Avoid division by zero with a small perturbation
        denominator = (fx1 - fx0) if (fx1 - fx0) != 0 else 1e-10

        # Compute new approximation
        x_new = prev_x1 - fx1 * (prev_x1 - prev_x0) / denominator

        # Update previous values
        prev_x0, prev_x1 = prev_x1, x_new
        iteration += 1

    return prev_x1  # Return best approximation after max_iter iterations


def main():
    """
    Main function to interactively compute probabilities or find the value of c
    for a given probability using the Gaussian Normal Distribution.
    """
    Again = True
    mean = 0
    stDev = 1.0
    c = 0.5
    OneSided = True
    GT = False
    yesOptions = ["y", "yes", "true"]

    while Again:
        response = input("Do you want to specify c and seek P (1) or specify P and seek c (2)? ").strip().lower()

        if response not in ["1", "2"]:
            print("Invalid choice. Please enter 1 or 2.")
            continue

        if response == "1":
            mean = float(input(f"Population mean? ({mean:0.3f})").strip() or mean)
            stDev = float(input(f"Standard deviation? ({stDev:0.3f})").strip() or stDev)
            c = float(input(f"c value? ({c:0.3f})").strip() or c)
            GT = input(f"Probability greater than c? ({GT})").strip().lower() in yesOptions
            OneSided = input(f"One sided? ({OneSided})").strip().lower() in yesOptions

            if OneSided:
                prob = Probability(GPDF, (mean, stDev), c, GT=GT)
                print(f"P(x" + (">" if GT else "<") + f"{c:0.2f}|{mean:0.2f}, {stDev:0.2f}) = {prob:0.2f}")
            else:
                prob = Probability(GPDF, (mean, stDev), c, GT=True)
                prob = 1 - 2 * prob
                print(f"P({mean - (c - mean)}<x<{mean + (c - mean)}|{mean:0.2f},{stDev:0.2f}) = {prob:0.3f}")

        else:
            mean = float(input(f"Population mean? ({mean:0.3f})").strip() or mean)
            stDev = float(input(f"Standard deviation? ({stDev:0.3f})").strip() or stDev)
            target_prob = float(input("Enter the desired probability: ").strip())
            GT = input(f"Probability greater than c? ({GT})").strip().lower() in yesOptions
            OneSided = input(f"One sided? ({OneSided})").strip().lower() in yesOptions

            def prob_func(c):
                if OneSided:
                    return Probability(GPDF, (mean, stDev), c, GT=GT)
                else:
                    prob = Probability(GPDF, (mean, stDev), c, GT=True)
                    return 1 - 2 * prob

            c0 = mean - 2 * stDev
            c1 = mean + 2 * stDev
            c = secant_method_v2(prob_func, target_prob, c0, c1)

            print(f"c value for P(x" + (
                ">" if GT else "<") + f"c|{mean:0.2f},{stDev:0.2f}) = {target_prob:0.3f} is {c:0.3f}")

        Again = input("Go again? (Y/N)").strip().lower() in yesOptions


if __name__ == "__main__":
    main()
