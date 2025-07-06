import math
import matplotlib.pyplot as plt
import pandas as pd
import os

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)


def theoretical_bfc_size(rhat: float) -> float:
    """
    Calculate theoretical BFC size from rhat.

    Args:
        rhat (float): Input rhat value

    Returns:
        float: Theoretical BFC size in KB
    """
    return 5.64 * rhat / 8 / 1000


def expected_levels_r(rhat: float) -> float:
    """
    Calculate expected levels (r) based on rhat.

    Args:
        rhat (float): Input rhat value

    Returns:
        float: Expected levels (r)
    """
    p = 0.5
    return 1 + (math.log(rhat) + 0.577) / math.log(1 / p)


def expected_levels_s(rhat: float) -> float:
    """
    Calculate expected levels (s) based on rhat.

    Args:
        rhat (float): Input rhat value

    Returns:
        float: Expected levels (s)
    """
    p = 0.5
    p0 = math.sqrt(p) / 2
    return (
        1
        + ((math.log(2 * rhat) + 0.577) / math.log(1 / p))
        - (1 - math.pow(1 - p0, 2 * rhat))
        * (math.floor(math.log(p0) / math.log(p)) - 1)
    )


def expected_levels(rhat: float) -> float:
    """
    Calculate total expected levels based on rhat.

    Args:
        rhat (float): Input rhat value

    Returns:
        float: Total expected levels
    """
    return min(2 * expected_levels_s(rhat) - 1, 2 * expected_levels_r(rhat))


def generate_data():
    """
    Generate data for plotting by varying volume and lifetime parameters.

    Returns:
        pandas.DataFrame: DataFrame containing Volume, Lifetime, Bitsize, and Levels
    """
    data = []
    for blobs in range(1, 6):
        rhat = blobs * 170000
        data.append([rhat, theoretical_bfc_size(rhat), expected_levels(rhat)])

    return pd.DataFrame(data, columns=["Capacity", "Bitsize", "Levels"])


def plot_issuance_size(df: pd.DataFrame):
    """
    Create and save the Yearly Issuance Volume vs Expected Size plot.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the data
    """
    fig = plt.figure(figsize=(6, 4))
    plt.title("Capacity vs Expected Size")

    y_data = df
    plt.plot(y_data["Capacity"], y_data["Bitsize"])

    plt.xlabel("Capacity")
    plt.ylabel("Expected Size in KB")
    # plt.yscale("log")
    plt.grid(True)
    plt.savefig("images/issuance_size.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_issuance_levels(df: pd.DataFrame):
    """
    Create and save the Yearly Issuance Volume vs Expected Levels plot.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the data
    """
    fig = plt.figure(figsize=(6, 4))
    plt.title("Capacity vs Expected Levels")

    y_data = df
    plt.plot(y_data["Capacity"], y_data["Levels"])

    plt.xlabel("Capacity")
    plt.ylabel("Expected Levels")
    plt.grid(True)
    plt.savefig("images/issuance_levels.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main function to generate data and create plots."""
    # Generate data
    df = generate_data()

    # Create plots
    plot_issuance_size(df)
    plot_issuance_levels(df)


if __name__ == "__main__":
    main()
