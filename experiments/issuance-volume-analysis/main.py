import math
import matplotlib.pyplot as plt
import pandas as pd
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

def rhat_from_vy(v: float, y: float) -> float:
    """
    Calculate rhat based on volume and lifetime.
    
    Args:
        v (float): Yearly issuance volume
        y (float): Lifetime in years
    
    Returns:
        float: Calculated rhat value
    """
    x = 0.05
    delta = 1.1
    t = 1
    fraction = (math.pow(delta, t+1)-1) * math.pow(delta, y-t)
    fraction = fraction / (delta-1)
    return v * (1-x) * fraction

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
    return 1 + (math.log(rhat) + 0.577) / math.log(1/p)

def expected_levels_s(rhat: float) -> float:
    """
    Calculate expected levels (s) based on rhat.
    
    Args:
        rhat (float): Input rhat value
    
    Returns:
        float: Expected levels (s)
    """
    p = 0.5
    p0 = math.sqrt(p)/2
    return 1 + ((math.log(2*rhat) + 0.577) / math.log(1/p)) - (1-math.pow(1-p0, 2*rhat)) * (math.floor(math.log(p0)/math.log(p))-1)

def expected_levels(rhat: float) -> float:
    """
    Calculate total expected levels based on rhat.
    
    Args:
        rhat (float): Input rhat value
    
    Returns:
        float: Total expected levels
    """
    return 2 * expected_levels_s(rhat) - 1.2 * expected_levels_r(rhat)

def generate_data():
    """
    Generate data for plotting by varying volume and lifetime parameters.
    
    Returns:
        pandas.DataFrame: DataFrame containing Volume, Lifetime, Bitsize, and Levels
    """
    data = []
    for vexp in range(3, 9):
        for y in [18, 14, 10, 6, 2]:
            v = math.pow(10, vexp)
            rhat = rhat_from_vy(v, y)
            data.append([v, y, theoretical_bfc_size(rhat), expected_levels(rhat)])
    
    return pd.DataFrame(data, columns=['Volume', 'Lifetime', 'Bitsize', 'Levels'])

def plot_issuance_size(df: pd.DataFrame):
    """
    Create and save the Yearly Issuance Volume vs Expected Size plot.
    
    Args:
        df (pandas.DataFrame): Input DataFrame containing the data
    """
    fig = plt.figure(figsize=(6, 4))
    plt.title('Yearly Issuance Volume vs Expected Size')
    
    for y in df['Lifetime'].unique():
        y_data = df[df['Lifetime'] == y]
        plt.plot(y_data['Volume'], y_data['Bitsize'], label=f'y={y}')
    
    plt.xlabel('Yearly Issuance Volume')
    plt.ylabel('Expected Size in KB')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/issuance_size.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_issuance_levels(df: pd.DataFrame):
    """
    Create and save the Yearly Issuance Volume vs Expected Levels plot.
    
    Args:
        df (pandas.DataFrame): Input DataFrame containing the data
    """
    fig = plt.figure(figsize=(6, 4))
    plt.title('Yearly Issuance Volume vs Expected Levels')
    
    for y in df['Lifetime'].unique():
        y_data = df[df['Lifetime'] == y]
        plt.plot(y_data['Volume'], y_data['Levels'], label=f'y={y}')
    
    plt.xlabel('Yearly Issuance Volume')
    plt.ylabel('Expected Levels')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/issuance_levels.png', dpi=300, bbox_inches='tight')
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