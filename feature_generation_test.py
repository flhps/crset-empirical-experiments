import pandas as pd
import numpy as np
import sys
import zlib
from scipy import stats
from scipy.fft import fft
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import math
from itertools import combinations
from scipy import stats
from tabulate import tabulate

def calculate_pattern_entropy_features(bitstring_data: str) -> Dict[str, float]:
    """
    Calculate pattern entropy features from bitstring data.
    
    Args:
        bitstring_data: Comma-separated bitstrings
    Returns:
        Dictionary of pattern entropy features
    """
    features = {}
    filters = bitstring_data.split(',')
    
    for idx, filter_str in enumerate(filters):
        # Skip empty filters
        if not filter_str:
            continue
            
        # Bigram entropy
        bigrams = [filter_str[i:i+2] for i in range(len(filter_str)-1)]
        bigram_counts = Counter(bigrams)
        bigram_probs = [count/len(bigrams) for count in bigram_counts.values()]
        features[f'bigram_entropy_filter_{idx}'] = -sum(p * math.log2(p) for p in bigram_probs) if bigram_probs else 0
        
        # Trigram entropy
        trigrams = [filter_str[i:i+3] for i in range(len(filter_str)-2)]
        trigram_counts = Counter(trigrams)
        trigram_probs = [count/len(trigrams) for count in trigram_counts.values()]
        features[f'trigram_entropy_filter_{idx}'] = -sum(p * math.log2(p) for p in trigram_probs) if trigram_probs else 0
        
        # Byte-level entropy
        bytes_patterns = [filter_str[i:i+8] for i in range(0, len(filter_str)-7, 8)]
        byte_counts = Counter(bytes_patterns)
        byte_probs = [count/len(bytes_patterns) for count in byte_counts.values()]
        features[f'byte_entropy_filter_{idx}'] = -sum(p * math.log2(p) for p in byte_probs) if byte_probs else 0
        
        # Sliding window entropy (window size 4)
        window_size = 4
        windows = [filter_str[i:i+window_size] for i in range(len(filter_str)-window_size+1)]
        window_counts = Counter(windows)
        window_probs = [count/len(windows) for count in window_counts.values()]
        features[f'sliding_window_entropy_filter_{idx}'] = -sum(p * math.log2(p) for p in window_probs) if window_probs else 0
        
        # Conditional entropy
        if len(filter_str) > 1:
            # Calculate joint and marginal probabilities
            pairs = list(zip(filter_str[:-1], filter_str[1:]))
            pair_counts = Counter(pairs)
            marginal_counts = Counter(filter_str[:-1])
            
            # Calculate conditional entropy
            cond_entropy = 0
            for pair, count in pair_counts.items():
                joint_prob = count / len(pairs)
                marginal_prob = marginal_counts[pair[0]] / len(filter_str[:-1])
                cond_entropy -= joint_prob * math.log2(joint_prob / marginal_prob)
                
            features[f'conditional_entropy_filter_{idx}'] = cond_entropy
    
    return features

def calculate_distribution_features(bitstring_data: str) -> Dict[str, float]:
    """
    Calculate distribution features from bitstring data.
    
    Args:
        bitstring_data: Comma-separated bitstrings
    Returns:
        Dictionary of distribution features
    """
    features = {}
    filters = bitstring_data.split(',')
    
    for idx, filter_str in enumerate(filters):
        if not filter_str:
            continue
            
        # Hamming weight distribution (in 8 segments)
        n_segments = 8
        segment_size = len(filter_str) // n_segments
        if segment_size > 0:
            segments = [filter_str[i:i+segment_size] for i in range(0, len(filter_str), segment_size)]
            segment_weights = [segment.count('1')/len(segment) for segment in segments if segment]
            features[f'hamming_weight_std_filter_{idx}'] = np.std(segment_weights)
            features[f'hamming_weight_range_filter_{idx}'] = max(segment_weights) - min(segment_weights)
        
        # Block density variation (10 blocks)
        n_blocks = 10
        block_size = len(filter_str) // n_blocks
        if block_size > 0:
            blocks = [filter_str[i:i+block_size] for i in range(0, len(filter_str), block_size)]
            block_densities = [block.count('1')/len(block) for block in blocks if block]
            features[f'block_density_std_filter_{idx}'] = np.std(block_densities)
            features[f'block_density_max_diff_filter_{idx}'] = max(block_densities) - min(block_densities)
        
        # Edge transition rate
        transitions = sum(1 for i in range(len(filter_str)-1) if filter_str[i] != filter_str[i+1])
        features[f'transition_rate_filter_{idx}'] = transitions / (len(filter_str)-1) if len(filter_str) > 1 else 0
        
        # Periodic pattern detection (autocorrelation)
        bits = np.array([int(b) for b in filter_str])
        acf = np.correlate(bits - bits.mean(), bits - bits.mean(), mode='full') / (bits.var() * len(bits))
        acf = acf[len(bits)-1:]
        features[f'autocorrelation_lag1_filter_{idx}'] = acf[1] if len(acf) > 1 else 0
        features[f'autocorrelation_lag2_filter_{idx}'] = acf[2] if len(acf) > 2 else 0
        
        # Local vs global density ratios
        global_density = filter_str.count('1') / len(filter_str)
        window_size = len(filter_str) // 4
        if window_size > 0:
            local_densities = [filter_str[i:i+window_size].count('1') / window_size 
                             for i in range(0, len(filter_str)-window_size+1, window_size)]
            features[f'local_global_density_ratio_std_filter_{idx}'] = np.std([d/global_density for d in local_densities]) if global_density else 0
    
    return features

def calculate_statistical_features(bitstring_data: str) -> Dict[str, float]:
    """
    Calculate statistical features from bitstring data.
    
    Args:
        bitstring_data: Comma-separated bitstrings
    Returns:
        Dictionary of statistical features
    """
    features = {}
    filters = bitstring_data.split(',')
    
    for idx, filter_str in enumerate(filters):
        if not filter_str:
            continue
            
        # Chi-square test statistic
        # Compare observed frequencies of 0s and 1s with expected uniform distribution
        observed = [filter_str.count('0'), filter_str.count('1')]
        expected = [len(filter_str)/2, len(filter_str)/2]
        chi_stat, p_value = stats.chisquare(observed, expected)
        features[f'chi_square_stat_filter_{idx}'] = chi_stat
        features[f'chi_square_pvalue_filter_{idx}'] = p_value
        
        # Kolmogorov complexity approximation using compression
        compressed = len(zlib.compress(filter_str.encode()))
        features[f'compression_ratio_filter_{idx}'] = compressed / len(filter_str)
        
        # Bit position bias
        # Divide string into quarters and compare 1s distribution
        quarters = np.array_split(list(map(int, filter_str)), 4)
        quarter_densities = [sum(quarter)/len(quarter) for quarter in quarters]
        features[f'position_bias_std_filter_{idx}'] = np.std(quarter_densities)
        features[f'position_bias_max_diff_filter_{idx}'] = max(quarter_densities) - min(quarter_densities)
        
        # Spectral entropy
        # Convert to numeric and calculate FFT
        bits = np.array([int(b) for b in filter_str])
        spectrum = np.abs(fft(bits))
        spectrum_norm = spectrum / sum(spectrum)
        spectral_entropy = -sum(p * np.log2(p) for p in spectrum_norm if p > 0)
        features[f'spectral_entropy_filter_{idx}'] = spectral_entropy
        
        # Information density gradient
        # Calculate entropy in sliding windows and measure how it changes
        window_size = min(32, len(filter_str)//4)
        if window_size > 0:
            windows = [filter_str[i:i+window_size] for i in range(0, len(filter_str)-window_size+1, window_size)]
            window_entropies = []
            for window in windows:
                counts = [window.count('0'), window.count('1')]
                probs = [c/len(window) for c in counts if c > 0]
                window_entropies.append(-sum(p * math.log2(p) for p in probs))
            
            if len(window_entropies) > 1:
                # Calculate gradient of entropy changes
                gradient = np.gradient(window_entropies)
                features[f'info_density_gradient_mean_filter_{idx}'] = np.mean(gradient)
                features[f'info_density_gradient_std_filter_{idx}'] = np.std(gradient)
    
    return features

def calculate_structural_features(bitstring_data: str) -> Dict[str, float]:
    """
    Calculate structural features from bitstring data.
    
    Args:
        bitstring_data: Comma-separated bitstrings
    Returns:
        Dictionary of structural features
    """
    features = {}
    filters = bitstring_data.split(',')
    
    # Filter size ratios
    filter_lengths = [len(f) for f in filters]
    for i in range(len(filter_lengths)-1):
        if filter_lengths[i] > 0:
            ratio = filter_lengths[i+1] / filter_lengths[i]
            features[f'size_ratio_filters_{i}_{i+1}'] = ratio
    
    # Inter-filter correlation
    for i, j in combinations(range(len(filters)), 2):
        filter1 = filters[i]
        filter2 = filters[j]
        # If filters have different lengths, truncate to shorter one
        min_length = min(len(filter1), len(filter2))
        if min_length > 0:
            bits1 = np.array([int(b) for b in filter1[:min_length]])
            bits2 = np.array([int(b) for b in filter2[:min_length]])
            correlation = np.corrcoef(bits1, bits2)[0, 1]
            features[f'filter_correlation_{i}_{j}'] = correlation
    
    # Process each filter individually
    for idx, filter_str in enumerate(filters):
        if not filter_str:
            continue
            
        # Position-weighted density
        # Weight decreases exponentially with position
        weights = np.exp(-np.arange(len(filter_str))/len(filter_str))
        bits = np.array([int(b) for b in filter_str])
        weighted_density = np.sum(bits * weights) / np.sum(weights)
        features[f'weighted_density_filter_{idx}'] = weighted_density
        
        # Symmetry measures
        mid = len(filter_str) // 2
        first_half = filter_str[:mid]
        second_half = filter_str[mid:2*mid]  # Ensure equal length
        if first_half:
            # Compare densities
            density_first = first_half.count('1') / len(first_half)
            density_second = second_half.count('1') / len(second_half)
            features[f'symmetry_density_ratio_filter_{idx}'] = density_first / density_second if density_second else 0
            
            # Compare patterns
            symmetry_score = sum(1 for i in range(len(first_half)) 
                               if first_half[i] == second_half[-(i+1)]) / len(first_half)
            features[f'symmetry_score_filter_{idx}'] = symmetry_score
        
        # Fractal dimension approximation using box-counting method
        box_sizes = [2, 4, 8, 16]  # Different box sizes to analyze
        box_counts = []
        for size in box_sizes:
            if len(filter_str) >= size:
                boxes = [filter_str[i:i+size] for i in range(0, len(filter_str), size)]
                unique_patterns = len(set(boxes))
                box_counts.append(unique_patterns)
        
        if len(box_counts) > 1 and all(c > 0 for c in box_counts):
            # Estimate fractal dimension from log-log plot
            x = np.log(box_sizes[:len(box_counts)])
            y = np.log(box_counts)
            slope, _, _, _, _ = stats.linregress(x, y)
            features[f'fractal_dimension_filter_{idx}'] = -slope
    
    return features

def process_file(input_file: str) -> pd.DataFrame:
    """
    Process a single CSV file and extract all features.
    
    Args:
        input_file: Path to input CSV file
    Returns:
        DataFrame with extracted features
    """
    # Read the input CSV file
    try:
        df = pd.read_csv(input_file, sep=';')
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
        
    print(f"\nProcessing file: {input_file}")
    print(f"Number of samples: {len(df)}\n")
    
    # Initialize list to store all processed rows
    processed_rows = []
    
    # Process each row
    for idx, row in df.iterrows():
        bitstring_data = row['concatenated_bitstrings']
        
        # Calculate all features
        features = {
            # Original metadata
            'num_included': row['num_included'],
            'num_excluded': row['num_excluded'],
            'duration': row['duration'],
            'tries': row['tries'],
            
            # Add calculated features
            **calculate_pattern_entropy_features(bitstring_data),
            **calculate_distribution_features(bitstring_data),
            **calculate_statistical_features(bitstring_data),
            **calculate_structural_features(bitstring_data)
        }
        
        processed_rows.append(features)
    
    return pd.DataFrame(processed_rows)

def print_feature_statistics(df: pd.DataFrame):
    """Print formatted statistics for each feature type."""
    
    def get_feature_stats(feature_names: List[str], df: pd.DataFrame) -> List[List]:
        """Calculate statistics for a group of features."""
        stats = []
        for feature in feature_names:
            values = df[feature].values
            stats.append([
                feature,
                f"{np.mean(values):.4f}",
                f"{np.std(values):.4f}",
                f"{np.min(values):.4f}",
                f"{np.max(values):.4f}"
            ])
        return stats
    
    # Group features by type
    pattern_features = [col for col in df.columns if 'entropy' in col]
    distribution_features = [col for col in df.columns if any(x in col for x in ['hamming', 'density', 'transition'])]
    statistical_features = [col for col in df.columns if any(x in col for x in ['chi', 'compression', 'spectral'])]
    structural_features = [col for col in df.columns if any(x in col for x in ['size_ratio', 'correlation', 'symmetry'])]
    
    # Print statistics for each feature group
    feature_groups = {
        'Pattern Entropy Features': pattern_features,
        'Distribution Features': distribution_features,
        'Statistical Features': statistical_features,
        'Structural Features': structural_features
    }
    
    for group_name, features in feature_groups.items():
        print(f"\n{group_name}:")
        print(tabulate(
            get_feature_stats(features, df),
            headers=['Feature', 'Mean', 'Std', 'Min', 'Max'],
            tablefmt='grid'
        ))
        print()

def main():
    # Hardcoded input and output paths
    input_file = "./data/single/single-data-1730297003111264000.csv"
    output_file = "./data/feature_data/sample_processed.csv"  # Replace with your output file path
    
    try:
        # Process file
        result_df = process_file(input_file)
        
        # Print statistics
        print_feature_statistics(result_df)
        
        # Save results
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_file, index=False)
        print(f"\nSaved processed data to: {output_file}")
            
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()