def normalize_data(features):
    """
    Normalize the audio features to have zero mean and unit variance.
    
    Args:
        features (list or np.array): The audio features to normalize.
        
    Returns:
        np.array: The normalized audio features.
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    normalized_features = (features - mean) / std
    return normalized_features