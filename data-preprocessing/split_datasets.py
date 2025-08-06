import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
import os

def analyze_dataset(file_path):
    """Analyze a dataset to count PD positive and negative samples."""
    pd_positive = []
    pd_negative = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                assistant_content = data['messages'][-1]['content'][0]['text']
                
                if "Detected tremor and hesitations consistent with Parkinson's symptoms" in assistant_content:
                    pd_positive.append(line.strip())  # Store the full line, not tuple
                elif "No significant signs of Parkinson's detected" in assistant_content or "No Parkinson's detected. Speech patterns appear normal" in assistant_content:
                    pd_negative.append(line.strip())  # Store the full line, not tuple
                else:
                    print(f"Unknown response pattern in {file_path}, line {line_num}: {assistant_content[:100]}...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in {file_path}, line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing {file_path}, line {line_num}: {e}")
    
    return pd_positive, pd_negative

def create_balanced_split(pd_positive, pd_negative, test_size=0.2, random_state=42):
    """Create balanced train/test split maintaining class distribution."""
    
    # Handle case where one class is empty
    if len(pd_positive) == 0:
        print("Warning: No PD positive samples found!")
        return [], [], pd_negative, []
    elif len(pd_negative) == 0:
        print("Warning: No PD negative samples found!")
        return pd_positive, [], [], []
    
    # Split each class separately
    pos_train, pos_test = train_test_split(
        pd_positive, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    neg_train, neg_test = train_test_split(
        pd_negative, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    return pos_train, pos_test, neg_train, neg_test

def write_split_data(train_data, test_data, output_dir, dataset_name):
    """Write train and test data to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Write training data
    train_file = os.path.join(output_dir, f"{dataset_name}_train.jsonl")
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line + '\n')
    
    # Write test data
    test_file = os.path.join(output_dir, f"{dataset_name}_test.jsonl")
    with open(test_file, 'w', encoding='utf-8') as f:
        for line in test_data:
            f.write(line + '\n')
    
    return train_file, test_file

def main():
    # Dataset paths
    datasets = {
        'processed_dataset': 'processed_dataset/messages_dataset.jsonl',
        'processed_italian_dataset': 'processed_italian_dataset/messages_dataset.jsonl',
        'processed_sjtu_denoised': 'processed_sjtu_denoised/messages_dataset.jsonl'
    }
    
    # Combined data for final split
    all_positive = []
    all_negative = []
    
    # Analyze each dataset
    for dataset_name, file_path in datasets.items():
        print(f"\n=== Analyzing {dataset_name} ===")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        pd_positive, pd_negative = analyze_dataset(file_path)
        
        print(f"PD Positive samples: {len(pd_positive)}")
        print(f"PD Negative samples: {len(pd_negative)}")
        print(f"Total samples: {len(pd_positive) + len(pd_negative)}")
        
        # Add to combined data
        all_positive.extend(pd_positive)
        all_negative.extend(pd_negative)
    
    print(f"\n=== Combined Dataset Statistics ===")
    print(f"Total PD Positive samples: {len(all_positive)}")
    print(f"Total PD Negative samples: {len(all_negative)}")
    print(f"Total samples: {len(all_positive) + len(all_negative)}")
    
    # If we have both classes, create balanced split
    if len(all_positive) > 0 and len(all_negative) > 0:
        print(f"\n=== Creating Balanced Train/Test Split ===")
        pos_train, pos_test, neg_train, neg_test = create_balanced_split(
            all_positive, all_negative, test_size=0.2, random_state=42
        )
        
        # Combine train and test data
        train_data = pos_train + neg_train
        test_data = pos_test + neg_test
        
        # Shuffle the combined data
        random.seed(42)
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        
        # Write split data
        output_dir = "split_datasets"
        train_file, test_file = write_split_data(train_data, test_data, output_dir, "combined")
        
        print(f"\n=== Split Complete ===")
        print(f"Training data written to: {train_file}")
        print(f"Testing data written to: {test_file}")
        
        # Verify the split maintains balance
        print(f"\n=== Verifying Split Balance ===")
        
        def count_classes(data):
            pos_count = 0
            neg_count = 0
            for line in data:
                if "Detected tremor and hesitations consistent with Parkinson's symptoms" in line:
                    pos_count += 1
                elif "No significant signs of Parkinson's detected" in line or "No Parkinson's detected. Speech patterns appear normal" in line:
                    neg_count += 1
            return pos_count, neg_count
        
        train_pos, train_neg = count_classes(train_data)
        test_pos, test_neg = count_classes(test_data)
        
        print(f"Training - PD Positive: {train_pos}, PD Negative: {train_neg}")
        print(f"Testing - PD Positive: {test_pos}, PD Negative: {test_neg}")
        
        # Calculate balance ratios
        train_ratio = train_pos / (train_pos + train_neg) if (train_pos + train_neg) > 0 else 0
        test_ratio = test_pos / (test_pos + test_neg) if (test_pos + test_neg) > 0 else 0
        
        print(f"Training PD Positive ratio: {train_ratio:.3f}")
        print(f"Testing PD Positive ratio: {test_ratio:.3f}")
        
        # Create the final finetune dataset (combining train and test)
        print(f"\n=== Creating Final Finetune Dataset ===")
        all_data = train_data + test_data
        random.shuffle(all_data)  # Shuffle again for final dataset
        
        finetune_file = "finetune_dataset.jsonl"
        with open(finetune_file, 'w', encoding='utf-8') as f:
            for line in all_data:
                f.write(line + '\n')
        
        print(f"Final finetune dataset written to: {finetune_file}")
        print(f"Total samples in finetune dataset: {len(all_data)}")
        
        # Final verification
        finetune_pos, finetune_neg = count_classes(all_data)
        finetune_ratio = finetune_pos / (finetune_pos + finetune_neg) if (finetune_pos + finetune_neg) > 0 else 0
        print(f"Finetune dataset - PD Positive: {finetune_pos}, PD Negative: {finetune_neg}")
        print(f"Finetune dataset PD Positive ratio: {finetune_ratio:.3f}")
    
    else:
        print(f"\n=== Creating Single-Class Dataset ===")
        # If we only have one class, just create a simple split
        all_data = all_positive + all_negative
        random.seed(42)
        random.shuffle(all_data)
        
        # Simple 80/20 split
        split_idx = int(len(all_data) * 0.8)
        train_data = all_data[:split_idx]
        test_data = all_data[split_idx:]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        
        # Write split data
        output_dir = "split_datasets"
        train_file, test_file = write_split_data(train_data, test_data, output_dir, "combined")
        
        print(f"\n=== Split Complete ===")
        print(f"Training data written to: {train_file}")
        print(f"Testing data written to: {test_file}")
        
        # Create the final finetune dataset
        print(f"\n=== Creating Final Finetune Dataset ===")
        finetune_file = "finetune_dataset.jsonl"
        with open(finetune_file, 'w', encoding='utf-8') as f:
            for line in all_data:
                f.write(line + '\n')
        
        print(f"Final finetune dataset written to: {finetune_file}")
        print(f"Total samples in finetune dataset: {len(all_data)}")

if __name__ == "__main__":
    main() 