import itertools
import csv

# Assume the task list is from 0 to 8
tasks = list(range(6))  # [0, 1, 2, 3, 4, 5]

# Generate all possible task sequences of length 2
sequence_length = 2
all_sequences = list(itertools.permutations(tasks, sequence_length))

# Filter task sequences that end with 2
filtered_sequences = [seq for seq in all_sequences if seq[-1] == 2]

# Print the total number of task sequences
print(f"Total task sequences (length {sequence_length}, ending with 2): {len(filtered_sequences)}")

# Print the first 10 task sequences as examples
print("First 10 task sequence examples:")
for seq in filtered_sequences[:10]:
    print(seq)

# Save all task sequences to a CSV file
output_file = './gain/sampled_sequences_filtered_len2_end2.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Task Sequence"])
    # Write each task sequence
    for seq in filtered_sequences:
        # Convert the tuple to a comma-separated string
        writer.writerow([",".join(map(str, seq))])

print(f"All task sequences of length {sequence_length} and ending with 2 have been saved to {output_file} file.")
