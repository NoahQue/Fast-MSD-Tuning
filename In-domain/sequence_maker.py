import itertools
import csv
import random  # 导入 random 模块

# 假设任务列表是从0到8
tasks = list(range(9))  # [0, 1, 2, 3, 4, 5, 6, 7, 8]

# 生成所有可能的任务序列，长度为4
sequence_length = 4
all_sequences = list(itertools.permutations(tasks, sequence_length))

# 打乱任务序列顺序
random.shuffle(all_sequences)

# 打印总的任务序列数
print(f"总任务序列数（长度为{sequence_length}）：{len(all_sequences)}")

# 打印前10个任务序列作为示例
print("前10个任务序列示例：")
for seq in all_sequences[:10]:
    print(seq)

# 将所有任务序列保存到CSV文件
output_file = './gain/sampled_sequences_all_len4.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(["Task Sequence"])
    # 写入每个任务序列
    for seq in all_sequences:
        # 将元组转换为以逗号分隔的字符串
        writer.writerow([",".join(map(str, seq))])

print(f"所有长度为{sequence_length}的任务序列已保存到 {output_file} 文件中。")
