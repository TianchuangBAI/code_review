def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = [line.strip().upper() for line in f1 if line.strip()]
        lines2 = [line.strip().upper() for line in f2 if line.strip()]
    
    if len(lines1) != len(lines2):
        print(f"警告：文件行数不同（{len(lines1)} vs {len(lines2)}），将以较短的文件行数为准")
    
    min_len = min(len(lines1), len(lines2))
    matches = []
    
    for i in range(min_len):
        if lines1[i] == lines2[i]:
            matches.append(1)
        else:
            matches.append(0)
    
    avg = sum(matches) / min_len if min_len > 0 else 0
    return matches, avg

# 使用示例
matches, avg_score = compare_files('answers.txt', 'answers1.txt')
print(f"逐行比较结果（1=相同，0=不同）: {matches}")
print(f"平均匹配率: {avg_score:.2%}")  # 以百分比形式输出