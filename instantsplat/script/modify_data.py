import os
import argparse

import os

def modify_files_for_train_test(dataset):
    base_path = "data/scenes"
    train_path = os.path.join(base_path, f"{dataset}_train", "sparse/0")
    test_path = os.path.join(base_path, f"{dataset}_test", "sparse/0")

    # 修改_train文件夹下的文件
    for file_name in ["images.txt", "cameras.txt"]:
        file_path = os.path.join(train_path, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # 保留头部注释部分
        header = [line for line in lines if line.startswith("#")]
        data_lines = [line for line in lines if not line.startswith("#") and line.strip()]

        # 删除从倒数第13行到倒数第2行的数据
        # modified_data_lines = data_lines[:-13] + data_lines[-1:]
        # modified_data_lines = data_lines[:-12] # + data_lines[-1:]
        modified_data_lines = data_lines

        modified_lines = header + modified_data_lines

        with open(file_path, 'w') as file:
            for line in modified_lines:
                file.write(line)
                file.write("\n")

    # 修改_test文件夹下的文件
    for file_name in ["images.txt", "cameras.txt"]:
        file_path = os.path.join(test_path, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # 保留头部注释部分
        header = [line for line in lines if line.startswith("#")]
        data_lines = [line for line in lines if not line.startswith("#") and line.strip()]

        # 只保留从倒数第13行到倒数第2行的数据
        # modified_data_lines = data_lines[-13:-1]
        # modified_data_lines = data_lines[-12:]
        modified_data_lines = data_lines

        modified_lines = header + modified_data_lines

        with open(file_path, 'w') as file:
            for line in modified_lines:
                file.write(line)
                file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify files for train and test.")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    args = parser.parse_args()
    modify_files_for_train_test(args.dataset)