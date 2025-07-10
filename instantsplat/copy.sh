#!/bin/bash

# 定义源文件夹和目标文件夹
scene_type="real10k"
scene_name="0b79ada01eb45be9"

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --scene_type) scene_type="$2"; shift ;;
        --scene_name) scene_name="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

source_folder="/pfs/mt-1oY5F7/ss-3d/realestate10k/dataset_clips/test/$scene_name"
destination_folder="/pfs/mt-1oY5F7/sunwenqiang/3d/3dgs/wild-gaussian-splatting/data/images/test_case/$scene_type/$scene_name"
instantsplat_folder="/pfs/mt-1oY5F7/sunwenqiang/3d/3dgs/wild-gaussian-splatting/data/images/instantsplat/$scene_type/$scene_name/$scene_name"
original_folder="${destination_folder}_original"

# 创建目标文件夹如果它不存在
mkdir -p "$destination_folder"
mkdir -p "$original_folder"

# 获取所有文件的列表
files=("$source_folder"/*)
total_files=${#files[@]}

# 确保有足够的文件进行操作
if [ "$total_files" -lt 3 ]; then
  echo "源文件夹中的文件不足。"
  exit 1
fi

# 复制并重命名第一张图片
cp "${files[0]}" "$destination_folder/1.png"

# 复制第一张图片不更改名字
cp "${files[0]}" "$original_folder/${files[0]##*/}"

# 复制并重命名最后一张图片
cp "${files[$total_files-1]}" "$destination_folder/$(($total_files-1)).png"
cp "${files[$total_files-1]}" "$original_folder/${files[$total_files-1]##*/}"

# 如果中间文件少于12个，全部复制并重命名
if [ "$total_files" -le 14 ]; then
  counter=2
  for ((i=1; i<$total_files-1; i++)); do
    cp "${files[i]}" "$destination_folder/$counter.png"
    cp "${files[i]}" "$original_folder/${files[i]##*/}"  # 保留原文件名
    ((counter++))
  done
else
  # 平均选择12个中间文件并重命名
  interval=$(( (total_files - 2) / 13 ))
  counter=2
  for ((i=1; i<13; i++)); do
    index=$(( i * interval ))
    cp "${files[index]}" "$destination_folder/$counter.png"
    cp "${files[index]}" "$original_folder/${files[index]##*/}"  # 保留原文件名
    ((counter++))
  done
fi

# 重命名最后一张图片
mv "$destination_folder/$(($total_files-1)).png" "$destination_folder/$counter.png"

echo "文件复制完成！"

# 将这个文件夹的第一张和最后一张复制到instant_splat文件夹
mkdir -p "$instantsplat_folder"

# 将这个文件夹的所有文件复制到instant_splat文件夹
cp "$destination_folder"/*.png "$instantsplat_folder"

# cp "$destination_folder/1.png" "$instantsplat_folder/1.png"
# cp "$destination_folder/$counter.png" "$instantsplat_folder/2.png"


# # 将destination_folder中的第一张和最后一张删除并重命名
# rm "$destination_folder/1.png"
# rm "$destination_folder/$counter.png"

# # 用循环来重命名
# for ((i=2; i<13; i++)); do
#   mv "$destination_folder/$i.png" "$destination_folder/$(($i-1)).png"
# done
