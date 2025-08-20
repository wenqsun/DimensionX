#!/bin/bash

scene_type="real10k"
scene_name="0b79ada01eb45be9"

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

mkdir -p "$destination_folder"
mkdir -p "$original_folder"

files=("$source_folder"/*)
total_files=${#files[@]}

if [ "$total_files" -lt 3 ]; then
  echo "File not enough"
  exit 1
fi

cp "${files[0]}" "$destination_folder/1.png"

cp "${files[0]}" "$original_folder/${files[0]##*/}"

cp "${files[$total_files-1]}" "$destination_folder/$(($total_files-1)).png"
cp "${files[$total_files-1]}" "$original_folder/${files[$total_files-1]##*/}"

if [ "$total_files" -le 14 ]; then
  counter=2
  for ((i=1; i<$total_files-1; i++)); do
    cp "${files[i]}" "$destination_folder/$counter.png"
    cp "${files[i]}" "$original_folder/${files[i]##*/}"  
    ((counter++))
  done
else
  interval=$(( (total_files - 2) / 13 ))
  counter=2
  for ((i=1; i<13; i++)); do
    index=$(( i * interval ))
    cp "${files[index]}" "$destination_folder/$counter.png"
    cp "${files[index]}" "$original_folder/${files[index]##*/}"  
    ((counter++))
  done
fi

mv "$destination_folder/$(($total_files-1)).png" "$destination_folder/$counter.png"

mkdir -p "$instantsplat_folder"
cp "$destination_folder"/*.png "$instantsplat_folder"