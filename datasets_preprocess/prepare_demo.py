import os
import shutil
import re
from pathlib import Path
from tqdm import tqdm
import argparse


def organize_files_by_prefix(input_dir, output_base_dir=None, copy_mode=True, dry_run=False):
    """
    根据文件名中的前缀（如1_1）将文件分类到不同文件夹

    Args:
        input_dir: 输入目录，包含要分类的文件
        output_base_dir: 输出基础目录，如果为None则在input_dir下创建organized文件夹
        copy_mode: True为复制文件，False为移动文件
        dry_run: 只显示将要执行的操作，不实际执行
    """
    # 设置输出目录
    if output_base_dir is None:
        output_base_dir = os.path.join(input_dir, 'organized')

    # 确保输入目录存在
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    # 创建输出目录
    output_path = Path(output_base_dir)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    # 统计信息
    stats = {
        'total_files': 0,
        'organized_files': 0,
        'skipped_files': 0,
        'folders_created': set()
    }

    print(f"扫描目录: {input_dir}")
    print(f"输出目录: {output_base_dir}")
    print(f"模式: {'复制' if copy_mode else '移动'}")
    print(f"干运行: {dry_run}")
    print("-" * 50)

    # 获取所有文件
    files = []
    for root, dirs, filenames in os.walk(input_dir):
        # 跳过输出目录（如果它在输入目录内）
        if str(output_path) in root:
            continue

        for filename in filenames:
            file_path = os.path.join(root, filename)
            files.append(file_path)

    stats['total_files'] = len(files)

    if not files:
        print("未找到任何文件")
        return

    # 处理每个文件
    for file_path in tqdm(files, desc="处理文件"):
        filename = os.path.basename(file_path)

        # 检查文件扩展名
        file_ext = Path(filename).suffix.lower()
        if file_ext not in image_extensions:
            if dry_run:
                print(f"跳过非图片文件: {filename}")
            stats['skipped_files'] += 1
            continue

        # 尝试多种模式匹配文件名
        prefix = None

        # 模式1: 1_1_frame_data000006.png
        match1 = re.match(r'^(\d+_\d+)_frame_data\d+\.', filename)
        if match1:
            prefix = match1.group(1)

        # 模式2: 1-1_frame_data000006.png (使用连字符)
        if not prefix:
            match2 = re.match(r'^(\d+-\d+)_frame_data\d+\.', filename)
            if match2:
                prefix = match2.group(1)

        # 模式3: 1_1_other_pattern.png (通用模式)
        if not prefix:
            match3 = re.match(r'^(\d+_\d+)_', filename)
            if match3:
                prefix = match3.group(1)

        # 模式4: 1-1_other_pattern.png (通用连字符模式)
        if not prefix:
            match4 = re.match(r'^(\d+-\d+)_', filename)
            if match4:
                prefix = match4.group(1)

        # 如果无法提取前缀，跳过文件
        if not prefix:
            if dry_run:
                print(f"无法提取前缀，跳过: {filename}")
            stats['skipped_files'] += 1
            continue

        # 创建目标文件夹
        target_dir = output_path / prefix
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)

        stats['folders_created'].add(prefix)

        # 构建目标文件路径
        target_file = target_dir / filename

        # 执行复制或移动操作
        try:
            if dry_run:
                action = "复制到" if copy_mode else "移动到"
                print(f"[干运行] {action}: {filename} -> {target_dir}/")
            else:
                if copy_mode:
                    shutil.copy2(file_path, target_file)
                else:
                    shutil.move(file_path, target_file)

            stats['organized_files'] += 1

        except Exception as e:
            print(f"错误处理文件 {filename}: {e}")
            stats['skipped_files'] += 1

    # 打印统计信息
    print("\n" + "=" * 50)
    print("处理完成！")
    print("=" * 50)
    print(f"总文件数: {stats['total_files']}")
    print(f"已整理文件: {stats['organized_files']}")
    print(f"跳过文件: {stats['skipped_files']}")
    print(f"创建的文件夹数: {len(stats['folders_created'])}")

    if stats['folders_created']:
        print("\n创建的文件夹:")
        for folder in sorted(stats['folders_created']):
            print(f"  - {folder}")

    if dry_run:
        print("\n注意: 这是干运行模式，没有实际执行任何操作。")
        print("要实际执行，请使用 --no-dry-run 参数。")


def main():
    parser = argparse.ArgumentParser(description='根据文件名前缀整理文件到不同文件夹')
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('-o', '--output-dir', help='输出目录路径（默认：input_dir/organized）')
    parser.add_argument('-m', '--move', action='store_true',
                        help='移动文件而不是复制（默认：复制）')
    parser.add_argument('-d', '--dry-run', action='store_true',
                        help='只显示将要执行的操作，不实际执行')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示详细信息')

    args = parser.parse_args()

    organize_files_by_prefix(
        input_dir=args.input_dir,
        output_base_dir=args.output_dir,
        copy_mode=not args.move,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()