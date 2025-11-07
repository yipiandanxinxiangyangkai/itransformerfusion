import os
import csv
import json
from tqdm import tqdm # 一个好用的进度条库, pip install tqdm

# --- 1. 你的路径 (保持不变) ---
ROOT_NEWS_DIRECTORY = "D:\\下载解压\\stocknet-dataset-master\\tweet\\raw\\AAPL"
OUTPUT_CSV_FILE = 'D:\\下载解压\\stocknet-dataset-master\\processed.csv'

# --- 脚本主体 (已修改) ---

def process_news_files(root_dir, output_file):
    print(f"开始处理新闻数据，源目录: {root_dir}")
    
    # 收集所有有效的日期文件
    try:
        all_entries = os.listdir(root_dir)
        date_files = [f for f in all_entries if os.path.isfile(os.path.join(root_dir, f))]
        date_files = [f for f in date_files if f.count('-') == 2 and len(f) == 10]
        date_files.sort()
        
    except FileNotFoundError:
        print(f"错误: 找不到根目录 {root_dir}。请检查 ROOT_NEWS_DIRECTORY 变量。")
        return
    except Exception as e:
        print(f"读取目录时出错: {e}")
        return

    if not date_files:
        print(f"警告: 在 {root_dir} 中没有找到任何日期格式的文件。") 
        return

    print(f"找到了 {len(date_files)} 个日期文件。准备写入到 {output_file}...")

    # 打开CSV文件准备写入
    try:
        # --- (修改 1: 解决Excel乱码问题) ---
        # "utf-8-sig" 会在文件头部添加一个 BOM, 帮助Excel正确识别UTF-8
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f_out:
            writer = csv.writer(f_out)
            # 写入表头
            writer.writerow(['date', 'text'])

            # 使用tqdm显示进度条
            for date_filename in tqdm(date_files, desc="处理日期"): 
                current_date = date_filename 
                file_path = os.path.join(root_dir, date_filename)
                
                try:
                    # 打开并逐行读取文件
                    with open(file_path, 'r', encoding='utf-8') as f_in:
                        for line in f_in:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                # 解析单行JSON (一个tweet)
                                tweet_data = json.loads(line)
                                
                                # 提取 'text' 字段
                                if 'text' in tweet_data:
                                    text_content = tweet_data['text']
                                    
                                    # --- (修改 2: 删除 http 链接) ---
                                    # 1. 找到 'http:' 的位置并切片
                                    text_content = text_content.split('http:', 1)[0]
                                    # 2. 找到 'https:' 的位置并切片 (以防万一)
                                    text_content = text_content.split('https:', 1)[0]
                                    # 3. 清理切片后可能留下的多余空格
                                    text_content = text_content.strip()
                                    # --- 清理结束 ---
                                    
                                    # 只有在清理后仍然有内容时才写入
                                    if text_content:
                                        # 写入一行数据: [日期, 清理后的新闻内容]
                                        writer.writerow([current_date, text_content])
                                    
                            except json.JSONDecodeError:
                                pass # 跳过非 JSON 行
                            except Exception as e:
                                print(f"警告: 处理文件 {file_path} 中的一行时出错: {e}")

                except Exception as e:
                    print(f"错误: 无法读取或处理文件 {file_path}: {e}")

    except IOError:
        print(f"错误: 无法写入到 {output_file}。请检查权限或路径。")
    except Exception as e:
        print(f"发生未知错误: {e}")

    print(f"\n处理完成！数据已保存到 {output_file}")


# --- 运行脚本 ---
if __name__ == "__main__":
    # 确保你安装了 tqdm: pip install tqdm
    process_news_files(ROOT_NEWS_DIRECTORY, OUTPUT_CSV_FILE)