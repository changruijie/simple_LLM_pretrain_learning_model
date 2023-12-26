from tokenizer.tokenization import LLMTokenizer
from datasets import load_dataset
import pickle
import os
import time


def process_firefly(data_dir, name):
    dataset_dict = load_dataset('YeungNLP/firefly-train-1.1M')
    # 获取训练集的可迭代对象
    train_iter = iter(dataset_dict['train'])
    # 拼接文件目录
    save_dir = os.path.join(data_dir, 'binary_{}'.format(name))

    # 创建目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获得实际文件名
    filename = "pretrain_data.bin"
    file_path = os.path.join(save_dir, filename)

    with open(file_path, 'wb') as binary_file:
        for sample in train_iter:
            # 处理当前行的数据
            kind_value = sample['kind']
            input_value = sample['input']
            target_value = sample['target']

            kind_id = tokenizer.encode(kind_value, add_special_tokens=False)
            kind_id.append(tokenizer.special_tokens['<eos>'])
            input_id = tokenizer.encode(input_value, add_special_tokens=False)
            input_id.append(tokenizer.special_tokens['<eos>'])
            target_id = tokenizer.encode(target_value, add_special_tokens=False)
            target_id.append(tokenizer.special_tokens['<eos>'])

            # 将数据组织成字典或其他合适的数据结构
            data_to_store = {
                'kind': kind_id,
                'input': input_id,
                'target': target_id
            }

            # 使用pickle将数据序列化并写入二进制文件
            pickle.dump(data_to_store, binary_file)


if __name__ == "__main__":
    tokenizer = LLMTokenizer(vocab_file='./tokenizer/tokenizer.model')
    data_dir = 'data'
    process_firefly(data_dir, 'firefly')

