import torch

INPUT_DIR = r"C:\Users\nmttu\Downloads\bkai-igh-neopolyp"
OUTPUT_DIR = "out"
TRAIN_DIR = f"{INPUT_DIR}/train/train"
TEST_DIR = f"{INPUT_DIR}/test/test"
CHECKPOINT_DIR = r"checkpoint"
CHECKPOINT_PATH = r"scratch_11152023_121350_5.pth"

num_workers = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 10**-3
end_learning_rate = 10*678*-5
num_epochs = 50
batch_size = 1
img_size = (736, 960)
pretrain = "scratch"
split_ratio = [0.8, 0.2]
loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.45, 0.45, 0.1], device=device))

report_step = 200