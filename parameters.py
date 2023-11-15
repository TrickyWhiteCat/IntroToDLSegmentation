import torch

INPUT_DIR = f"/kaggle/input/bkai-igh-neopolyp"
OUTPUT_DIR = "/kaggle/working/out"
TRAIN_DIR = f"{INPUT_DIR}/train/train"
TEST_DIR = f"{INPUT_DIR}/test/test"
CHECKPOINT_DIR = r"/kaggle/input/scratch-polyp-checkpoint"
CHECKPOINT_PATH = r"scratch_2023-11-15 00_34_31.07661207_00_40.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 10**-3
end_learning_rate = 10**-5
num_epochs = 50
batch_size = 2
img_size = (736, 960)
pretrain = "scratch"
loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.45, 0.45, 0.1], device=device))