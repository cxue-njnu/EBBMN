clip_dir = r"F:/Train_CHB_MIT/sliced/clipLen12_timeStepSize1"
# save_dir = r"F:/Train_CHB_MIT/savew"
seq_len = 12
time_step = 1
init_lr = 1e-4
l2_weight_decay = 5e-4
max_grad_norm = 5.0
rnn_layers = 2
rnn_units = 64
diffusion_step = 2
input_dim = 128
num_nodes = 18
num_workers = 6
dropout_rate = 0.0
train_batch_size = 128
test_batch_size = 256

eval_interval = 2  # evaluate per n epochs
early_stopping_patience = 5  # stop training after n epochs without loss decreasing
task = "detection"  # or "classification"
metric = "auroc"  # or "f1" or "acc" or "loss"
classes = 1  # 1 if detection, n if classification
global_graph_mat = r"data\global_mat.pkl"
csv_data_path = r"F:\Train_CHB_MIT\loocv_csv\chb01_leave5"
save_dir = csv_data_path + "/save"


is_train = True
num_epochs = 2
