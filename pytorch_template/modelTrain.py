# =======================================================================================================================
# =======================================================================================================================
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import time
from torch.optim.lr_scheduler import LambdaLR
import random

# =======================================================================================================================
# =======================================================================================================================
# Parameters Setting for Data
NUM_FEEDBACK_BITS = 48
if NUM_FEEDBACK_BITS == 128:
    from modelDesign_128 import *
elif NUM_FEEDBACK_BITS == 48:
    from modelDesign_48 import *
CHANNEL_SHAPE_DIM1 = 12
CHANNEL_SHAPE_DIM2 = 32
CHANNEL_SHAPE_DIM3 = 2
CHANNEL_SHAPE_DIM_TOTAL = CHANNEL_SHAPE_DIM1 * CHANNEL_SHAPE_DIM2 * CHANNEL_SHAPE_DIM3
NUM_SAMPLES = 600000
# Parameters Setting for Training
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-1
PRINT_RREQ = 100
torch.manual_seed(1)
# =======================================================================================================================
# =======================================================================================================================
# Data Loading
mat = sio.loadmat('channelData/W_train.mat')
data = mat['W_train']
data = np.reshape(data, (NUM_SAMPLES, CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
data = np.reshape(data, [NUM_SAMPLES, -1])
data = data.astype('float32')
split = int(data.shape[0] * 0.9)
data_train, data_test = data[:split], data[split:]
train_dataset = DatasetFolder(data_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                           pin_memory=True)
test_dataset = DatasetFolder(data_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                          pin_memory=True)
# =======================================================================================================================
# =======================================================================================================================

max_steps = EPOCHS * (len(data_train) // BATCH_SIZE)


def lr_scheduler(step, warm_up_step=1000, max_step=max_steps):
    if step < warm_up_step:
        return 1e-2 + (1 - 1e-2) * step / warm_up_step
    if step <= max_step * 0.4:
        return 1
    if step <= max_step * 0.9:
        return 0.1
    return 0.01


# Model Constructing
autoencoderModel = AutoEncoder(NUM_FEEDBACK_BITS)
autoencoderModel = autoencoderModel.cuda()
# criterion = nn.MSELoss().cuda()
criterion = MyLoss().cuda()
optimizer = torch.optim.AdamW(autoencoderModel.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_scheduler)
# optimizer = torch.optim.Adam(autoencoderModel.parameters())
# =======================================================================================================================
# =======================================================================================================================
if __name__ == '__main__':
    # Model Training and Saving
    start_time = time.time()
    bestLoss = 1
    ea = 2
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        autoencoderModel.train()
        quant = epoch >= ea
        for i, autoencoderInput in enumerate(train_loader):
            if ea <= epoch < EPOCHS * 0.9:
                quant = random.random() >= 0.1 * (EPOCHS - epoch) / (EPOCHS - ea)
            autoencoderInput = autoencoderInput.cuda()
            autoencoderOutput = autoencoderModel(autoencoderInput, quant)
            loss = criterion(autoencoderOutput, autoencoderInput)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % PRINT_RREQ == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t' 'Loss {loss:.4f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))
        # Model Evaluatin
        autoencoderModel.eval()
        totalLoss = 0
        with torch.no_grad():
            inputs = []
            outputs = []
            for i, autoencoderInput in enumerate(test_loader):
                autoencoderInput = autoencoderInput.cuda()
                autoencoderOutput = autoencoderModel(autoencoderInput)
                totalLoss += criterion(autoencoderOutput, autoencoderInput).item() * autoencoderInput.size(0)
                inputs.append(autoencoderInput)
                outputs.append(autoencoderOutput)
            averageLoss = totalLoss / len(test_dataset)
            inputs = torch.cat(inputs, 0).detach().cpu().numpy()
            outputs = torch.cat(outputs, 0).detach().cpu().numpy()
            print("averageLoss: {}".format(averageLoss))
            print("score", cal_score(inputs, outputs, len(inputs), 12))
            if averageLoss < bestLoss:
                # Model saving
                # Encoder Saving
                torch.save({'state_dict': autoencoderModel.encoder.state_dict(), },
                           './modelSubmit/encoder_' + str(NUM_FEEDBACK_BITS) + '.pth.tar')
                # Decoder Saving
                torch.save({'state_dict': autoencoderModel.decoder.state_dict(), },
                           './modelSubmit/decoder_' + str(NUM_FEEDBACK_BITS) + '.pth.tar')
                print("Model saved")
                bestLoss = averageLoss
        print("cost {}s for epoch {}".format(time.time() - epoch_start_time, epoch))
    print('Training for ' + str(NUM_FEEDBACK_BITS) + ' bits is finished!')
    print("total using {}s".format(time.time() - start_time))
    # =======================================================================================================================
    # =======================================================================================================================
