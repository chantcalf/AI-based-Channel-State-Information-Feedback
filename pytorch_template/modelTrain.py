# =======================================================================================================================
# =======================================================================================================================
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio

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
EPOCHS = 100
LEARNING_RATE = 1e-3
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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                                           pin_memory=True)
test_dataset = DatasetFolder(data_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                                          pin_memory=True)
# =======================================================================================================================
# =======================================================================================================================
# Model Constructing
autoencoderModel = AutoEncoder(NUM_FEEDBACK_BITS)
autoencoderModel = autoencoderModel.cuda()
criterion = nn.MSELoss().cuda()
# optimizer = torch.optim.Adam(autoencoderModel.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.Adam(autoencoderModel.parameters())
# =======================================================================================================================
# =======================================================================================================================
# Model Training and Saving
bestLoss = 1
for epoch in range(EPOCHS):
    autoencoderModel.train()
    for i, autoencoderInput in enumerate(train_loader):
        autoencoderInput = autoencoderInput.cuda()
        autoencoderOutput = autoencoderModel(autoencoderInput)
        loss = criterion(autoencoderOutput, autoencoderInput)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % PRINT_RREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss:.4f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))
    # Model Evaluating
    autoencoderModel.eval()
    totalLoss = 0
    with torch.no_grad():
        for i, autoencoderInput in enumerate(test_loader):
            autoencoderInput = autoencoderInput.cuda()
            autoencoderOutput = autoencoderModel(autoencoderInput)
            totalLoss += criterion(autoencoderOutput, autoencoderInput).item() * autoencoderInput.size(0)
        averageLoss = totalLoss / len(test_dataset)
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
print('Training for ' + str(NUM_FEEDBACK_BITS) + ' bits is finished!')
# =======================================================================================================================
# =======================================================================================================================
