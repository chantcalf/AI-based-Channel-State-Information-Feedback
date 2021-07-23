# =======================================================================================================================
# =======================================================================================================================
import numpy as np
from modelDesign_48 import *
import torch
import scipy.io as sio

# =======================================================================================================================
# =======================================================================================================================
# Parameters Setting
NUM_FEEDBACK_BITS1 = 48
NUM_FEEDBACK_BITS2 = 128
CHANNEL_SHAPE_DIM1 = 12
CHANNEL_SHAPE_DIM2 = 32
CHANNEL_SHAPE_DIM3 = 2
CHANNEL_SHAPE_DIM_TOTAL = CHANNEL_SHAPE_DIM1 * CHANNEL_SHAPE_DIM2 * CHANNEL_SHAPE_DIM3
NUM_SAMPLES = 80000
NUM_SUBBAND = 12
# =======================================================================================================================
# =======================================================================================================================
# Data Loading
mat = sio.loadmat('channelData/W_test.mat')
data = mat['W_test']
data = np.reshape(data, (NUM_SAMPLES, CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
data = np.reshape(data, [NUM_SAMPLES, -1])
data = data.astype('float32')
data_test = data

encode_feature1 = np.load('./encOutput_48.npy')
encode_feature2 = np.load('./encOutput_128.npy')
test_dataset1 = DatasetFolder(encode_feature1)
test_dataset2 = DatasetFolder(encode_feature2)
test_loader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
# =======================================================================================================================
# =======================================================================================================================
# Model Loading and Decoding
from modelDesign_48 import *

autoencoderModel1 = AutoEncoder(NUM_FEEDBACK_BITS1).cuda()
model_decoder1 = autoencoderModel1.decoder
model_decoder1.load_state_dict(torch.load('./modelSubmit/decoder_48.pth.tar', map_location='cpu')['state_dict'])
from modelDesign_128 import *

autoencoderModel2 = AutoEncoder(NUM_FEEDBACK_BITS2).cuda()
model_decoder2 = autoencoderModel2.decoder
model_decoder2.load_state_dict(torch.load('./modelSubmit/decoder_128.pth.tar', map_location='cpu')['state_dict'])
print("weight loaded")
# =======================================================================================================================
# =======================================================================================================================
# Decoding
model_decoder1.eval()
model_decoder2.eval()
W_pre1 = []
W_pre2 = []
with torch.no_grad():
    for i, decoderOutput in enumerate(test_loader1):
        # convert numpy to Tensor
        decoderOutput = decoderOutput.cuda()
        output1 = model_decoder1(decoderOutput)
        output1 = output1.cpu().numpy()
        if i == 0:
            W_pre1 = output1
        else:
            W_pre1 = np.concatenate((W_pre1, output1), axis=0)

    for i, decoderOutput in enumerate(test_loader2):
        # convert numpy to Tensor
        decoderOutput = decoderOutput.cuda()
        output2 = model_decoder2(decoderOutput)
        output2 = output2.cpu().numpy()
        if i == 0:
            W_pre2 = output2
        else:
            W_pre2 = np.concatenate((W_pre2, output2), axis=0)
# =======================================================================================================================
# =======================================================================================================================
# Score Calculating
score1 = cal_score(data_test, W_pre1, NUM_SAMPLES, NUM_SUBBAND)
score2 = cal_score(data_test, W_pre2, NUM_SAMPLES, NUM_SUBBAND)
score = (score1 + score2) / 2
print('The score for 48 bits is ' + str(score1))
print('The score for 128 bits is ' + str(score2))
print('The final score is ' + str(score))
print('Finished!')
# =======================================================================================================================
# =======================================================================================================================
