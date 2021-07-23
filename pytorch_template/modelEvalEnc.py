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
# =======================================================================================================================
# =======================================================================================================================
# Data Loading
mat = sio.loadmat('channelData/W_test.mat')
data = mat['W_test']
data = np.reshape(data, (NUM_SAMPLES, CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
data = np.reshape(data, [NUM_SAMPLES, -1])
data = data.astype('float32')
data_test = data
test_dataset = DatasetFolder(data_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
# =======================================================================================================================
# =======================================================================================================================
# Model Loading
from modelDesign_48 import *

autoencoderModel1 = AutoEncoder(NUM_FEEDBACK_BITS1).cuda()
model_encoder1 = autoencoderModel1.encoder
model_encoder1.load_state_dict(torch.load('./modelSubmit/encoder_48.pth.tar')['state_dict'])
from modelDesign_128 import *

autoencoderModel2 = AutoEncoder(NUM_FEEDBACK_BITS2).cuda()
model_encoder2 = autoencoderModel2.encoder
model_encoder2.load_state_dict(torch.load('./modelSubmit/encoder_128.pth.tar')['state_dict'])
print("weight loaded")
# =======================================================================================================================
# =======================================================================================================================
# Encoding
model_encoder1.eval()
model_encoder2.eval()
encode_feature1 = []
encode_feature2 = []
with torch.no_grad():
    for i, autoencoderInput in enumerate(test_loader):
        autoencoderInput = autoencoderInput.cuda()
        autoencoderOutput1 = model_encoder1(autoencoderInput)
        autoencoderOutput1 = autoencoderOutput1.cpu().numpy()
        autoencoderOutput2 = model_encoder2(autoencoderInput)
        autoencoderOutput2 = autoencoderOutput2.cpu().numpy()
        if i == 0:
            encode_feature1 = autoencoderOutput1
            encode_feature2 = autoencoderOutput2
        else:
            encode_feature1 = np.concatenate((encode_feature1, autoencoderOutput1), axis=0)
            encode_feature2 = np.concatenate((encode_feature2, autoencoderOutput2), axis=0)

if encode_feature1.ndim != 2 or encode_feature2.ndim != 2:
    print("Invalid dimension of feedback bits sequence")
elif np.all(np.multiply(encode_feature1, encode_feature1) != encode_feature1) or np.all(
        np.multiply(encode_feature2, encode_feature2) != encode_feature2):
    print("Invalid form of feedback bits sequence")
elif np.shape(encode_feature1)[-1] != 48 or np.shape(encode_feature2)[-1] != 128:
    print("Invalid length of feedback bits sequence")
else:
    print(
        "Feedback bits length is " + str(np.shape(encode_feature1)[-1]) + " and " + str(np.shape(encode_feature2)[-1]))
    np.save('./encOutput_48.npy', encode_feature1)
    np.save('./encOutput_128.npy', encode_feature2)
    print('Finished!')
# =======================================================================================================================
# =======================================================================================================================
